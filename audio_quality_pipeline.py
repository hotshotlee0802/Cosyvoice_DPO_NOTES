"""Audio quality and ASR evaluation pipeline with flexible file mapping."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from hashlib import md5
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

try:  # SpeechScore is an optional runtime dependency for tests
    from speechscore import SpeechScore as _SpeechScore
except ImportError:  # pragma: no cover - exercised only when dependency missing
    _SpeechScore = None

# FunASRCERCalculator imports many heavy dependencies; load lazily when needed
_FUNASR_CALCULATOR = None


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
_VARIANT_SUFFIX = re.compile(r"^(?P<base>.+?)_(?P<index>\d+)$")


def strip_variant_suffix(stem: str) -> str:
    """Return the canonical base name by removing trailing numeric variants."""

    match = _VARIANT_SUFFIX.match(stem)
    return match.group("base") if match else stem


def collect_audio_files(path: str) -> List[str]:
    """Collect audio files from ``path``.

    When ``path`` is a directory, recursively walk the tree to find supported
    audio extensions. When it's a file, return a single-item list.
    """

    p = Path(path)
    if p.is_file():
        return [str(p.resolve())]

    files: List[str] = []
    for candidate in sorted(p.rglob("*")):
        if candidate.suffix.lower() in AUDIO_EXTENSIONS and candidate.is_file():
            files.append(str(candidate.resolve()))
    return files


def parse_transcript_file(transcript_path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Parse ``transcript_path`` in ``filepath|speaker|lang|text`` format."""

    mapping: Dict[str, Dict[str, str]] = {}
    if not transcript_path:
        return mapping

    with open(transcript_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid transcript format at line {line_number}: {raw_line!r}"
                )

            filepath, speaker, lang_id, text = parts[:4]
            basename = Path(filepath).stem
            base_key = strip_variant_suffix(basename)
            mapping[base_key] = {
                "text": text.strip(),
                "filepath": filepath.strip(),
                "speaker": speaker.strip(),
                "lang_id": lang_id.strip(),
            }
    return mapping


def build_reference_lookup(
    test_files: Iterable[str],
    reference_dir: Optional[str],
    transcripts: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Optional[str]]]:
    """Build lookup for reference audio and text per test file.

    The mapping ignores numeric suffices like ``*_0001.wav`` while matching to
    reference audio in ``reference_dir`` and transcripts.
    """

    lookup: Dict[str, Dict[str, Optional[str]]] = {}
    reference_path = Path(reference_dir) if reference_dir else None

    for test_file in test_files:
        test_path = Path(test_file)
        base_key = strip_variant_suffix(test_path.stem)
        transcript_entry = transcripts.get(base_key)

        reference_candidate: Optional[Path] = None
        if reference_path:
            if transcript_entry:
                # Honour the exact filename described in transcripts
                reference_candidate = reference_path / Path(transcript_entry["filepath"]).name
                if not reference_candidate.exists():
                    reference_candidate = None
            if reference_candidate is None:
                pattern = f"{base_key}*"
                matches = sorted(reference_path.glob(pattern))
                reference_candidate = matches[0] if matches else None

        lookup[str(test_path)] = {
            "reference": str(reference_candidate) if reference_candidate else None,
            "transcript": transcript_entry["text"] if transcript_entry else None,
            "base_key": base_key,
        }
    return lookup


class AudioQualityEvaluator:
    """SpeechScore-based audio quality evaluation helper."""

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        preload_models: bool = True,
        use_cache: bool = True,
        cache_dir: str = ".cache",
    ) -> None:
        if _SpeechScore is None:  # pragma: no cover - dependency missing in tests
            raise ImportError(
                "SpeechScore is required to instantiate AudioQualityEvaluator."
            )

        if metrics is None:
            metrics = [
                "SRMR",
                "PESQ",
                "NB_PESQ",
                "STOI",
                "SISDR",
                "FWSEGSNR",
                "LSD",
                "BSSEval",
                "DNSMOS",
                "SNR",
                "SSNR",
                "LLR",
                "CSIG",
                "CBAK",
                "COVL",
                "MCD",
            ]

        self.metrics = metrics
        self.speech_score = _SpeechScore(metrics)
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        if use_cache:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        if preload_models and "DNSMOS" in metrics:
            try:  # pragma: no cover - heavy dependency
                import numpy as np

                dummy_audio = np.zeros(16000, dtype=np.float32)
                try:
                    self.speech_score._preload_dnsmos_model(dummy_audio)
                except AttributeError:
                    self.speech_score(test_path=dummy_audio, reference_path=None)
            except Exception:
                print("预加载DNSMOS模型失败，将在首次使用时加载")

    # Cache helpers -----------------------------------------------------
    def _cache_key(
        self,
        test_path: str,
        reference_path: Optional[str],
        window: Optional[float],
        score_rate: int,
    ) -> str:
        test_mtime = os.path.getmtime(test_path) if os.path.exists(test_path) else 0
        ref_mtime = (
            os.path.getmtime(reference_path)
            if reference_path and os.path.exists(reference_path)
            else 0
        )
        key = "_".join(
            [
                test_path,
                str(test_mtime),
                str(reference_path),
                str(ref_mtime),
                str(window),
                str(score_rate),
                ",".join(sorted(self.metrics)),
            ]
        )
        return md5(key.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return Path(self.cache_dir) / f"{key}.pkl"

    def _load_cache(self, key: str):
        cache_file = self._cache_path(key)
        if cache_file.exists():
            try:
                with cache_file.open("rb") as handle:
                    return pickle.load(handle)
            except Exception:
                print(f"读取缓存 {cache_file} 失败")
        return None

    def _save_cache(self, key: str, data) -> None:
        cache_file = self._cache_path(key)
        try:
            with cache_file.open("wb") as handle:
                pickle.dump(data, handle)
        except Exception:
            print(f"保存缓存 {cache_file} 失败")

    # Evaluation --------------------------------------------------------
    def evaluate_file(
        self,
        test_file: str,
        reference_file: Optional[str] = None,
        window: Optional[float] = None,
        score_rate: int = 16000,
    ):
        cache_key = None
        if self.use_cache:
            cache_key = self._cache_key(test_file, reference_file, window, score_rate)
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        non_reference_metrics = {"DNSMOS", "SRMR"}
        needs_reference = any(m not in non_reference_metrics for m in self.metrics)

        if reference_file is None and needs_reference:
            available = [m for m in self.metrics if m in non_reference_metrics]
            if not available:
                return {"error": "没有可用的指标！请提供参考音频或选择不需要参考的指标。"}
            evaluator = _SpeechScore(available)
            result = evaluator(
                test_path=test_file,
                reference_path=None,
                window=window,
                score_rate=score_rate,
                return_mean=False,
            )
        else:
            result = self.speech_score(
                test_path=test_file,
                reference_path=reference_file,
                window=window,
                score_rate=score_rate,
                return_mean=False,
            )

        if cache_key is not None:
            self._save_cache(cache_key, result)
        return result

    def evaluate_with_mapping(
        self,
        mapping: Dict[str, Dict[str, Optional[str]]],
        window: Optional[float] = None,
        score_rate: int = 16000,
        return_mean: bool = False,
    ) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        for test_file, info in mapping.items():
            metrics = self.evaluate_file(
                test_file,
                info.get("reference"),
                window=window,
                score_rate=score_rate,
            )
            results[Path(test_file).name] = metrics

        if return_mean and results:
            results["Mean_Score"] = self._calculate_mean_scores(results)
        return results

    def evaluate_parallel_with_mapping(
        self,
        mapping: Dict[str, Dict[str, Optional[str]]],
        window: Optional[float] = None,
        score_rate: int = 16000,
        max_workers: Optional[int] = None,
        return_mean: bool = False,
    ) -> Dict[str, Dict]:
        items = list(mapping.items())
        if not items:
            return {}

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        def _task(test_file: str, reference: Optional[str]):
            return Path(test_file).name, self.evaluate_file(
                test_file,
                reference,
                window=window,
                score_rate=score_rate,
            )

        results: Dict[str, Dict] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_task, test_file, info.get("reference")): test_file
                for test_file, info in items
            }
            for future in as_completed(futures):
                test_file = futures[future]
                try:
                    key, metrics = future.result()
                    results[key] = metrics
                except Exception as exc:
                    results[Path(test_file).name] = {"error": str(exc)}

        if return_mean and results:
            results["Mean_Score"] = self._calculate_mean_scores(results)
        return results

    def _calculate_mean_scores(self, results: Dict[str, Dict]) -> Dict[str, float]:
        metric_names = [k for k in results if k != "Mean_Score"]
        if not metric_names:
            return {}

        aggregate: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for name in metric_names:
            result = results[name]
            if not isinstance(result, dict):
                continue
            for metric, value in result.items():
                if isinstance(value, (int, float)):
                    aggregate[metric] = aggregate.get(metric, 0.0) + float(value)
                    counts[metric] = counts.get(metric, 0) + 1

        mean_scores = {
            metric: aggregate[metric] / counts[metric]
            for metric in aggregate
            if counts.get(metric)
        }
        return mean_scores

    def save_results(self, scores: Dict, output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(scores, handle, indent=4, ensure_ascii=False)
        print(f"评估结果已保存到: {output_file}")


def load_asr_calculator(**kwargs):  # pragma: no cover - requires heavy deps
    global _FUNASR_CALCULATOR
    if _FUNASR_CALCULATOR is None:
        from chinese_tts_cer_evaluator import FunASRCERCalculator

        _FUNASR_CALCULATOR = FunASRCERCalculator(**kwargs)
    return _FUNASR_CALCULATOR


def evaluate_dataset(
    evaluator: AudioQualityEvaluator,
    mapping: Dict[str, Dict[str, Optional[str]]],
    window: Optional[float] = None,
    score_rate: int = 16000,
    enable_asr: bool = False,
    asr_kwargs: Optional[Dict] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    return_mean: bool = False,
) -> Dict[str, Dict]:
    """Run quality (and optional ASR) evaluation for the dataset."""

    if parallel:
        quality_scores = evaluator.evaluate_parallel_with_mapping(
            mapping,
            window=window,
            score_rate=score_rate,
            max_workers=max_workers,
            return_mean=False,
        )
    else:
        quality_scores = evaluator.evaluate_with_mapping(
            mapping,
            window=window,
            score_rate=score_rate,
            return_mean=False,
        )

    if enable_asr and mapping:
        asr = load_asr_calculator(**(asr_kwargs or {}))
        for test_file, info in mapping.items():
            transcript = info.get("transcript")
            if not transcript:
                continue
            try:  # pragma: no cover - requires audio deps
                import soundfile as sf

                audio, sample_rate = sf.read(test_file)
                cer_result = asr.compute_cer(
                    audio,
                    sample_rate,
                    transcript,
                    consider_polyphones=True,
                    detailed_analysis=False,
                )
                quality_scores.setdefault(Path(test_file).name, {})["asr"] = cer_result
            except Exception as exc:
                quality_scores.setdefault(Path(test_file).name, {})["asr_error"] = str(exc)

    if return_mean and quality_scores:
        quality_scores["Mean_Score"] = evaluator._calculate_mean_scores(quality_scores)
    return quality_scores


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Audio quality evaluation pipeline")
    parser.add_argument("--test", required=True, help="测试音频文件或目录")
    parser.add_argument("--reference", help="参考音频目录")
    parser.add_argument("--transcripts", help="包含 filepath|speaker|lang|text 的文本文件")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--metrics", nargs="+", help="要评估的指标列表")
    parser.add_argument("--sample-rate", type=int, default=16000, help="采样率")
    parser.add_argument("--window", type=float, help="窗口大小（秒）")
    parser.add_argument("--mean", action="store_true", help="是否返回平均分")
    parser.add_argument("--parallel", action="store_true", help="是否并行计算")
    parser.add_argument("--workers", type=int, help="并行工作进程数")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录")
    parser.add_argument("--fast", action="store_true", help="快速模式，仅计算轻量指标")
    parser.add_argument("--enable-asr", action="store_true", help="是否启用ASR评估")
    parser.add_argument("--asr-device", default="cuda:0", help="FunASR 推理设备")
    parser.add_argument("--asr-model", default="paraformer-zh", help="FunASR模型名称")
    parser.add_argument("--polyphone", default="polyphone.json", help="多音字词典路径")

    args = parser.parse_args()

    metrics = args.metrics
    if args.fast:
        metrics = ["SRMR", "SNR", "SSNR"]

    test_files = collect_audio_files(args.test)
    transcripts = parse_transcript_file(args.transcripts)
    mapping = build_reference_lookup(test_files, args.reference, transcripts)

    evaluator = AudioQualityEvaluator(
        metrics=metrics,
        preload_models=not args.fast,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )

    results = evaluate_dataset(
        evaluator,
        mapping,
        window=args.window,
        score_rate=args.sample_rate,
        enable_asr=args.enable_asr,
        asr_kwargs={
            "model_name": args.asr_model,
            "polyphone_path": args.polyphone,
            "device": args.asr_device,
        },
        parallel=args.parallel,
        max_workers=args.workers,
        return_mean=args.mean,
    )

    evaluator.save_results(results, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
