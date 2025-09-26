from pathlib import Path

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_quality_pipeline import (
    build_reference_lookup,
    collect_audio_files,
    parse_transcript_file,
    strip_variant_suffix,
)


def test_strip_variant_suffix():
    assert strip_variant_suffix("aa_0001") == "aa"
    assert strip_variant_suffix("bb_12") == "bb"
    assert strip_variant_suffix("cc") == "cc"
    assert strip_variant_suffix("prefix_suffix_0003") == "prefix_suffix"


def test_parse_transcript_file(tmp_path: Path):
    transcript = tmp_path / "transcripts.txt"
    transcript.write_text(
        "reference_audio/aa.wav|spk1|zh|你好世界\n"
        "reference_audio/bb.wav|spk2|zh|欢迎使用\n",
        encoding="utf-8",
    )

    mapping = parse_transcript_file(str(transcript))
    assert mapping["aa"]["text"] == "你好世界"
    assert mapping["aa"]["filepath"] == "reference_audio/aa.wav"
    assert mapping["bb"]["speaker"] == "spk2"


def test_build_reference_lookup(tmp_path: Path):
    test_dir = tmp_path / "test_audio"
    ref_dir = tmp_path / "reference_audio"
    test_dir.mkdir()
    ref_dir.mkdir()

    (test_dir / "aa_0001.wav").write_bytes(b"")
    (test_dir / "aa_0002.wav").write_bytes(b"")
    (test_dir / "bb_0001.wav").write_bytes(b"")
    (ref_dir / "aa.wav").write_bytes(b"")
    (ref_dir / "bb.wav").write_bytes(b"")

    transcript = tmp_path / "transcripts.txt"
    transcript.write_text(
        "reference_audio/aa.wav|spk1|zh|你好世界\n"
        "reference_audio/bb.wav|spk2|zh|欢迎使用\n",
        encoding="utf-8",
    )

    transcripts = parse_transcript_file(str(transcript))
    test_files = collect_audio_files(str(test_dir))
    lookup = build_reference_lookup(test_files, str(ref_dir), transcripts)

    key = str(test_dir / "aa_0001.wav")
    assert lookup[key]["reference"] == str(ref_dir / "aa.wav")
    assert lookup[key]["transcript"] == "你好世界"

    key2 = str(test_dir / "bb_0001.wav")
    assert lookup[key2]["reference"] == str(ref_dir / "bb.wav")
    assert lookup[key2]["transcript"] == "欢迎使用"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("aa_0001.wav", "aa"),
        ("bb.wav", "bb"),
        ("complex_name_12345.flac", "complex_name"),
    ],
)
def test_lookup_handles_variants(filename, expected, tmp_path: Path):
    test_dir = tmp_path / "test_audio"
    ref_dir = tmp_path / "reference_audio"
    test_dir.mkdir()
    ref_dir.mkdir()

    test_path = test_dir / filename
    ref_path = ref_dir / f"{expected}.wav"
    test_path.write_bytes(b"")
    ref_path.write_bytes(b"")

    transcript = tmp_path / "transcripts.txt"
    transcript.write_text(
        f"reference_audio/{expected}.wav|spk|zh|text\n",
        encoding="utf-8",
    )

    transcripts = parse_transcript_file(str(transcript))
    lookup = build_reference_lookup([str(test_path)], str(ref_dir), transcripts)
    assert lookup[str(test_path)]["reference"] == str(ref_path)
    assert lookup[str(test_path)]["transcript"] == "text"
