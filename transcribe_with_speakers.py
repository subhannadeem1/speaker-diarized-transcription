#!/usr/bin/env python3
"""
Transcribe audio and add speaker labels using WhisperX + pyannote diarization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def format_ts(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def merge_segments(segments: list[dict[str, Any]], max_gap: float = 0.8) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        speaker = seg.get("speaker") or "SPEAKER_UNKNOWN"
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        if merged:
            prev = merged[-1]
            if prev["speaker"] == speaker and start - prev["end"] <= max_gap:
                prev["end"] = end
                prev["text"] = f"{prev['text']} {text}".strip()
                continue

        merged.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return merged


def write_outputs(
    base_output_path: Path,
    merged_lines: list[dict[str, Any]],
    full_result: dict[str, Any],
) -> None:
    txt_path = base_output_path.with_suffix(".txt")
    json_path = base_output_path.with_suffix(".json")

    with txt_path.open("w", encoding="utf-8") as f:
        for line in merged_lines:
            f.write(
                f"[{format_ts(line['start'])} - {format_ts(line['end'])}] "
                f"{line['speaker']}: {line['text']}\n"
            )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)

    print(f"Wrote transcript: {txt_path}")
    print(f"Wrote raw result: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio and assign speaker labels."
    )
    parser.add_argument("audio", help="Path to input audio/video file")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. en, hi, fr). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type for WhisperX (e.g. int8, float16). Default: int8",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Transcription batch size (default: 8)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face token for diarization. "
            "If omitted, reads HF_TOKEN env variable."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output base path (without extension). Default: input filename + _speakers",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional minimum number of speakers for diarization",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional maximum number of speakers for diarization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Input file not found: {audio_path}")

    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN or pass --hf-token."
        )

    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "whisperx is not installed. Run: pip install -r requirements.txt"
        ) from exc

    print("Loading audio...")
    audio = whisperx.load_audio(str(audio_path))

    print(f"Loading Whisper model: {args.model}")
    model = whisperx.load_model(
        args.model,
        args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    print("Running transcription...")
    result = model.transcribe(audio, batch_size=args.batch_size, language=args.language)

    align_language = args.language or result.get("language")
    if align_language:
        try:
            print(f"Running alignment for language: {align_language}")
            align_model, metadata = whisperx.load_align_model(
                language_code=align_language,
                device=args.device,
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                args.device,
                return_char_alignments=False,
            )
        except Exception as exc:
            print(f"Alignment failed, continuing without alignment: {exc}")

    print("Running speaker diarization...")
    diarize_pipeline = whisperx.DiarizationPipeline(
        use_auth_token=token, device=args.device
    )
    diarize_segments = diarize_pipeline(
        audio,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print("Assigning speaker labels to transcript segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    merged_lines = merge_segments(result.get("segments", []))
    result["speaker_lines"] = merged_lines

    if args.output:
        base_output_path = Path(args.output).expanduser().resolve()
    else:
        base_output_path = audio_path.with_name(f"{audio_path.stem}_speakers")

    write_outputs(base_output_path, merged_lines, result)
    print("Done.")


if __name__ == "__main__":
    main()
