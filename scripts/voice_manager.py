#!/usr/bin/env python3
"""Manage persistent Qwen3-TTS reference voices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


VOICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")


def default_voice_dir() -> Path:
    data_home = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))
    data_dir = Path(os.getenv("QWEN3_TTS_DATA_DIR", data_home / "qwen3-tts"))
    return Path(os.getenv("QWEN_RESOURCES_DIR", data_dir / "voices")).expanduser()


def valid_name(value: str) -> str:
    if not VOICE_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "must be 1-64 letters, numbers, underscores, or hyphens"
        )
    return value


def voice_file(directory: Path, name: str) -> Path | None:
    for extension in AUDIO_EXTENSIONS:
        candidate = directory / f"{name}{extension}"
        if candidate.is_file():
            return candidate
    return None


def metadata_file(directory: Path, name: str) -> Path:
    return directory / f"{name}.json"


def transcript_file(directory: Path, name: str) -> Path:
    return directory / f"{name}.txt"


def write_transcript(directory: Path, name: str, text: str | None) -> None:
    destination = transcript_file(directory, name)
    if not text:
        destination.unlink(missing_ok=True)
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=f".{name}-", suffix=".txt", delete=False
    ) as temporary:
        temporary.write(text.rstrip("\n"))
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)


def read_metadata(directory: Path, name: str) -> dict:
    path = metadata_file(directory, name)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid metadata for {name}: {error}") from error


def write_metadata(directory: Path, name: str, metadata: dict) -> None:
    destination = metadata_file(directory, name)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=f".{name}-", delete=False
    ) as temporary:
        json.dump(metadata, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_voice(directory: Path, name: str) -> Path:
    result = voice_file(directory, name)
    if result is None:
        raise SystemExit(f"voice not found: {name}")
    return result


def command_add(args: argparse.Namespace) -> None:
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"source file not found: {source}")
    if voice_file(args.directory, args.name) and not args.replace:
        raise SystemExit(f"voice already exists: {args.name} (use --replace)")

    args.directory.mkdir(parents=True, exist_ok=True)
    destination = args.directory / f"{args.name}.wav"
    with tempfile.NamedTemporaryFile(
        dir=args.directory, prefix=f".{args.name}-", suffix=".wav", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if args.start:
        command.extend(["-ss", args.start])
    command.extend(["-i", str(source), "-vn"])
    if args.duration:
        command.extend(["-t", str(args.duration)])
    command.extend(["-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(temporary_path)])

    try:
        subprocess.run(command, check=True)
        if temporary_path.stat().st_size <= 44:
            raise SystemExit("ffmpeg produced an empty voice clip")
        for extension in AUDIO_EXTENSIONS:
            old_file = args.directory / f"{args.name}{extension}"
            if old_file != destination:
                old_file.unlink(missing_ok=True)
        temporary_path.replace(destination)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"ffmpeg could not extract audio from {source}") from error
    finally:
        temporary_path.unlink(missing_ok=True)

    metadata = {
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": source.name,
        "source_sha256": sha256(source),
        "audio_sha256": sha256(destination),
        "start": args.start,
        "duration": args.duration,
        "reference_text": args.text,
    }
    write_metadata(args.directory, args.name, metadata)
    write_transcript(args.directory, args.name, args.text)
    print(f"added {args.name}: {destination}")


def command_list(args: argparse.Namespace) -> None:
    found = []
    if args.directory.is_dir():
        for path in sorted(args.directory.iterdir()):
            if path.suffix.lower() not in AUDIO_EXTENSIONS or not VOICE_NAME.fullmatch(path.stem):
                continue
            if voice_file(args.directory, path.stem) != path:
                continue
            metadata = read_metadata(args.directory, path.stem)
            found.append((path.stem, metadata.get("reference_text") or "auto-transcribe"))
    if args.json:
        print(json.dumps([{"name": name, "reference_text": text} for name, text in found]))
        return
    if not found:
        print("no voices")
        return
    for name, text in found:
        print(f"{name}\t{text}")


def command_info(args: argparse.Namespace) -> None:
    audio = require_voice(args.directory, args.name)
    result = read_metadata(args.directory, args.name)
    result.update({"name": args.name, "audio_file": str(audio), "bytes": audio.stat().st_size})
    print(json.dumps(result, indent=2, sort_keys=True))


def command_remove(args: argparse.Namespace) -> None:
    require_voice(args.directory, args.name)
    if args.name == "default_en" and not args.force:
        raise SystemExit("refusing to remove default_en without --force")
    for extension in (*AUDIO_EXTENSIONS, ".json", ".txt"):
        (args.directory / f"{args.name}{extension}").unlink(missing_ok=True)
    print(f"removed {args.name}")


def command_rename(args: argparse.Namespace) -> None:
    source = require_voice(args.directory, args.name)
    if voice_file(args.directory, args.new_name):
        raise SystemExit(f"voice already exists: {args.new_name}")
    destination = args.directory / f"{args.new_name}{source.suffix.lower()}"
    source.replace(destination)
    metadata = read_metadata(args.directory, args.name)
    metadata_file(args.directory, args.name).unlink(missing_ok=True)
    transcript = transcript_file(args.directory, args.name)
    if transcript.is_file():
        transcript.replace(transcript_file(args.directory, args.new_name))
    metadata["name"] = args.new_name
    metadata["renamed_at"] = datetime.now(timezone.utc).isoformat()
    write_metadata(args.directory, args.new_name, metadata)
    print(f"renamed {args.name} to {args.new_name}")


def command_set_text(args: argparse.Namespace) -> None:
    require_voice(args.directory, args.name)
    metadata = read_metadata(args.directory, args.name)
    metadata["name"] = args.name
    metadata["reference_text"] = args.text
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_metadata(args.directory, args.name, metadata)
    write_transcript(args.directory, args.name, args.text)
    print(f"updated transcript for {args.name}")


def command_default(args: argparse.Namespace) -> None:
    source = require_voice(args.directory, args.name)
    destination = args.directory / "default_en.wav"
    with tempfile.NamedTemporaryFile(
        dir=args.directory, prefix=".default_en-", suffix=".wav", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    metadata = read_metadata(args.directory, args.name)
    metadata.update({"name": "default_en", "source_voice": args.name})
    write_metadata(args.directory, "default_en", metadata)
    source_transcript = transcript_file(args.directory, args.name)
    default_transcript = transcript_file(args.directory, "default_en")
    if source_transcript.is_file():
        shutil.copyfile(source_transcript, default_transcript)
    else:
        default_transcript.unlink(missing_ok=True)
    print(f"default voice is now {args.name}")


def command_export(args: argparse.Namespace) -> None:
    source = require_voice(args.directory, args.name)
    destination = args.destination.expanduser().resolve()
    if destination.is_dir():
        destination = destination / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=default_voice_dir())
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", aliases=["create"], help="create a voice from audio or video")
    add.add_argument("name", type=valid_name)
    add.add_argument("source", type=Path)
    add.add_argument("--start", help="clip start time, e.g. 12.5 or 00:01:12")
    add.add_argument("--duration", type=float, default=15.0)
    add.add_argument("--text", help="exact words spoken; skips Whisper when supplied")
    add.add_argument("--replace", action="store_true")
    add.set_defaults(handler=command_add)

    listing = commands.add_parser("list", help="list installed voices")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=command_list)

    info = commands.add_parser("info", help="show voice metadata")
    info.add_argument("name", type=valid_name)
    info.set_defaults(handler=command_info)

    remove = commands.add_parser("remove", aliases=["delete"], help="remove a voice")
    remove.add_argument("name", type=valid_name)
    remove.add_argument("--force", action="store_true")
    remove.set_defaults(handler=command_remove)

    rename = commands.add_parser("rename", help="rename a voice")
    rename.add_argument("name", type=valid_name)
    rename.add_argument("new_name", type=valid_name)
    rename.set_defaults(handler=command_rename)

    text = commands.add_parser("set-text", help="set the exact reference transcript")
    text.add_argument("name", type=valid_name)
    text.add_argument("text")
    text.set_defaults(handler=command_set_text)

    default = commands.add_parser("set-default", help="use a voice for /base_tts/")
    default.add_argument("name", type=valid_name)
    default.set_defaults(handler=command_default)

    export = commands.add_parser("export", help="copy a normalized voice WAV elsewhere")
    export.add_argument("name", type=valid_name)
    export.add_argument("destination", type=Path)
    export.set_defaults(handler=command_export)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.directory = args.directory.expanduser().resolve()
    args.handler(args)


if __name__ == "__main__":
    main()
