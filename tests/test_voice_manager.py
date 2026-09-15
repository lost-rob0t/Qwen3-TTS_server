import json
import math
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "voice_manager.py"


class VoiceManagerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.voices = self.root / "voices"
        self.sample = self.root / "sample.wav"
        self.write_sample(self.sample)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_sample(path: Path):
        sample_rate = 24000
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            frames = bytearray()
            for index in range(sample_rate):
                value = int(5000 * math.sin(2 * math.pi * 220 * index / sample_rate))
                frames.extend(value.to_bytes(2, byteorder="little", signed=True))
            output.writeframes(frames)

    def run_cli(self, *arguments, check=True):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--directory", str(self.voices), *arguments],
            check=check,
            capture_output=True,
            text=True,
        )

    def test_add_list_info_rename_default_and_remove(self):
        self.run_cli("add", "sample_voice", str(self.sample), "--text", "Hello from the test.")
        self.assertTrue((self.voices / "sample_voice.wav").is_file())
        metadata = json.loads((self.voices / "sample_voice.json").read_text())
        self.assertEqual(metadata["reference_text"], "Hello from the test.")

        listing = json.loads(self.run_cli("list", "--json").stdout)
        self.assertEqual(listing[0]["name"], "sample_voice")

        self.run_cli("rename", "sample_voice", "renamed_voice")
        self.assertFalse((self.voices / "sample_voice.wav").exists())
        self.assertTrue((self.voices / "renamed_voice.wav").exists())

        self.run_cli("set-default", "renamed_voice")
        default = json.loads((self.voices / "default_en.json").read_text())
        self.assertEqual(default["source_voice"], "renamed_voice")

        self.run_cli("remove", "renamed_voice")
        self.assertFalse((self.voices / "renamed_voice.wav").exists())

    def test_extracts_audio_from_video(self):
        video = self.root / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                "-i", str(self.sample), "-shortest", "-c:v", "mpeg4", str(video),
            ],
            check=True,
        )
        self.run_cli("create", "video_voice", str(video), "--duration", "0.5")
        self.assertGreater((self.voices / "video_voice.wav").stat().st_size, 44)

    def test_rejects_unsafe_name(self):
        result = self.run_cli("add", "../outside", str(self.sample), check=False)
        self.assertNotEqual(result.returncode, 0)


class VoiceTranscriptContractTest(unittest.TestCase):
    """The GGML server entrypoint pairs each voices/<name>.wav with an
    optional same-stem <name>.txt whose contents become the ref_text ICL
    clone payload. Without a transcript the server can only fall back to
    the lower fidelity x_vector_only path, so the manager must keep the
    .txt artifact in sync with every lifecycle operation."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.voices = self.root / "voices"
        self.sample = self.root / "sample.wav"
        VoiceManagerTest.write_sample(self.sample)

    def tearDown(self):
        self.temporary.cleanup()

    def run_cli(self, *arguments, check=True):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--directory", str(self.voices), *arguments],
            check=check,
            capture_output=True,
            text=True,
        )

    def test_add_with_text_writes_transcript_file(self):
        self.run_cli("add", "voiced", str(self.sample), "--text", "Exact words spoken.")
        transcript = self.voices / "voiced.txt"
        self.assertTrue(transcript.is_file())
        self.assertEqual(transcript.read_text(encoding="utf-8").strip(), "Exact words spoken.")

    def test_add_without_text_writes_no_transcript_file(self):
        self.run_cli("add", "unvoiced", str(self.sample))
        self.assertFalse((self.voices / "unvoiced.txt").exists())

    def test_set_text_creates_and_updates_transcript(self):
        self.run_cli("add", "voiced", str(self.sample))
        self.run_cli("set-text", "voiced", "First transcript.")
        self.assertEqual((self.voices / "voiced.txt").read_text(encoding="utf-8").strip(), "First transcript.")
        self.run_cli("set-text", "voiced", "Updated transcript.")
        self.assertEqual((self.voices / "voiced.txt").read_text(encoding="utf-8").strip(), "Updated transcript.")

    def test_rename_moves_transcript(self):
        self.run_cli("add", "voiced", str(self.sample), "--text", "Traveling transcript.")
        self.run_cli("rename", "voiced", "moved")
        self.assertFalse((self.voices / "voiced.txt").exists())
        self.assertEqual((self.voices / "moved.txt").read_text(encoding="utf-8").strip(), "Traveling transcript.")

    def test_remove_deletes_transcript(self):
        self.run_cli("add", "voiced", str(self.sample), "--text", "Doomed transcript.")
        self.run_cli("remove", "voiced", "--force")
        self.assertFalse((self.voices / "voiced.txt").exists())

    def test_set_default_copies_transcript(self):
        self.run_cli("add", "voiced", str(self.sample), "--text", "Default transcript.")
        self.run_cli("set-default", "voiced")
        self.assertEqual((self.voices / "default_en.txt").read_text(encoding="utf-8").strip(), "Default transcript.")

    def test_transcript_with_special_characters_survives_roundtrip(self):
        text = 'He said "quotes", back\\slash, and 100% newlines\nare fine.'
        self.run_cli("add", "quoted", str(self.sample), "--text", text)
        self.assertEqual((self.voices / "quoted.txt").read_text(encoding="utf-8").strip(), text.strip())


if __name__ == "__main__":
    unittest.main()
