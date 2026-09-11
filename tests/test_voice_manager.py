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


if __name__ == "__main__":
    unittest.main()
