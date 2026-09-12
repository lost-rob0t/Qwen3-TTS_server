import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = (
    "Dockerfile",
    "Dockerfile.cpu",
    "Dockerfile.cuda",
    "Dockerfile.rocm",
)


class ContainerAssetsTest(unittest.TestCase):
    def test_every_server_image_copies_audio_utils(self):
        for name in DOCKERFILES:
            with self.subTest(dockerfile=name):
                content = (ROOT / name).read_text(encoding="utf-8")
                self.assertRegex(content, r"COPY .*audio_utils\.py")


if __name__ == "__main__":
    unittest.main()
