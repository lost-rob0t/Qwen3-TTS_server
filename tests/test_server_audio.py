import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_utils import normalize_reference_audio


class ServerAudioTest(unittest.TestCase):
    def test_stereo_reference_is_mixed_to_mono_float32(self):
        audio = np.array(
            [
                [0.25, 0.75],
                [-0.5, 0.5],
            ],
            dtype=np.float64,
        )

        normalized = normalize_reference_audio(audio)

        np.testing.assert_array_equal(
            normalized,
            np.array([0.5, 0.0], dtype=np.float32),
        )
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(normalized.ndim, 1)

    def test_mono_reference_is_preserved_as_float32(self):
        audio = np.array([0.25, -0.5], dtype=np.float64)

        normalized = normalize_reference_audio(audio)

        np.testing.assert_array_equal(
            normalized,
            np.array([0.25, -0.5], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
