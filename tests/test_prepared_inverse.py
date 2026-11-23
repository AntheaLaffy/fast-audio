import unittest

import torch
from tools.gtcrn_torch_worker import PreparedInverse


class PreparedInverseTests(unittest.TestCase):
    def test_full_roundtrip_and_modified_spectra_match_original_inverse(self):
        generator = torch.Generator().manual_seed(731)
        window = torch.hann_window(512).sqrt()
        with torch.inference_mode():
            for batch, length in [(1, 4096), (1, 16000), (1, 64000), (2, 16000)]:
                x = torch.randn(batch, length, generator=generator) * 0.1
                spectrum = torch.stft(x, 512, 256, 512, window, return_complex=True)
                inverse = PreparedInverse(window, spectrum.shape[-1], length)
                for value in [spectrum, spectrum * (0.73 + 0.19j)]:
                    actual = inverse(value)
                    expected = torch.istft(value, 512, 256, 512, window, length=length)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(inverse(spectrum), x, atol=1e-6, rtol=1e-5)

    def test_nola_geometry_and_spectrum_shape_are_checked(self):
        with self.assertRaisesRegex(ValueError, "NOLA"):
            PreparedInverse(torch.zeros(512), 17, 4096)
        with self.assertRaisesRegex(ValueError, "tail"):
            PreparedInverse(torch.ones(512), 2, 4096)
        inverse = PreparedInverse(torch.ones(512), 17, 4096)
        with self.assertRaisesRegex(ValueError, "shape"):
            inverse(torch.zeros(1, 17, 257, dtype=torch.complex64))


if __name__ == "__main__":
    unittest.main()
