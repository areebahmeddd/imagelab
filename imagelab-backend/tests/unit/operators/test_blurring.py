import numpy as np
import pytest

from app.operators.blurring.blur import Blur
from app.operators.blurring.gaussian_blur import GaussianBlur
from app.operators.blurring.median_blur import MedianBlur


class TestBlur:
    def test_color_shape_preserved(self, color_image):
        assert Blur({}).compute(color_image).shape == color_image.shape

    def test_gray_shape_preserved(self, gray_image):
        assert Blur({}).compute(gray_image).shape == gray_image.shape

    def test_dtype_preserved(self, color_image):
        assert Blur({}).compute(color_image).dtype == color_image.dtype

    def test_custom_kernel(self, color_image):
        result = Blur({"widthSize": "7", "heightSize": "7"}).compute(color_image)
        assert result.shape == color_image.shape

    def test_asymmetric_kernel(self, color_image):
        result = Blur({"widthSize": "3", "heightSize": "9"}).compute(color_image)
        assert result.shape == color_image.shape

    def test_custom_anchor(self, color_image):
        result = Blur({"widthSize": "5", "heightSize": "5", "pointX": "0", "pointY": "0"}).compute(color_image)
        assert result.shape == color_image.shape

    def test_large_kernel_reduces_variance(self):
        rng = np.random.default_rng(42)
        noisy = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        assert float(Blur({"widthSize": "21", "heightSize": "21"}).compute(noisy).var()) < float(noisy.var())

    def test_uniform_image_unchanged(self):
        uniform = np.full((50, 50, 3), 128, dtype=np.uint8)
        np.testing.assert_array_equal(Blur({"widthSize": "5", "heightSize": "5"}).compute(uniform), uniform)


class TestGaussianBlur:
    def test_color_shape_preserved(self, color_image):
        assert GaussianBlur({}).compute(color_image).shape == color_image.shape

    def test_gray_shape_preserved(self, gray_image):
        assert GaussianBlur({}).compute(gray_image).shape == gray_image.shape

    @pytest.mark.parametrize("w,h", [("4", "3"), ("3", "4"), ("2", "2"), ("6", "6")])
    def test_even_kernel_corrected(self, color_image, w, h):
        result = GaussianBlur({"widthSize": w, "heightSize": h}).compute(color_image)
        assert result.shape == color_image.shape

    def test_odd_kernel_passthrough(self, color_image):
        result = GaussianBlur({"widthSize": "5", "heightSize": "5"}).compute(color_image)
        assert result.shape == color_image.shape

    def test_large_kernel_reduces_variance(self):
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        assert float(GaussianBlur({"widthSize": "21", "heightSize": "21"}).compute(noisy).var()) < float(noisy.var())

    def test_uniform_image_unchanged(self):
        uniform = np.full((60, 60, 3), 200, dtype=np.uint8)
        np.testing.assert_array_equal(GaussianBlur({"widthSize": "5", "heightSize": "5"}).compute(uniform), uniform)


class TestMedianBlur:
    def test_color_shape_preserved(self, color_image):
        assert MedianBlur({}).compute(color_image).shape == color_image.shape

    def test_gray_shape_preserved(self, gray_image):
        assert MedianBlur({}).compute(gray_image).shape == gray_image.shape

    @pytest.mark.parametrize("k", ["2", "4", "6"])
    def test_even_kernel_corrected(self, color_image, k):
        result = MedianBlur({"kernelSize": k}).compute(color_image)
        assert result.shape == color_image.shape

    @pytest.mark.parametrize("k", ["3", "7", "11"])
    def test_odd_kernels(self, color_image, k):
        result = MedianBlur({"kernelSize": k}).compute(color_image)
        assert result.shape == color_image.shape

    def test_removes_salt_and_pepper(self):
        base = 128
        img = np.full((50, 50), base, dtype=np.uint8)
        rng = np.random.default_rng(7)
        coords = rng.integers(0, 50, (20, 2))
        for r, c in coords:
            img[r, c] = 255 if (r + c) % 2 == 0 else 0
        result = MedianBlur({"kernelSize": "3"}).compute(img)
        assert np.sum(np.abs(result.astype(int) - base) < 20) > 50 * 50 * 0.9

    def test_uniform_image_unchanged(self):
        uniform = np.full((40, 40, 3), 100, dtype=np.uint8)
        np.testing.assert_array_equal(MedianBlur({"kernelSize": "3"}).compute(uniform), uniform)
