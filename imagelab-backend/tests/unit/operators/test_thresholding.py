import numpy as np

from app.operators.thresholding.adaptive_threshold import AdaptiveThreshold
from app.operators.thresholding.apply_borders import ApplyBorders
from app.operators.thresholding.apply_threshold import ApplyThreshold
from app.operators.thresholding.otsu_threshold import OtsuThreshold


class TestApplyThreshold:
    def test_gray_output_shape_preserved(self, gray_image):
        result = ApplyThreshold({"thresholdValue": "100", "maxValue": "255"}).compute(gray_image)
        assert result.shape == gray_image.shape

    def test_output_values_are_binary(self, gray_image):
        result = ApplyThreshold({"thresholdValue": "100", "maxValue": "255"}).compute(gray_image)
        assert set(np.unique(result)).issubset({0, 255})

    def test_zero_threshold_keeps_all(self, gray_image):
        result = ApplyThreshold({"thresholdValue": "0", "maxValue": "255"}).compute(gray_image)
        assert result.max() == 255


class TestAdaptiveThreshold:
    def test_color_input_converted_to_gray(self, color_image):
        result = AdaptiveThreshold({}).compute(color_image)
        assert result.ndim == 2

    def test_gray_input_shape_preserved(self, gray_image):
        result = AdaptiveThreshold({}).compute(gray_image)
        assert result.shape == gray_image.shape

    def test_output_values_are_binary(self, gray_image):
        result = AdaptiveThreshold({}).compute(gray_image)
        assert set(np.unique(result)).issubset({0, 255})

    def test_even_block_size_corrected(self, gray_image):
        result = AdaptiveThreshold({"blockSize": "4"}).compute(gray_image)
        assert result.ndim == 2

    def test_small_block_size_clamped_to_3(self, gray_image):
        result = AdaptiveThreshold({"blockSize": "1"}).compute(gray_image)
        assert result.ndim == 2

    def test_mean_method(self, gray_image):
        result = AdaptiveThreshold({"adaptiveMethod": "MEAN"}).compute(gray_image)
        assert result.ndim == 2

    def test_gaussian_method(self, gray_image):
        result = AdaptiveThreshold({"adaptiveMethod": "GAUSSIAN"}).compute(gray_image)
        assert result.ndim == 2

    def test_custom_cvalue_accepted(self, gray_image):
        result = AdaptiveThreshold({"cValue": "5"}).compute(gray_image)
        assert result.ndim == 2
        assert set(np.unique(result)).issubset({0, 255})


class TestOtsuThreshold:
    def test_color_input_produces_binary(self, color_image):
        result = OtsuThreshold({}).compute(color_image)
        assert result.ndim == 2
        assert set(np.unique(result)).issubset({0, 255})

    def test_gray_input_shape_preserved(self, gray_image):
        result = OtsuThreshold({}).compute(gray_image)
        assert result.shape == gray_image.shape

    def test_custom_max_value(self, gray_image):
        result = OtsuThreshold({"maxValue": "128"}).compute(gray_image)
        assert result.max() <= 128


class TestApplyBorders:
    def test_uniform_border_increases_dimensions(self, color_image):
        h, w = color_image.shape[:2]
        result = ApplyBorders({"border_all_sides": "10"}).compute(color_image)
        assert result.shape[0] == h + 20
        assert result.shape[1] == w + 20

    def test_individual_borders(self, color_image):
        h, w = color_image.shape[:2]
        result = ApplyBorders({"borderTop": "5", "borderBottom": "10", "borderLeft": "3", "borderRight": "7"}).compute(
            color_image
        )
        assert result.shape[0] == h + 15
        assert result.shape[1] == w + 10

    def test_zero_border_unchanged(self, color_image):
        result = ApplyBorders({"borderTop": "0", "borderBottom": "0", "borderLeft": "0", "borderRight": "0"}).compute(
            color_image
        )
        np.testing.assert_array_equal(result, color_image)

    def test_border_is_black(self, color_image):
        result = ApplyBorders({"border_all_sides": "5"}).compute(color_image)
        assert result[0, 0, 0] == 0  # top-left corner pixel is black

    def test_border_all_takes_priority(self, color_image):
        h, w = color_image.shape[:2]
        result = ApplyBorders({"border_all_sides": "10", "borderTop": "99"}).compute(color_image)
        assert result.shape[0] == h + 20
