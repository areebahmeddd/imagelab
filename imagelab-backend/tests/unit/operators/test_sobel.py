import numpy as np
import pytest

from app.operators.sobel_derivatives.scharr_derivative import ScharrDerivative
from app.operators.sobel_derivatives.sobel_derivative import SobelDerivative


class TestSobelDerivative:
    @pytest.mark.parametrize("direction", ["HORIZONTAL", "VERTICAL", "COMBINED"])
    def test_all_directions_preserve_shape(self, color_image, direction):
        result = SobelDerivative({"type": direction}).compute(color_image)
        assert result.shape == color_image.shape

    def test_default_direction_is_horizontal(self, color_image):
        default_result = SobelDerivative({}).compute(color_image)
        explicit_result = SobelDerivative({"type": "HORIZONTAL"}).compute(color_image)
        np.testing.assert_array_equal(default_result, explicit_result)

    def test_edge_detection_on_binary_image(self, binary_image):
        result = SobelDerivative({"type": "HORIZONTAL"}).compute(binary_image)
        assert result.shape == binary_image.shape

    def test_grayscale_input(self, gray_image):
        result = SobelDerivative({"type": "VERTICAL"}).compute(gray_image)
        assert result.shape == gray_image.shape

    def test_output_dtype(self, color_image):
        result = SobelDerivative({}).compute(color_image)
        assert result.dtype in (np.float32, np.float64)


class TestScharrDerivative:
    @pytest.mark.parametrize("direction", ["HORIZONTAL", "VERTICAL"])
    def test_all_directions_preserve_shape(self, color_image, direction):
        result = ScharrDerivative({"type": direction}).compute(color_image)
        assert result.shape == color_image.shape

    def test_default_is_horizontal(self, color_image):
        default = ScharrDerivative({}).compute(color_image)
        explicit = ScharrDerivative({"type": "HORIZONTAL"}).compute(color_image)
        np.testing.assert_array_equal(default, explicit)

    def test_grayscale_input(self, gray_image):
        result = ScharrDerivative({"type": "VERTICAL"}).compute(gray_image)
        assert result.shape == gray_image.shape

    def test_detects_edges_on_binary_image(self, binary_image):
        result = ScharrDerivative({"type": "HORIZONTAL"}).compute(binary_image)
        assert result.max() > 0

    def test_output_dtype(self, color_image):
        result = ScharrDerivative({}).compute(color_image)
        assert result.dtype in (np.float32, np.float64)
