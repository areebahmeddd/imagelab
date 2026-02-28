import numpy as np
import pytest

from app.operators.filtering.bilateral_filter import BilateralFilter
from app.operators.filtering.box_filter import BoxFilter
from app.operators.filtering.dilation import Dilation
from app.operators.filtering.erosion import Erosion
from app.operators.filtering.morphological import Morphological
from app.operators.filtering.pyramid_down import PyramidDown
from app.operators.filtering.pyramid_up import PyramidUp
from app.operators.filtering.sharpen import Sharpen


class TestBilateralFilter:
    def test_color_shape_preserved(self, color_image):
        assert BilateralFilter({}).compute(color_image).shape == color_image.shape

    def test_custom_params(self, color_image):
        result = BilateralFilter({"filterSize": "9", "sigmaColor": "100", "sigmaSpace": "100"}).compute(color_image)
        assert result.shape == color_image.shape

    def test_dtype_preserved(self, color_image):
        assert BilateralFilter({}).compute(color_image).dtype == color_image.dtype

    def test_rgba_input_converted_to_bgr(self):
        rgba = np.zeros((50, 50, 4), dtype=np.uint8)
        rgba[10:40, 10:40] = [100, 150, 200, 255]
        result = BilateralFilter({}).compute(rgba)
        assert result.shape == (50, 50, 3)


class TestBoxFilter:
    def test_color_shape_preserved(self, color_image):
        assert BoxFilter({}).compute(color_image).shape == color_image.shape

    def test_custom_kernel(self, color_image):
        result = BoxFilter({"width": "10", "height": "10", "depth": "-1"}).compute(color_image)
        assert result.shape == color_image.shape


class TestDilation:
    def test_color_shape_preserved(self, color_image):
        assert Dilation({}).compute(color_image).shape == color_image.shape

    def test_gray_shape_preserved(self, gray_image):
        assert Dilation({}).compute(gray_image).shape == gray_image.shape

    def test_expands_bright_region(self, binary_image):
        result = Dilation({"iteration": "1"}).compute(binary_image)
        assert result.sum() >= binary_image.sum()

    def test_multiple_iterations(self, binary_image):
        result = Dilation({"iteration": "3"}).compute(binary_image)
        assert result.shape == binary_image.shape


class TestErosion:
    def test_color_shape_preserved(self, color_image):
        assert Erosion({}).compute(color_image).shape == color_image.shape

    def test_shrinks_bright_region(self, binary_image):
        result = Erosion({"iteration": "1"}).compute(binary_image)
        assert result.sum() <= binary_image.sum()

    def test_dilation_then_erosion_shape_preserved(self, color_image):
        dilated = Dilation({"iteration": "1"}).compute(color_image)
        eroded = Erosion({"iteration": "1"}).compute(dilated)
        assert eroded.shape == color_image.shape


class TestMorphological:
    @pytest.mark.parametrize("morph_type", ["OPEN", "CLOSE", "GRADIENT", "TOPHAT", "BLACKHAT"])
    def test_all_morph_types_preserve_shape(self, color_image, morph_type):
        result = Morphological({"type": morph_type}).compute(color_image)
        assert result.shape == color_image.shape

    def test_default_morph_type_is_tophat(self, color_image):
        default_result = Morphological({}).compute(color_image)
        tophat_result = Morphological({"type": "TOPHAT"}).compute(color_image)
        np.testing.assert_array_equal(default_result, tophat_result)


class TestPyramidDown:
    def test_halves_spatial_dimensions(self, large_color_image):
        result = PyramidDown({}).compute(large_color_image)
        h, w = large_color_image.shape[:2]
        assert result.shape[0] == h // 2
        assert result.shape[1] == w // 2

    def test_channel_count_preserved(self, large_color_image):
        result = PyramidDown({}).compute(large_color_image)
        assert result.shape[2] == large_color_image.shape[2]


class TestPyramidUp:
    def test_doubles_spatial_dimensions(self, color_image):
        result = PyramidUp({}).compute(color_image)
        h, w = color_image.shape[:2]
        assert result.shape[0] == h * 2
        assert result.shape[1] == w * 2

    def test_channel_count_preserved(self, color_image):
        result = PyramidUp({}).compute(color_image)
        assert result.shape[2] == color_image.shape[2]


class TestSharpen:
    def test_color_shape_preserved(self, color_image):
        assert Sharpen({}).compute(color_image).shape == color_image.shape

    def test_dtype_is_uint8(self, color_image):
        assert Sharpen({}).compute(color_image).dtype == np.uint8

    def test_zero_strength_equals_original(self, color_image):
        np.testing.assert_array_equal(Sharpen({"strength": "0"}).compute(color_image), color_image)

    def test_strength_is_clamped_above(self, color_image):
        # strength > 2.0 should be clamped; result should not raise
        result = Sharpen({"strength": "99"}).compute(color_image)
        assert result.shape == color_image.shape

    def test_strength_is_clamped_below(self, color_image):
        result = Sharpen({"strength": "-5"}).compute(color_image)
        np.testing.assert_array_equal(result, color_image)
