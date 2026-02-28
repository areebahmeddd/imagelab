import numpy as np
import pytest

from app.operators.transformation.distance_transform import DistanceTransform
from app.operators.transformation.laplacian import Laplacian


class TestDistanceTransform:
    def test_color_output_shape_matches_hw(self, color_image):
        result = DistanceTransform({}).compute(color_image)
        assert result.shape == color_image.shape[:2]

    def test_output_dtype_is_uint8(self, color_image):
        assert DistanceTransform({}).compute(color_image).dtype == np.uint8

    def test_output_values_in_range(self, color_image):
        result = DistanceTransform({}).compute(color_image)
        assert result.min() >= 0
        assert result.max() <= 255

    @pytest.mark.parametrize("dist_type", ["DIST_L1", "DIST_L2", "DIST_C"])
    def test_all_distance_types(self, color_image, dist_type):
        result = DistanceTransform({"type": dist_type}).compute(color_image)
        assert result.shape == color_image.shape[:2]

    def test_binary_image_with_whites_produces_nonzero(self, binary_image):
        result = DistanceTransform({}).compute(binary_image)
        assert result.max() > 0


class TestLaplacian:
    def test_color_output_shape_preserved(self, color_image):
        assert Laplacian({}).compute(color_image).shape == color_image.shape

    def test_gray_output_shape_preserved(self, gray_image):
        assert Laplacian({}).compute(gray_image).shape == gray_image.shape

    def test_output_dtype_is_uint8(self, color_image):
        assert Laplacian({}).compute(color_image).dtype == np.uint8

    def test_uniform_image_produces_zero(self):
        uniform = np.full((60, 60, 3), 100, dtype=np.uint8)
        result = Laplacian({}).compute(uniform)
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_edge_in_binary_image_detected(self, binary_image):
        result = Laplacian({}).compute(binary_image)
        assert result.max() > 0
