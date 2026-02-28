import numpy as np
import pytest

from app.operators.geometric.affine_image import AffineImage
from app.operators.geometric.crop_image import CropImage
from app.operators.geometric.reflect_image import ReflectImage
from app.operators.geometric.rotate_image import RotateImage
from app.operators.geometric.scale_image import ScaleImage


class TestScaleImage:
    def test_identity_scale_preserves_shape(self, color_image):
        assert ScaleImage({}).compute(color_image).shape == color_image.shape

    def test_scale_up_doubles_dimensions(self, color_image):
        h, w = color_image.shape[:2]
        result = ScaleImage({"fx": "2", "fy": "2"}).compute(color_image)
        assert result.shape[0] == h * 2
        assert result.shape[1] == w * 2

    def test_scale_down_halves_dimensions(self, color_image):
        h, w = color_image.shape[:2]
        result = ScaleImage({"fx": "0.5", "fy": "0.5"}).compute(color_image)
        assert result.shape[0] == h // 2
        assert result.shape[1] == w // 2

    @pytest.mark.parametrize("fx,fy", [("2", "1"), ("1", "3"), ("1.5", "1.5")])
    def test_various_scale_factors(self, color_image, fx, fy):
        result = ScaleImage({"fx": fx, "fy": fy}).compute(color_image)
        assert result.shape[2] == color_image.shape[2]

    def test_channel_count_preserved(self, color_image):
        result = ScaleImage({"fx": "2", "fy": "2"}).compute(color_image)
        assert result.shape[2] == color_image.shape[2]

    def test_grayscale_shape_preserved(self, gray_image):
        result = ScaleImage({}).compute(gray_image)
        assert result.shape == gray_image.shape


class TestRotateImage:
    def test_default_preserves_shape(self, color_image):
        assert RotateImage({}).compute(color_image).shape == color_image.shape

    @pytest.mark.parametrize("angle", ["0", "90", "180", "270", "45"])
    def test_various_angles_preserve_shape(self, color_image, angle):
        assert RotateImage({"angle": angle}).compute(color_image).shape == color_image.shape

    def test_zero_degrees_is_identity(self):
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        img[20:40, 20:40] = [255, 0, 0]
        np.testing.assert_array_equal(RotateImage({"angle": "0"}).compute(img), img)

    def test_180_degrees_shifts_pixel(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[1, 1] = [255, 0, 0]
        result = RotateImage({"angle": "180"}).compute(img)
        assert result[99, 99, 0] > 200

    def test_custom_scale_preserves_shape(self, color_image):
        assert RotateImage({"angle": "45", "scale": "2"}).compute(color_image).shape == color_image.shape


class TestReflectImage:
    @pytest.mark.parametrize("flip_type", ["X", "Y", "Both"])
    def test_all_types_preserve_shape(self, color_image, flip_type):
        assert ReflectImage({"type": flip_type}).compute(color_image).shape == color_image.shape

    def test_default_falls_back_to_x(self, color_image):
        np.testing.assert_array_equal(
            ReflectImage({}).compute(color_image),
            ReflectImage({"type": "X"}).compute(color_image),
        )

    def test_flip_x_reverses_rows(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = [10, 20, 30]
        result = ReflectImage({"type": "X"}).compute(img)
        np.testing.assert_array_equal(result[3, 0], [10, 20, 30])

    def test_flip_y_reverses_columns(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = [10, 20, 30]
        result = ReflectImage({"type": "Y"}).compute(img)
        np.testing.assert_array_equal(result[0, 3], [10, 20, 30])

    def test_flip_both_reverses_all(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = [10, 20, 30]
        result = ReflectImage({"type": "Both"}).compute(img)
        np.testing.assert_array_equal(result[3, 3], [10, 20, 30])

    def test_double_flip_is_identity(self, color_image):
        once = ReflectImage({"type": "X"}).compute(color_image)
        twice = ReflectImage({"type": "X"}).compute(once)
        np.testing.assert_array_equal(twice, color_image)


class TestAffineImage:
    def test_color_shape_preserved(self, color_image):
        assert AffineImage({}).compute(color_image).shape == color_image.shape

    def test_gray_shape_preserved(self, gray_image):
        assert AffineImage({}).compute(gray_image).shape == gray_image.shape

    def test_translation_shifts_pixel(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[0, 0] = [255, 0, 0]
        result = AffineImage({}).compute(img)
        np.testing.assert_array_equal(result[100, 50], [255, 0, 0])

    def test_result_differs_from_input(self, color_image):
        assert not np.array_equal(AffineImage({}).compute(color_image), color_image)

    def test_top_left_is_zero_after_translation(self):
        img = np.full((200, 200, 3), 100, dtype=np.uint8)
        result = AffineImage({}).compute(img)
        assert result[0, 0, 0] == 0


class TestCropImage:
    def test_default_params_returns_full_image(self, color_image):
        result = CropImage({}).compute(color_image)
        np.testing.assert_array_equal(result, color_image)

    def test_valid_crop_reduces_size(self, color_image):
        result = CropImage({"x1": "10", "y1": "10", "x2": "60", "y2": "60"}).compute(color_image)
        assert result.shape == (50, 50, 3)

    def test_out_of_bounds_are_clamped(self, color_image):
        result = CropImage({"x1": "0", "y1": "0", "x2": "999", "y2": "999"}).compute(color_image)
        assert result.shape == color_image.shape

    def test_invalid_coordinates_returns_original(self, color_image):
        result = CropImage({"x1": "50", "y1": "50", "x2": "10", "y2": "10"}).compute(color_image)
        np.testing.assert_array_equal(result, color_image)

    def test_grayscale_crop(self, gray_image):
        result = CropImage({"x1": "0", "y1": "0", "x2": "50", "y2": "50"}).compute(gray_image)
        assert result.shape == (50, 50)

    def test_crop_preserves_dtype(self, color_image):
        result = CropImage({"x1": "10", "y1": "10", "x2": "50", "y2": "50"}).compute(color_image)
        assert result.dtype == color_image.dtype
