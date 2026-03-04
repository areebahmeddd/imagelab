import cv2
import numpy as np
import pytest

from app.operators.geometric.affine_image import AffineImage


def _solid_image(rows: int = 200, cols: int = 200, channels: int = 3) -> np.ndarray:
    """Return a plain non-zero image so pixel comparisons are meaningful."""
    img = np.zeros((rows, cols, channels), dtype=np.uint8)
    # Paint a small coloured square near the origin so translation is visible.
    img[10:50, 10:50] = [255, 128, 0]
    return img


class TestAffineImageDefaults:
    def test_output_shape_preserved(self):
        """warpAffine must return the same (rows, cols) as the input."""
        img = _solid_image(200, 300)
        result = AffineImage(params={}).compute(img)
        assert result.shape == img.shape

    def test_output_dtype_preserved(self):
        img = _solid_image()
        result = AffineImage(params={}).compute(img)
        assert result.dtype == img.dtype

    def test_default_translates_content(self):
        """Defaults map src→dst by (+50, +100), so the painted square
        should shift to row+100, col+50 in the output."""
        img = _solid_image(300, 300)
        result = AffineImage(params={}).compute(img)

        # Original square centre is around (30, 30); after translation it
        # should be around (130, 80).  Confirm destination area is non-zero
        # and source area has been moved away.
        translated_patch = result[110:150, 60:100]
        assert translated_patch.max() > 0, "Translated content should be visible at destination"

        # The original location should now be empty (content translated away).
        source_patch = result[10:50, 10:50]
        assert source_patch.max() == 0, "Source region should be empty after translation"

    def test_default_matches_old_hardcoded_behaviour(self):
        """Default params must reproduce the matrix [[1,0,50],[0,1,100]] within ±1 LSB.

        Uses ``assert_allclose`` rather than ``assert_array_equal`` because
        ``cv2.getAffineTransform`` (float32 path) and a raw float64 matrix can differ
        by up to 1 LSB after ``warpAffine`` bilinear interpolation on some OpenCV builds.
        """
        img = _solid_image(300, 300)
        expected = cv2.warpAffine(img, np.float64([[1, 0, 50], [0, 1, 100]]), (300, 300))
        result = AffineImage(params={}).compute(img)
        np.testing.assert_allclose(
            result.astype(np.float64),
            expected.astype(np.float64),
            atol=1,
            err_msg="Default params should reproduce the old hardcoded translation",
        )


class TestAffineImageCustomParams:
    def test_identity_transform_leaves_image_unchanged(self):
        """src == dst  →  identity affine  →  output == input."""
        params = {
            "src_x0": 0,
            "src_y0": 0,
            "src_x1": 100,
            "src_y1": 0,
            "src_x2": 0,
            "src_y2": 100,
            "dst_x0": 0,
            "dst_y0": 0,
            "dst_x1": 100,
            "dst_y1": 0,
            "dst_x2": 0,
            "dst_y2": 100,
        }
        img = _solid_image()
        result = AffineImage(params=params).compute(img)
        np.testing.assert_array_equal(result, img)

    def test_custom_translation(self):
        """Translate by (+10, +20) and confirm content moved."""
        params = {
            "src_x0": 0,
            "src_y0": 0,
            "src_x1": 100,
            "src_y1": 0,
            "src_x2": 0,
            "src_y2": 100,
            "dst_x0": 10,
            "dst_y0": 20,
            "dst_x1": 110,
            "dst_y1": 20,
            "dst_x2": 10,
            "dst_y2": 120,
        }
        img = _solid_image(200, 200)
        result = AffineImage(params=params).compute(img)
        # Content that was at (10-50, 10-50) should now be at (30-70, 20-60).
        assert result[30:70, 20:60].max() > 0

    def test_different_params_produce_different_output(self):
        """Two distinct transforms must not produce identical images."""
        img = _solid_image()

        result_a = AffineImage(params={}).compute(img)

        custom_params = {
            "src_x0": 0,
            "src_y0": 0,
            "src_x1": 100,
            "src_y1": 0,
            "src_x2": 0,
            "src_y2": 100,
            "dst_x0": 20,
            "dst_y0": 30,
            "dst_x1": 120,
            "dst_y1": 30,
            "dst_x2": 20,
            "dst_y2": 130,
        }
        result_b = AffineImage(params=custom_params).compute(img)

        assert not np.array_equal(result_a, result_b)

    def test_string_param_values_are_coerced(self):
        """Blockly sends all field values as strings; the operator must coerce them."""
        params = {
            "src_x0": "0",
            "src_y0": "0",
            "src_x1": "100",
            "src_y1": "0",
            "src_x2": "0",
            "src_y2": "100",
            "dst_x0": "0",
            "dst_y0": "0",
            "dst_x1": "100",
            "dst_y1": "0",
            "dst_x2": "0",
            "dst_y2": "100",
        }
        img = _solid_image()
        result = AffineImage(params=params).compute(img)
        np.testing.assert_array_equal(result, img)


class TestAffineImageEdgeCases:
    def test_grayscale_image(self):
        """Operator must handle single-channel (H, W) images."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[10:40, 10:40] = 200
        result = AffineImage(params={}).compute(img)
        assert result.shape == img.shape

    def test_small_image(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = AffineImage(params={}).compute(img)
        assert result.shape == img.shape

    def test_collinear_source_points_raises(self):
        """Three collinear src points produce a singular transform; operator must raise."""
        params = {
            # All three source points lie on y=0 — degenerate triangle.
            "src_x0": 0,
            "src_y0": 0,
            "src_x1": 50,
            "src_y1": 0,
            "src_x2": 100,
            "src_y2": 0,
            "dst_x0": 0,
            "dst_y0": 0,
            "dst_x1": 50,
            "dst_y1": 50,
            "dst_x2": 100,
            "dst_y2": 100,
        }
        img = _solid_image()
        with pytest.raises(ValueError, match="collinear"):
            AffineImage(params=params).compute(img)

    def test_collinear_destination_points_raises(self):
        """Three collinear dst points produce a singular transform; operator must raise."""
        params = {
            "src_x0": 0,
            "src_y0": 0,
            "src_x1": 50,
            "src_y1": 50,
            "src_x2": 100,
            "src_y2": 100,
            # All three destination points lie on y=0 — degenerate triangle.
            "dst_x0": 0,
            "dst_y0": 0,
            "dst_x1": 50,
            "dst_y1": 0,
            "dst_x2": 100,
            "dst_y2": 0,
        }
        img = _solid_image()
        with pytest.raises(ValueError, match="collinear"):
            AffineImage(params=params).compute(img)

    def test_invalid_param_type_raises(self):
        """Non-numeric param values must raise ``ValueError`` with a clear message."""
        params = {
            "src_x0": "not-a-number",
            "src_y0": 0,
            "src_x1": 100,
            "src_y1": 0,
            "src_x2": 0,
            "src_y2": 100,
            "dst_x0": 50,
            "dst_y0": 100,
            "dst_x1": 150,
            "dst_y1": 100,
            "dst_x2": 50,
            "dst_y2": 200,
        }
        img = _solid_image()
        with pytest.raises(ValueError, match="src_x0"):
            AffineImage(params=params).compute(img)
