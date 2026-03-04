import cv2
import numpy as np

from app.operators.base import BaseOperator


def _to_float(value: object, param_name: str) -> float:
    """Convert *value* to float, raising ``ValueError`` with a clear message on failure.

    Blockly serialises all field values as strings, so ``"0"`` and ``0`` are both valid.
    Non-numeric input (empty string, ``None``, ``"abc"``) raises ``ValueError`` instead of
    producing a silent 500 error.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"Affine transform param '{param_name}' must be a number, got: {value!r}") from None


def _area_of_triangle(pts: np.ndarray) -> float:
    """Return twice the unsigned area of the triangle formed by *pts*.

    A result below ``1e-6`` indicates that the three points are collinear (degenerate),
    which would make ``cv2.getAffineTransform`` produce a singular / NaN matrix.
    """
    (x0, y0), (x1, y1), (x2, y2) = pts
    return abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


class AffineImage(BaseOperator):
    def compute(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]

        # Defaults reproduce the old hardcoded matrix [[1,0,50],[0,1,100]] (see issue #43
        # and tests/operators/geometric/test_affine_image.py::test_default_matches_old_hardcoded_behaviour).
        src_pts = np.float32(
            [
                [
                    _to_float(self.params.get("src_x0", 0), "src_x0"),
                    _to_float(self.params.get("src_y0", 0), "src_y0"),
                ],
                [
                    _to_float(self.params.get("src_x1", 100), "src_x1"),
                    _to_float(self.params.get("src_y1", 0), "src_y1"),
                ],
                [
                    _to_float(self.params.get("src_x2", 0), "src_x2"),
                    _to_float(self.params.get("src_y2", 100), "src_y2"),
                ],
            ]
        )

        dst_pts = np.float32(
            [
                [
                    _to_float(self.params.get("dst_x0", 50), "dst_x0"),
                    _to_float(self.params.get("dst_y0", 100), "dst_y0"),
                ],
                [
                    _to_float(self.params.get("dst_x1", 150), "dst_x1"),
                    _to_float(self.params.get("dst_y1", 100), "dst_y1"),
                ],
                [
                    _to_float(self.params.get("dst_x2", 50), "dst_x2"),
                    _to_float(self.params.get("dst_y2", 200), "dst_y2"),
                ],
            ]
        )

        if _area_of_triangle(src_pts) < 1e-6:
            raise ValueError("Source points are collinear; they must form a non-degenerate triangle.")
        if _area_of_triangle(dst_pts) < 1e-6:
            raise ValueError("Destination points are collinear; they must form a non-degenerate triangle.")

        M = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(image, M, (cols, rows))
