import cv2
import numpy as np

from app.operators.base import BaseOperator


class AffineImage(BaseOperator):
    def compute(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]

        # Source points — three points that define the origin coordinate frame.
        # Defaults form the identity-like basis at the top-left corner.
        src_pts = np.float32(
            [
                [float(self.params.get("src_x0", 0)), float(self.params.get("src_y0", 0))],
                [float(self.params.get("src_x1", 100)), float(self.params.get("src_y1", 0))],
                [float(self.params.get("src_x2", 0)), float(self.params.get("src_y2", 100))],
            ]
        )

        # Destination points — where each source point maps to after the transform.
        # Defaults translate the image by (+50, +100), matching the old behaviour.
        dst_pts = np.float32(
            [
                [float(self.params.get("dst_x0", 50)), float(self.params.get("dst_y0", 100))],
                [float(self.params.get("dst_x1", 150)), float(self.params.get("dst_y1", 100))],
                [float(self.params.get("dst_x2", 50)), float(self.params.get("dst_y2", 200))],
            ]
        )

        M = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(image, M, (cols, rows))
