import base64

import numpy as np
import pytest

from app.utils.color import hex_to_bgr
from app.utils.image import decode_base64_image, encode_image_base64


class TestHexToBgr:
    @pytest.mark.parametrize(
        "hex_color, expected",
        [
            ("#ff0000", (0, 0, 255)),  # red → B=0, G=0, R=255
            ("#00ff00", (0, 255, 0)),  # green
            ("#0000ff", (255, 0, 0)),  # blue → B=255, G=0, R=0
            ("#ffffff", (255, 255, 255)),
            ("#000000", (0, 0, 0)),
            ("#2828cc", (204, 40, 40)),  # default color used in drawing ops
        ],
    )
    def test_correct_bgr_conversion(self, hex_color, expected):
        assert hex_to_bgr(hex_color) == expected

    def test_strips_leading_hash(self):
        assert hex_to_bgr("#aabbcc") == hex_to_bgr("aabbcc")

    def test_returns_tuple_of_three_ints(self):
        result = hex_to_bgr("#ffffff")
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)

    def test_channel_order_is_bgr_not_rgb(self):
        # #ff0000 is pure red: R=255, G=0, B=0 → in BGR order: (0, 0, 255)
        b, g, r = hex_to_bgr("#ff0000")
        assert b == 0
        assert g == 0
        assert r == 255


class TestDecodeBase64Image:
    def test_valid_png_returns_ndarray(self, base64_png_image):
        result = decode_base64_image(base64_png_image)
        assert isinstance(result, np.ndarray)

    def test_decoded_image_has_correct_shape(self, base64_png_image):
        result = decode_base64_image(base64_png_image)
        assert result.shape == (100, 100, 3)

    def test_decoded_image_dtype_is_uint8(self, base64_png_image):
        result = decode_base64_image(base64_png_image)
        assert result.dtype == np.uint8

    def test_invalid_base64_raises(self):
        with pytest.raises(ValueError):
            decode_base64_image("!not_base64!!!")

    def test_valid_base64_of_non_image_raises_value_error(self):
        garbage = base64.b64encode(b"this is not an image").decode()
        with pytest.raises(ValueError, match="Could not decode"):
            decode_base64_image(garbage)


class TestEncodeImageBase64:
    def test_returns_string(self, color_image):
        assert isinstance(encode_image_base64(color_image), str)

    def test_result_is_non_empty(self, color_image):
        assert len(encode_image_base64(color_image)) > 0

    def test_result_is_valid_base64(self, color_image):
        encoded = encode_image_base64(color_image)
        decoded_bytes = base64.b64decode(encoded)
        assert len(decoded_bytes) > 0

    def test_roundtrip_preserves_shape(self, color_image):
        encoded = encode_image_base64(color_image, "png")
        decoded = decode_base64_image(encoded)
        assert decoded.shape == color_image.shape

    def test_grayscale_roundtrip(self, gray_image):
        encoded = encode_image_base64(gray_image, "png")
        decoded = decode_base64_image(encoded)
        # OpenCV reloads single-channel PNG as single-channel
        assert decoded.ndim in (2, 3)
