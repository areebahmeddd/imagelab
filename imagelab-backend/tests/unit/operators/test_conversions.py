import cv2
import numpy as np
import pytest

from app.operators.conversions.bgr_to_hsv import BgrToHsv
from app.operators.conversions.bgr_to_lab import BgrToLab
from app.operators.conversions.bgr_to_ycrcb import BgrToYcrcb
from app.operators.conversions.channel_split import ChannelSplit
from app.operators.conversions.color_maps import ColorMaps
from app.operators.conversions.color_to_binary import ColorToBinary
from app.operators.conversions.gray_image import GrayImage
from app.operators.conversions.gray_to_binary import GrayToBinary
from app.operators.conversions.hsv_to_bgr import HsvToBgr
from app.operators.conversions.lab_to_bgr import LabToBgr
from app.operators.conversions.ycrcb_to_bgr import YcrcbToBgr


class TestGrayImage:
    def test_color_to_gray_produces_single_channel(self, color_image):
        result = GrayImage({}).compute(color_image)
        assert result.ndim == 2

    def test_output_shape_matches_hw(self, color_image):
        result = GrayImage({}).compute(color_image)
        assert result.shape == color_image.shape[:2]

    def test_dtype_is_uint8(self, color_image):
        assert GrayImage({}).compute(color_image).dtype == np.uint8


class TestChannelSplit:
    @pytest.mark.parametrize("channel,idx", [("RED", 2), ("GREEN", 1), ("BLUE", 0)])
    def test_returns_correct_channel(self, color_image, channel, idx):
        result = ChannelSplit({"channel": channel}).compute(color_image)
        assert result.ndim == 2
        expected = cv2.split(color_image)[idx]
        np.testing.assert_array_equal(result, expected)

    def test_default_is_red_channel(self, color_image):
        result = ChannelSplit({}).compute(color_image)
        expected = cv2.split(color_image)[2]  # RED is index 2 in BGR order
        np.testing.assert_array_equal(result, expected)

    def test_grayscale_input_returned_as_is(self, gray_image):
        result = ChannelSplit({"channel": "RED"}).compute(gray_image)
        np.testing.assert_array_equal(result, gray_image)


class TestGrayToBinary:
    def test_output_shape_preserved(self, gray_image):
        assert GrayToBinary({}).compute(gray_image).shape == gray_image.shape

    def test_output_values_are_binary(self, gray_image):
        result = GrayToBinary({"thresholdValue": "100", "maxValue": "255"}).compute(gray_image)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})


class TestColorToBinary:
    def test_output_is_single_channel(self, color_image):
        result = ColorToBinary({}).compute(color_image)
        assert result.ndim == 2

    def test_output_values_are_binary(self, color_image):
        result = ColorToBinary({"thresholdValue": "100", "maxValue": "255"}).compute(color_image)
        assert set(np.unique(result)).issubset({0, 255})

    def test_binary_inv_type(self, color_image):
        result = ColorToBinary(
            {"thresholdType": "threshold_binary_inv", "thresholdValue": "100", "maxValue": "255"}
        ).compute(color_image)
        assert result.ndim == 2


class TestColorMaps:
    @pytest.mark.parametrize(
        "cmap", ["HOT", "JET", "AUTUMN", "BONE", "COOL", "OCEAN", "PARULA", "PINK", "RAINBOW", "HSV"]
    )
    def test_colormap_applied(self, gray_image, cmap):
        result = ColorMaps({"type": cmap}).compute(gray_image)
        assert result.shape == (gray_image.shape[0], gray_image.shape[1], 3)

    def test_default_colormap_works(self, gray_image):
        result = ColorMaps({}).compute(gray_image)
        assert result.ndim == 3

    def test_unknown_colormap_falls_back_to_hot(self, gray_image):
        default_result = ColorMaps({"type": "HOT"}).compute(gray_image)
        fallback_result = ColorMaps({"type": "INVALID_MAP"}).compute(gray_image)
        np.testing.assert_array_equal(default_result, fallback_result)


class TestBgrToHsv:
    def test_output_shape_preserved(self, color_image):
        assert BgrToHsv({}).compute(color_image).shape == color_image.shape

    def test_accepts_grayscale_input(self, gray_image):
        result = BgrToHsv({}).compute(gray_image)
        assert result.shape == (100, 100, 3)

    def test_output_dtype(self, color_image):
        assert BgrToHsv({}).compute(color_image).dtype == np.uint8


class TestBgrToLab:
    def test_output_shape_preserved(self, color_image):
        assert BgrToLab({}).compute(color_image).shape == color_image.shape

    def test_accepts_grayscale_input(self, gray_image):
        result = BgrToLab({}).compute(gray_image)
        assert result.shape == (100, 100, 3)


class TestBgrToYcrcb:
    def test_output_shape_preserved(self, color_image):
        assert BgrToYcrcb({}).compute(color_image).shape == color_image.shape

    def test_accepts_grayscale_input(self, gray_image):
        result = BgrToYcrcb({}).compute(gray_image)
        assert result.shape == (100, 100, 3)


class TestHsvToBgr:
    def test_hsv_input_produces_bgr(self, color_image):
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        result = HsvToBgr({}).compute(hsv)
        assert result.shape == (100, 100, 3)

    def test_grayscale_input_produces_bgr(self, gray_image):
        result = HsvToBgr({}).compute(gray_image)
        assert result.shape == (100, 100, 3)

    def test_output_dtype_is_uint8(self, color_image):
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        assert HsvToBgr({}).compute(hsv).dtype == np.uint8


class TestLabToBgr:
    def test_lab_input_produces_bgr(self, color_image):
        lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        result = LabToBgr({}).compute(lab)
        assert result.shape == (100, 100, 3)

    def test_grayscale_input_produces_bgr(self, gray_image):
        result = LabToBgr({}).compute(gray_image)
        assert result.shape == (100, 100, 3)


class TestYcrcbToBgr:
    def test_ycrcb_input_produces_bgr(self, color_image):
        ycrcb = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)
        result = YcrcbToBgr({}).compute(ycrcb)
        assert result.shape == (100, 100, 3)

    def test_grayscale_input_produces_bgr(self, gray_image):
        result = YcrcbToBgr({}).compute(gray_image)
        assert result.shape == (100, 100, 3)


class TestColorspaceRoundtrip:
    def test_bgr_hsv_bgr_roundtrip(self, color_image):
        hsv = BgrToHsv({}).compute(color_image)
        recovered = HsvToBgr({}).compute(hsv)
        assert recovered.shape == color_image.shape

    def test_bgr_lab_bgr_roundtrip(self, color_image):
        lab = BgrToLab({}).compute(color_image)
        recovered = LabToBgr({}).compute(lab)
        assert recovered.shape == color_image.shape

    def test_bgr_ycrcb_bgr_roundtrip(self, color_image):
        ycrcb = BgrToYcrcb({}).compute(color_image)
        recovered = YcrcbToBgr({}).compute(ycrcb)
        assert recovered.shape == color_image.shape
