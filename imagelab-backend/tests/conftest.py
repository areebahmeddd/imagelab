import base64

import cv2
import numpy as np
import pytest


@pytest.fixture
def color_image() -> np.ndarray:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [100, 150, 200]
    return img


@pytest.fixture
def gray_image() -> np.ndarray:
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 128
    return img


@pytest.fixture
def binary_image() -> np.ndarray:
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255
    return img


@pytest.fixture
def large_color_image() -> np.ndarray:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = [80, 120, 200]
    return img


@pytest.fixture
def base64_png_image() -> str:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [0, 200, 100]
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")
