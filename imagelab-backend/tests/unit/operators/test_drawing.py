import numpy as np

from app.operators.drawing.draw_arrow_line import DrawArrowLine
from app.operators.drawing.draw_circle import DrawCircle
from app.operators.drawing.draw_ellipse import DrawEllipse
from app.operators.drawing.draw_line import DrawLine
from app.operators.drawing.draw_rectangle import DrawRectangle
from app.operators.drawing.draw_text import DrawText

_DRAW_PARAMS = {
    "starting_point_x1": "10",
    "starting_point_y1": "10",
    "ending_point_x": "80",
    "ending_point_y": "80",
    "rgbcolors_input": "#ff0000",
    "thickness": "2",
}


class TestDrawLine:
    def test_shape_preserved(self, color_image):
        assert DrawLine(_DRAW_PARAMS).compute(color_image).shape == color_image.shape

    def test_returns_copy_not_same_object(self, color_image):
        result = DrawLine(_DRAW_PARAMS).compute(color_image)
        assert result is not color_image

    def test_draws_on_image(self, color_image):
        result = DrawLine(_DRAW_PARAMS).compute(color_image)
        assert not np.array_equal(result, color_image)

    def test_default_params_do_not_crash(self, color_image):
        assert DrawLine({}).compute(color_image).shape == color_image.shape


class TestDrawCircle:
    def test_shape_preserved(self, color_image):
        result = DrawCircle({"center_point_x": "50", "center_point_y": "50", "radius": "20"}).compute(color_image)
        assert result.shape == color_image.shape

    def test_circle_drawn_on_image(self, color_image):
        result = DrawCircle({"center_point_x": "50", "center_point_y": "50", "radius": "20"}).compute(color_image)
        assert not np.array_equal(result, color_image)

    def test_returns_copy(self, color_image):
        assert DrawCircle({}).compute(color_image) is not color_image


class TestDrawEllipse:
    def test_shape_preserved(self, color_image):
        result = DrawEllipse({"center_point_x": "50", "center_point_y": "50", "height": "30", "width": "20"}).compute(
            color_image
        )
        assert result.shape == color_image.shape

    def test_draws_on_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = DrawEllipse(
            {
                "center_point_x": "50",
                "center_point_y": "50",
                "height": "30",
                "width": "20",
                "rgbcolors_input": "#ffffff",
            }
        ).compute(img)
        assert result.sum() > 0

    def test_returns_copy(self, color_image):
        assert DrawEllipse({}).compute(color_image) is not color_image


class TestDrawRectangle:
    def test_shape_preserved(self, color_image):
        result = DrawRectangle(
            {"starting_point_x": "10", "starting_point_y": "10", "ending_point_x": "80", "ending_point_y": "80"}
        ).compute(color_image)
        assert result.shape == color_image.shape

    def test_rectangle_drawn(self, color_image):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = DrawRectangle(
            {
                "starting_point_x": "10",
                "starting_point_y": "10",
                "ending_point_x": "80",
                "ending_point_y": "80",
                "rgbcolors_input": "#ffffff",
            }
        ).compute(img)
        assert result.sum() > 0

    def test_returns_copy(self, color_image):
        assert DrawRectangle({}).compute(color_image) is not color_image


class TestDrawArrowLine:
    def test_shape_preserved(self, color_image):
        result = DrawArrowLine(
            {"starting_point_x": "10", "starting_point_y": "10", "ending_point_x": "80", "ending_point_y": "80"}
        ).compute(color_image)
        assert result.shape == color_image.shape

    def test_draws_on_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = DrawArrowLine(
            {
                "starting_point_x": "10",
                "starting_point_y": "50",
                "ending_point_x": "80",
                "ending_point_y": "50",
                "rgbcolors_input": "#ffffff",
            }
        ).compute(img)
        assert result.sum() > 0

    def test_returns_copy(self, color_image):
        assert DrawArrowLine({}).compute(color_image) is not color_image


class TestDrawText:
    def test_shape_preserved(self, color_image):
        result = DrawText({"draw_text": "test", "starting_point_x": "10", "starting_point_y": "50"}).compute(
            color_image
        )
        assert result.shape == color_image.shape

    def test_draws_on_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = DrawText(
            {"draw_text": "Hi", "starting_point_x": "10", "starting_point_y": "60", "rgbcolors_input": "#ffffff"}
        ).compute(img)
        assert result.sum() > 0

    def test_default_params_do_not_crash(self, color_image):
        result = DrawText({}).compute(color_image)
        assert result.shape == color_image.shape

    def test_returns_copy(self, color_image):
        assert DrawText({}).compute(color_image) is not color_image

    def test_custom_scale_and_thickness(self, color_image):
        result = DrawText({"scale": "2.0", "thickness": "3", "draw_text": "Hi"}).compute(color_image)
        assert result.shape == color_image.shape
