import base64

from app.models.pipeline import PipelineRequest, PipelineStep
from app.services.pipeline_executor import execute_pipeline


class TestPipelineExecutorSuccess:
    def test_empty_pipeline_returns_success(self, base64_png_image):
        req = PipelineRequest(image=base64_png_image, pipeline=[])
        resp = execute_pipeline(req)
        assert resp.success is True
        assert resp.image is not None
        assert resp.error is None

    def test_image_format_preserved(self, base64_png_image):
        req = PipelineRequest(image=base64_png_image, image_format="png", pipeline=[])
        resp = execute_pipeline(req)
        assert resp.image_format == "png"

    def test_single_blur_step(self, base64_png_image):
        req = PipelineRequest(
            image=base64_png_image,
            pipeline=[PipelineStep(type="blurring_applyblur", params={"widthSize": "3", "heightSize": "3"})],
        )
        resp = execute_pipeline(req)
        assert resp.success is True
        assert resp.image is not None

    def test_multi_step_chain(self, base64_png_image):
        req = PipelineRequest(
            image=base64_png_image,
            pipeline=[
                PipelineStep(type="blurring_applyblur", params={"widthSize": "3", "heightSize": "3"}),
                PipelineStep(type="imageconvertions_grayimage"),
            ],
        )
        resp = execute_pipeline(req)
        assert resp.success is True
        assert resp.image is not None

    def test_noop_types_are_skipped(self, base64_png_image):
        req = PipelineRequest(
            image=base64_png_image,
            pipeline=[
                PipelineStep(type="basic_readimage"),
                PipelineStep(type="basic_writeimage"),
            ],
        )
        resp = execute_pipeline(req)
        assert resp.success is True

    def test_geometric_scale_step(self, base64_png_image):
        req = PipelineRequest(
            image=base64_png_image,
            pipeline=[PipelineStep(type="geometric_scaleimage", params={"fx": "0.5", "fy": "0.5"})],
        )
        resp = execute_pipeline(req)
        assert resp.success is True

    def test_response_contains_base64_image(self, base64_png_image):
        req = PipelineRequest(image=base64_png_image, pipeline=[])
        resp = execute_pipeline(req)
        decoded = base64.b64decode(resp.image)
        assert len(decoded) > 0


class TestPipelineExecutorErrors:
    def test_unknown_operator_returns_failure(self, base64_png_image):
        req = PipelineRequest(
            image=base64_png_image,
            pipeline=[PipelineStep(type="nonexistent_op")],
        )
        resp = execute_pipeline(req)
        assert resp.success is False
        assert resp.error is not None

    def test_unknown_operator_reports_correct_step_index(self, base64_png_image):
        req = PipelineRequest(
            image=base64_png_image,
            pipeline=[
                PipelineStep(type="blurring_applyblur"),
                PipelineStep(type="bad_op"),
            ],
        )
        resp = execute_pipeline(req)
        assert resp.success is False
        assert resp.step == 1

    def test_invalid_base64_returns_failure_at_step_0(self):
        req = PipelineRequest(image="!!!not_base64!!!", pipeline=[])
        resp = execute_pipeline(req)
        assert resp.success is False
        assert resp.step == 0

    def test_failure_has_nonempty_error_message(self, base64_png_image):
        req = PipelineRequest(
            image=base64_png_image,
            pipeline=[PipelineStep(type="does_not_exist")],
        )
        resp = execute_pipeline(req)
        assert resp.error != ""
        assert resp.image is None
