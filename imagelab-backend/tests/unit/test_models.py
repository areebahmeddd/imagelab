import pytest
from pydantic import ValidationError

from app.models.pipeline import PipelineRequest, PipelineResponse, PipelineStep


class TestPipelineStep:
    def test_requires_type_field(self):
        with pytest.raises(ValidationError):
            PipelineStep()

    def test_default_params_is_empty_dict(self):
        step = PipelineStep(type="some_op")
        assert step.params == {}

    def test_custom_params_accepted(self):
        step = PipelineStep(type="blur", params={"kernelSize": "5"})
        assert step.params["kernelSize"] == "5"

    def test_type_stored_as_string(self):
        step = PipelineStep(type="blurring_applyblur")
        assert step.type == "blurring_applyblur"


class TestPipelineRequest:
    def test_requires_image_field(self):
        with pytest.raises(ValidationError):
            PipelineRequest(pipeline=[])

    def test_requires_pipeline_field(self):
        with pytest.raises(ValidationError):
            PipelineRequest(image="abc123")

    def test_default_image_format_is_png(self):
        req = PipelineRequest(image="img", pipeline=[])
        assert req.image_format == "png"

    def test_custom_format_accepted(self):
        req = PipelineRequest(image="img", image_format="jpeg", pipeline=[])
        assert req.image_format == "jpeg"

    def test_pipeline_accepts_list_of_steps(self):
        step = PipelineStep(type="blur")
        req = PipelineRequest(image="img", pipeline=[step])
        assert len(req.pipeline) == 1

    def test_empty_pipeline_is_valid(self):
        req = PipelineRequest(image="img", pipeline=[])
        assert req.pipeline == []


class TestPipelineResponse:
    def test_success_response_structure(self):
        resp = PipelineResponse(success=True, image="base64data", image_format="png")
        assert resp.success is True
        assert resp.error is None
        assert resp.step is None

    def test_failure_response_structure(self):
        resp = PipelineResponse(success=False, error="Operator not found", step=2)
        assert resp.success is False
        assert resp.image is None
        assert resp.step == 2

    def test_all_fields_optional_except_success(self):
        resp = PipelineResponse(success=True)
        assert resp.image is None
        assert resp.image_format is None
        assert resp.error is None
        assert resp.step is None
