import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client):
        data = client.get("/api/health").json()
        assert data == {"status": "ok"}


class TestPipelineExecuteSuccess:
    def test_empty_pipeline_returns_200(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={"image": base64_png_image, "image_format": "png", "pipeline": []},
        )
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={"image": base64_png_image, "pipeline": []},
        )
        data = resp.json()
        assert "success" in data

    def test_success_response_has_image(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={"image": base64_png_image, "pipeline": []},
        )
        data = resp.json()
        assert data["success"] is True
        assert data["image"] is not None

    def test_single_blur_step(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={
                "image": base64_png_image,
                "pipeline": [{"type": "blurring_applyblur", "params": {"widthSize": "3", "heightSize": "3"}}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_multi_step_pipeline(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={
                "image": base64_png_image,
                "pipeline": [
                    {"type": "blurring_applyblur"},
                    {"type": "imageconvertions_grayimage"},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_image_format_preserved_in_response(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={"image": base64_png_image, "image_format": "png", "pipeline": []},
        )
        assert resp.json()["image_format"] == "png"

    def test_noop_blocks_skipped(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={
                "image": base64_png_image,
                "pipeline": [
                    {"type": "basic_readimage"},
                    {"type": "basic_writeimage"},
                ],
            },
        )
        assert resp.json()["success"] is True

    def test_geometric_operator(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={
                "image": base64_png_image,
                "pipeline": [{"type": "geometric_scaleimage", "params": {"fx": "0.5", "fy": "0.5"}}],
            },
        )
        assert resp.json()["success"] is True


class TestPipelineExecuteErrors:
    def test_unknown_operator_returns_200_with_failure(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={
                "image": base64_png_image,
                "pipeline": [{"type": "totally_unknown_op"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_failure_response_has_error_message(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={
                "image": base64_png_image,
                "pipeline": [{"type": "bad_op"}],
            },
        )
        assert resp.json().get("error") is not None

    def test_failure_response_image_is_none(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={
                "image": base64_png_image,
                "pipeline": [{"type": "bad_op"}],
            },
        )
        assert resp.json().get("image") is None

    def test_invalid_base64_image_returns_failure(self, client):
        resp = client.post(
            "/api/pipeline/execute",
            json={"image": "not-valid-base64!!!", "pipeline": []},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_unknown_operator_step_index_reported(self, client, base64_png_image):
        resp = client.post(
            "/api/pipeline/execute",
            json={
                "image": base64_png_image,
                "pipeline": [
                    {"type": "blurring_applyblur"},
                    {"type": "bad_second_op"},
                ],
            },
        )
        data = resp.json()
        assert data["success"] is False
        assert data.get("step") == 1

    def test_missing_image_field_returns_422(self, client):
        resp = client.post(
            "/api/pipeline/execute",
            json={"pipeline": []},
        )
        assert resp.status_code == 422

    def test_missing_pipeline_field_returns_422(self, client):
        resp = client.post(
            "/api/pipeline/execute",
            json={"image": "abc"},
        )
        assert resp.status_code == 422
