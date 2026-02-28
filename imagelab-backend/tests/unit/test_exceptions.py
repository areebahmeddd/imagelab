import pytest

from app.exceptions import AppException


class TestAppException:
    def test_is_exception_subclass(self):
        assert issubclass(AppException, Exception)

    def test_default_status_code_is_400(self):
        exc = AppException("something went wrong")
        assert exc.status_code == 400

    def test_custom_status_code(self):
        exc = AppException("not found", status_code=404)
        assert exc.status_code == 404

    def test_message_attribute(self):
        exc = AppException("test message")
        assert exc.message == "test message"

    def test_str_representation_contains_message(self):
        exc = AppException("my error")
        assert "my error" in str(exc)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(AppException) as exc_info:
            raise AppException("boom", status_code=503)
        assert exc_info.value.status_code == 503
        assert exc_info.value.message == "boom"
