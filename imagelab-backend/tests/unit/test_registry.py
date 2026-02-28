import pytest

from app.operators.base import BaseOperator
from app.operators.registry import OPERATOR_REGISTRY, get_operator

ALL_REGISTERED_KEYS = [
    # Blurring
    "blurring_applyblur",
    "blurring_applygaussianblur",
    "blurring_applymedianblur",
    # Geometric
    "geometric_reflectimage",
    "geometric_rotateimage",
    "geometric_scaleimage",
    "geometric_affineimage",
    "geometric_cropimage",
    # Conversions
    "imageconvertions_grayimage",
    "imageconvertions_channelsplit",
    "imageconvertions_graytobinary",
    "imageconvertions_colortobinary",
    "imageconvertions_colormaps",
    "imageconvertions_bgrtohsv",
    "imageconvertions_bgrtolab",
    "imageconvertions_bgrtoycrcb",
    "imageconvertions_hsvtobgr",
    "imageconvertions_labtobgr",
    "imageconvertions_ycrcbtobgr",
    # Drawing
    "drawingoperations_drawline",
    "drawingoperations_drawcircle",
    "drawingoperations_drawellipse",
    "drawingoperations_drawrectangle",
    "drawingoperations_drawarrowline",
    "drawingoperations_drawtext",
    # Filtering
    "filtering_boxfilter",
    "filtering_bilateral",
    "filtering_sharpen",
    "filtering_pyramidup",
    "filtering_pyramiddown",
    "filtering_erosion",
    "filtering_dilation",
    "filtering_morphological",
    # Thresholding
    "thresholding_applythreshold",
    "thresholding_adaptivethreshold",
    "thresholding_applyborders",
    "thresholding_otsuthreshold",
    # Sobel derivatives
    "sobelderivatives_soblederivate",
    "sobelderivatives_scharrderivate",
    # Transformation
    "transformation_distance",
    "transformation_laplacian",
]


class TestOperatorRegistry:
    @pytest.mark.parametrize("key", ALL_REGISTERED_KEYS)
    def test_operator_is_registered(self, key):
        assert get_operator(key) is not None, f"'{key}' missing from registry"

    def test_unknown_key_returns_none(self):
        assert get_operator("does_not_exist") is None

    def test_empty_string_returns_none(self):
        assert get_operator("") is None

    def test_all_registered_values_are_base_operator_subclasses(self):
        for key, cls in OPERATOR_REGISTRY.items():
            assert issubclass(cls, BaseOperator), f"{key}: {cls} is not a BaseOperator subclass"

    def test_registry_is_not_empty(self):
        assert len(OPERATOR_REGISTRY) > 0

    def test_registry_has_expected_size(self):
        assert len(OPERATOR_REGISTRY) >= len(ALL_REGISTERED_KEYS)

    def test_all_registered_classes_are_instantiable(self):
        for _key, cls in OPERATOR_REGISTRY.items():
            instance = cls({})
            assert isinstance(instance, BaseOperator)
