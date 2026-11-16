"""Grad-CAM reimplementado do zero.

Fases 0-7: carregamento de modelos, utilitarios de imagem, saliencia por
gradiente (vanilla | deconv | guided), Grad-CAM (Eqs. 1-2), CAM e o
contrafactual (negacao de alpha em (1)).
"""

from .core import (
    VGG16_LAYERS,
    cam,
    counterfactual,
    gradcam,
    resolve_cam_layers,
    resolve_layer,
)
from .metrics import (
    MapFn,
    class_discrimination,
    gradcam_map_fn,
    saliency_map_fn,
    saliency_map_fns,
    spearman,
    summarize,
)
from .models import (
    COCO_DIR,
    MEAN,
    STD,
    WEIGHTS_DIR,
    denormalize,
    disable_inplace_relu,
    imagenet_labels,
    load_model,
    predict,
    preprocess,
)
from .relu_variants import MODES, ReluVariant, relu_mode, set_relu_mode
from .saliency import guided_backprop, input_gradient, saliency, saliency_map

__all__ = [
    "COCO_DIR",
    "MEAN",
    "MODES",
    "STD",
    "VGG16_LAYERS",
    "WEIGHTS_DIR",
    "MapFn",
    "ReluVariant",
    "cam",
    "class_discrimination",
    "counterfactual",
    "denormalize",
    "disable_inplace_relu",
    "gradcam",
    "gradcam_map_fn",
    "guided_backprop",
    "imagenet_labels",
    "input_gradient",
    "load_model",
    "predict",
    "preprocess",
    "relu_mode",
    "resolve_cam_layers",
    "resolve_layer",
    "saliency",
    "saliency_map",
    "saliency_map_fn",
    "saliency_map_fns",
    "set_relu_mode",
    "spearman",
    "summarize",
]