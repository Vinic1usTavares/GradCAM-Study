"""Grad-CAM reimplementado do zero.

Fases 0-4: carregamento de modelos, utilitarios de imagem, saliencia por
gradiente (vanilla | deconv | guided) e Grad-CAM (Eqs. 1-2).
"""

from .core import VGG16_LAYERS, gradcam, resolve_layer
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
from .relu_variants import  relu_mode, MODES, ReluVariant, set_relu_mode
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
    "class_discrimination",
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
    "resolve_layer",
    "saliency",
    "saliency_map",
    "saliency_map_fn",
    "saliency_map_fns",
    "set_relu_mode",
    "spearman",
    "summarize",
]