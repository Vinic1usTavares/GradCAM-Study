"""Grad-CAM reimplementado do zero.

Fases 0-2 por enquanto: carregamento de modelos, utilitarios de imagem e
saliencia por gradiente (vanilla | deconv | guided).
"""

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
from .metrics import (
    MapFn,
    class_discrimination,
    saliency_map_fn,
    saliency_map_fns,
    spearman,
    summarize,
)
from .relu_variants import MODES, ReluVariant, set_relu_mode
from .saliency import guided_backprop, input_gradient, saliency, saliency_map

__all__ = [
    "COCO_DIR",
    "MEAN",
    "MODES",
    "STD",
    "WEIGHTS_DIR",
    "MapFn",
    "ReluVariant",
    "class_discrimination",
    "denormalize",
    "disable_inplace_relu",
    "guided_backprop",
    "imagenet_labels",
    "input_gradient",
    "load_model",
    "predict",
    "preprocess",
    "saliency",
    "saliency_map",
    "saliency_map_fn",
    "saliency_map_fns",
    "set_relu_mode",
    "spearman",
    "summarize",
]