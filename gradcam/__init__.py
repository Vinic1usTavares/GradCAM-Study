"""Grad-CAM reimplementado do zero.

Fases 0-10: carregamento de modelos, utilitarios de imagem, saliencia por
gradiente (vanilla | deconv | guided), CAM (Zhou et al.), Grad-CAM (Eqs. 1-2),
Guided Grad-CAM (`gradcam * guided backprop`), o contrafactual (negacao de
alpha em (1)) e o cascade de randomizacao de parametros (Adebayo et al.).
"""

from .core import (
    VGG16_LAYERS,
    cam,
    counterfactual,
    gradcam,
    resolve_cam_layers,
    resolve_layer,
)
from .guided import compose_guided_gradcam, guided_gradcam
from .metrics import (
    MapFn,
    class_discrimination,
    gradcam_map_fn,
    guided_gradcam_map_fn,
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
from .sanity import (
    DEFAULT_CUTOFFS,
    cascade,
    parameter_modules,
    randomization_order,
    reinitialize_,
    sanity_curve,
)

__all__ = [
    "COCO_DIR",
    "DEFAULT_CUTOFFS",
    "MEAN",
    "MODES",
    "STD",
    "VGG16_LAYERS",
    "WEIGHTS_DIR",
    "MapFn",
    "ReluVariant",
    "cam",
    "cascade",
    "class_discrimination",
    "compose_guided_gradcam",
    "counterfactual",
    "denormalize",
    "disable_inplace_relu",
    "gradcam",
    "gradcam_map_fn",
    "guided_backprop",
    "guided_gradcam",
    "guided_gradcam_map_fn",
    "imagenet_labels",
    "input_gradient",
    "load_model",
    "parameter_modules",
    "predict",
    "preprocess",
    "randomization_order",
    "reinitialize_",
    "relu_mode",
    "resolve_cam_layers",
    "resolve_layer",
    "saliency",
    "saliency_map",
    "saliency_map_fn",
    "saliency_map_fns",
    "sanity_curve",
    "set_relu_mode",
    "spearman",
    "summarize",
]