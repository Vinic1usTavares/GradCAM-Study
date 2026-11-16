"""Grad-CAM reimplementado do zero.

Fases 0-8: carregamento de modelos, utilitarios de imagem, saliencia por
gradiente (vanilla | deconv | guided), CAM (Zhou et al.), Grad-CAM (Eqs. 1-2),
Guided Grad-CAM (`gradcam * guided backprop`), o contrafactual (negacao de
alpha em (1)) e a avaliacao em lote (pointing game + Spearman vs oclusao).
"""

from .core import (
    VGG16_LAYERS,
    cam,
    counterfactual,
    gradcam,
    resolve_cam_layers,
    resolve_layer,
)
from .eval import (
    cue_object,
    evaluation_pool,
    grid_cells,
    occlusion_correlation,
    occlusion_map,
    peak_point,
    pointing_hit,
    pool_to_grid,
    scaled_boxes,
    wilson_interval,
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
    "compose_guided_gradcam",
    "counterfactual",
    "cue_object",
    "denormalize",
    "disable_inplace_relu",
    "evaluation_pool",
    "gradcam",
    "gradcam_map_fn",
    "grid_cells",
    "guided_backprop",
    "guided_gradcam",
    "guided_gradcam_map_fn",
    "imagenet_labels",
    "input_gradient",
    "load_model",
    "occlusion_correlation",
    "occlusion_map",
    "peak_point",
    "pointing_hit",
    "pool_to_grid",
    "predict",
    "preprocess",
    "relu_mode",
    "resolve_cam_layers",
    "resolve_layer",
    "saliency",
    "saliency_map",
    "saliency_map_fn",
    "saliency_map_fns",
    "scaled_boxes",
    "set_relu_mode",
    "spearman",
    "summarize",
    "wilson_interval",
]