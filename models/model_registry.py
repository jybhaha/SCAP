"""Single torchvision factory for the nine classifier interfaces in the paper."""

from __future__ import annotations

from torchvision import models

from models.imagenet_model import ensure_imagenet_normalized


MODEL_SPECS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "vgg16": (models.vgg16, models.VGG16_Weights),
    "densenet121": (models.densenet121, models.DenseNet121_Weights),
    "inception_v3": (models.inception_v3, models.Inception_V3_Weights),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights),
    "swin_t": (models.swin_t, models.Swin_T_Weights),
}
MODEL_NAMES = tuple(MODEL_SPECS)


def available_models():
    return MODEL_NAMES


def default_weight_identifier(name):
    """Return the explicit default torchvision weight enum member name."""
    key = name.lower()
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    weight_enum = MODEL_SPECS[key][1]
    return f"{weight_enum.__name__}.{weight_enum.DEFAULT.name}"


def build_model(name, weights="DEFAULT", device=None, normalize=True, **kwargs):
    """Build one classifier; pass ``weights=None`` to guarantee no download.

    The default return value includes the same ImageNet preprocessing used by
    every attack and evaluation stage. Pass ``normalize=False`` only for custom
    pipelines that explicitly manage preprocessing themselves.
    """
    key = name.lower()
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    constructor, weight_enum = MODEL_SPECS[key]
    if isinstance(weights, str):
        try:
            weights = weight_enum.DEFAULT if weights == "DEFAULT" else weight_enum[weights]
        except KeyError as exc:
            choices = [member.name for member in weight_enum]
            raise ValueError(f"Unknown weights {weights!r} for {key}; choose from {choices} or None") from exc
    model = constructor(weights=weights, **kwargs)
    if normalize:
        input_size = (299, 299) if key == "inception_v3" else (224, 224)
        model = ensure_imagenet_normalized(model, input_size=input_size)
    if device is not None:
        model = model.to(device)
    return model.eval()
