from .DDPM import DDPM
from .EDM import EDM
from .unet import UNet

CLASSES = {
    cls.__name__: cls
    for cls in [DDPM, EDM, UNet]
}


def get_models_class(model_type, net_type):
    return CLASSES[model_type], CLASSES[net_type]
