from enum import Enum
from typing import Dict, Tuple


class Resolutions(Enum):
    Mini = 'mini'
    Half = 'half'
    Full = 'full'
    TransFuseMini = 'trans-fuse_mini'

    def __str__(self):
        return self.value


shape_by_resolution: Dict[Resolutions, Tuple[int, int]] = {
    Resolutions.Full: (480, 640),
    Resolutions.Half: (240, 320),
    Resolutions.Mini: (224, 224),
    Resolutions.TransFuseMini: (192, 256),
}
