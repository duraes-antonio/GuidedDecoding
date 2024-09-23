from enum import Enum
from typing import Dict, Tuple


class Resolutions(Enum):
    Mini = 'mini'
    Half = 'half'
    Full = 'full'
    HalfSquare = 'half-square'
    Half2 = 'half2'
    Full2 = 'full2'
    Square384 = 'square-384'
    Square224 = 'square-224'

    def __str__(self):
        return self.value


# (height, width)
shape_by_resolution: Dict[Resolutions, Tuple[int, int]] = {
    # (15, 20)
    Resolutions.Full: (480, 640),

    Resolutions.Half: (240, 320),

    Resolutions.HalfSquare: (320, 320),

    # (7, 7)
    Resolutions.Mini: (224, 224),

    # (9, 12)
    Resolutions.Half2: (288, 384),

    Resolutions.Square224: (224, 224),
    Resolutions.Square384: (384, 384),

    # (13, 17)
    Resolutions.Full2: (416, 544),
}
