from enum import Enum, auto


class Environment(Enum):
    KAGGLE = auto(),
    COLAB = auto(),
    LOCAL = auto(),
