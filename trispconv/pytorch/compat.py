from enum import Enum


class ConvAlgo(Enum):
    Native = 0
    MaskImplicitGemm = 1
    MaskSplitImplicitGemm = 2
