"""Qualtran circuits for the midpoint-dropped signed inverse-square SLAC operator."""

from .circuits import (
    CompleteInverseSquareBlockEncoding,
    CompleteMultiscaleInverseSquare,
    PrefixMultiscaleShannonQSWT,
    WaveletScalePreconditioner,
    inverse_square_column,
    slac_inverse_square_alpha,
)

__version__ = "0.1.0"

__all__ = [
    "CompleteInverseSquareBlockEncoding",
    "CompleteMultiscaleInverseSquare",
    "PrefixMultiscaleShannonQSWT",
    "WaveletScalePreconditioner",
    "inverse_square_column",
    "slac_inverse_square_alpha",
]
