import importlib
from os import PathLike
from typing import Optional

from zizou.anomaly_base import AnomalyDetectionBaseClass
# from zizou.pca import PCA
from zizou.data import FeatureRequest
from zizou.feature_base import FeatureBaseClass
from zizou.rsam import RSAM
from zizou.spectral_features import SpectralFeatures
from zizou.ssam import SSAM

__all__ = [
    "AnomalyDetectionBaseClass",
    "FeatureRequest",
    "FeatureBaseClass",
    "RSAM",
    "SpectralFeatures",
    "SSAM",
    "get_data",
]


def get_data(filename: Optional[PathLike] = None) -> str:
    """Return path to zizou package.

    Parameters
    ----------
    filename : Pathlike, default None
        Append `filename` to returned path.

    Returns
    -------
    pkgdir_path

    """
    f = importlib.resources.files(__package__)
    return str(f) if filename is None else str(f / filename)
