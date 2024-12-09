import importlib
from os import PathLike
from typing import Optional

try:
    from .anomaly_base import (
        AnomalyDetectionBaseClass,
        SliceBatchSampler,
        ZizouDataset,
        min_max_scale_netcdf,
    )
    from .autoencoder import AutoEncoder
except (ModuleNotFoundError, ImportError):
    pass

from .data import DataSource, FDSNWaveforms, MockSDSWaveforms, S3Waveforms, SDSWaveforms
from .dsar import DSAR
from .feature_base import FeatureBaseClass
from .rsam import RSAM, EnergyExplainedByRSAM
from .spectral_features import SpectralFeatures
from .ssam import SSAM


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
