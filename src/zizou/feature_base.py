import os
import stat

import pandas as pd
from zizou.util import xarray2hdf5
import xarray as xr
from obspy import UTCDateTime


class FeatureBaseClass(object):

    __feature_name__ = "undefined"

    def __init__(self):
        self.feature = None
        self.trace = None

    def compute(self):
        msg = "compute is not implemented yet"
        raise NotImplementedError(msg)
