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

    def save(self, h5FileName):
        """
        Save data to disk as netcdf4 files. Filenames will
        be 'YYYYMMDD_{sitename}.nc' where the date is taking
        from the xarray attributes 'starttime' and 'endtime'.

        :param h5FileName: Directory to save files to.
        :type h5FileName: str
        :param overwrite: Overwrite instead of merge if the
                          file already exists.
        :type overwrite: boolean
        """
        if self.feature is None:
            msg = "self.feature is None"
            raise TypeError(msg)

        xarray2hdf5(self.feature, h5FileName)

    def compute(self):
        msg = "compute is not implemented yet"
        raise NotImplementedError(msg)
