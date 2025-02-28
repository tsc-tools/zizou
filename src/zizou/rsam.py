"""
Compute the Real-time Seismic-Amplitude Measurement (RSAM) and the Energy explained by RSAM (rsam_energy_prop).
"""

import logging

import numpy as np
import pandas as pd
import scipy.integrate
import xarray as xr
import yaml
from obspy.signal.filter import bandpass
from obspy import Trace

from zizou import FeatureBaseClass
from zizou.util import window_array

logger = logging.getLogger(__name__)

__all__ = ["RSAM"]


def rsam_wideband_energy_ratio(arr_rsam, arr_wideband, axis=-1):
    energy_rsam_band = scipy.integrate.simpson(arr_rsam**2, axis=axis)
    energy_wide_band = scipy.integrate.simpson(arr_wideband**2, axis=axis)
    return energy_rsam_band / energy_wide_band


class RSAM(FeatureBaseClass):
    """
    Compute RSAM as the mean of the absolute value of a signal filtered
    by default between 2 and 5 Hz and the proportion of signal energy in the RSAM
    bandwidth, relative to a 0.5-10 Hz bandwidth.
 
    :param interval: Length of the time interval in seconds over
                     which to compute RSAM.
    :type interval: int, optional
    :param per_lap: If the value is less than 1 it is treated as the percentage
                    of segment overlap; else it is the step size in sample points
    :type per_lap: float or int
    :param timestamp: Can be either 'center' or 'start'. If 'center', the
                      timestamp of each value is the center of the windows
                      from which the value was computed. If 'start' it is
                      the timestamp of the first sample of the first window.
    :type timestamp: str
    :param filterfreq: The low and high cutoff frequency for bandpass filter.
    :type filterfreq: tuple, optional
    :param configfile: Configuration as a .yml file or a dictionary.
    :type configfile: str or dict
    """

    features = ["rsam", "rsam_energy_prop"]

    def __init__(
        self,
        interval=600.0,
        per_lap=0,
        timestamp="start",
        filterfreq=(2, 5),
        configfile=None,
    ):
        super(RSAM, self).__init__()
        self.interval = interval
        self.per_lap = per_lap
        self.timestamp = timestamp
        self.filterfreq = filterfreq
        self.feature = None

        if configfile is not None:
            try:
                with open(configfile, "r") as fh:
                    c = yaml.safe_load(fh)
            except OSError:
                c = yaml.safe_load(configfile)
            self.interval = c["default"].get("interval", interval)
            cr = c.get("rsam")
            if cr is not None:
                self.per_lap = cr.get("per_lap", per_lap)
                freq = cr.get("filterfreq")
                if freq is not None:
                    self.filterfreq = (freq["low"], freq["high"])

    def compute(self, trace: Trace) -> xr.Dataset:
        """
        :param trace: The seismic waveform data
        :type trace: :class:`obspy.Trace`
        """
        if len(trace) < 1:
            raise ValueError("Trace is empty.")

        logger.info(
            "Computing RSAM for {} between {} and {}.".format(
                ".".join(
                    (
                        trace.stats.network,
                        trace.stats.station,
                        trace.stats.location,
                        trace.stats.channel,
                    )
                ),
                trace.stats.starttime.isoformat(),
                trace.stats.endtime.isoformat(),
            )
        )

        tr = trace.copy()
        # convert to nanometres so dealing with whole numbers
        npts = tr.stats.npts
        Fs = tr.stats.sampling_rate

        # handle gaps (i.e. NaNs) by interpolating across gaps and
        # filling gaps at the beginning and end with zeros
        nans = np.isnan(tr.data)
        indices = np.arange(npts)
        tr.data[nans] = np.interp(indices[nans], indices[~nans],
                                  tr.data[~nans], left=0., right=0.)

        interval = min(self.interval * Fs, npts)
        if self.per_lap < 1.0:
            noverlap = int(interval * float(self.per_lap))
        else:
            noverlap = interval - self.per_lap

        tr_win = window_array(tr.data, int(interval), noverlap,
                              taper=False, return_window=False)
        tr_win = np.transpose(tr_win)
        rsam = bandpass(tr_win, self.filterfreq[0],
                        self.filterfreq[1], Fs, corners=4, zerophase=False)
        rsam_wb = bandpass(tr_win, .5, 10., Fs, corners=4, zerophase=False)
        energy_ratio = rsam_wideband_energy_ratio(rsam, rsam_wb)

        rsam = np.absolute(rsam).mean(axis=-1)
        rsam /= 1e-9

        # Calculate time stamps
        steplength = interval - noverlap
        steps = np.arange(rsam.shape[0]) * steplength
        if self.timestamp == "center":
            t = (steps + interval / 2) / Fs
        elif self.timestamp == "start":
            t = steps / Fs
        datetime = [(trace.stats.starttime + _t).datetime for _t in t] 

        xda = xr.DataArray(
            rsam, coords=[pd.DatetimeIndex(datetime)], dims="datetime"
        )
        xdb = xr.DataArray(
            energy_ratio, coords=[pd.DatetimeIndex(datetime)], dims="datetime"
        )        
        xdf = xr.Dataset({"rsam": xda, "rsam_energy_prob": xdb})
        xdf.attrs["starttime"] = trace.stats.starttime.isoformat()
        xdf.attrs["endtime"] = trace.stats.endtime.isoformat()
        xdf.attrs["station"] = trace.stats.station
        self.feature = xdf
        return self.feature
