"""
Compute the Real-time Seismic-Amplitude Measurement (RSAM).
"""

import logging

import numpy as np
import pandas as pd
import scipy.integrate
import xarray as xr
import yaml
from obspy import UTCDateTime

from zizou import FeatureBaseClass
from zizou.util import (
    apply_freq_filter,
    round_time,
    trace_window_data,
    trace_window_times,
)

logger = logging.getLogger(__name__)

__all__ = ["RSAM", "EnergyExplainedByRSAM"]


def rsam(arr, axis=None):
    return np.absolute(arr).mean(axis=axis)


def rsam_wideband_energy_ratio(arr_rsam, arr_wideband, axis=-1):
    energy_rsam_band = scipy.integrate.simpson(arr_rsam**2, axis=axis)
    energy_wide_band = scipy.integrate.simpson(arr_wideband**2, axis=axis)
    return energy_rsam_band / energy_wide_band


class RSAM(FeatureBaseClass):
    """
    RSAM is the mean of the absolute value of a signal filtered
    between 2 and 5 Hz.

    :param interval: Length of the time interval in seconds over
                     which to compute RSAM.
    :type interval: int, optional
    :param filtertype: The type of filtering that is applied before
                       computing RSAM. Can be either band-pass ('bp'),
                       high-pass ('hp'), or low-pass ('lp').
    :type filtertype: str, optional
    :param filterfreq: The low and high cutoff frequency. If filtertype is
                       highpass or lowpass, only the first or last value is
                       used, respectively.

    :type filterfreq: tuple, optional
    """

    features = ["rsam"]

    def __init__(
        self,
        interval=600.0,
        filtertype="bandpass",
        filterfreq=(2, 5),
        configfile=None,
        reindex=True,
    ):
        super(RSAM, self).__init__()
        self.interval = interval
        self.filtertype = filtertype
        self.filterfreq = filterfreq
        self.feature = None
        self.reindex = reindex

        if configfile is not None:
            try:
                with open(configfile, "r") as fh:
                    c = yaml.safe_load(fh)
            except OSError:
                c = yaml.safe_load(configfile)
            self.interval = c["default"].get("interval", interval)
            cr = c.get("rsam")
            if cr is not None:
                self.filtertype = cr.get("filtertype", filtertype)
                freq = cr.get("filterfreq")
                if freq is not None:
                    self.filterfreq = (freq["low"], freq["high"])
                self.reindex = cr.get("reindex", reindex)

    def compute(self, trace):
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
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime

        # handle gaps (i.e. NaNs)
        mdata = np.ma.masked_invalid(tr.data)
        tr.data = mdata
        st = tr.split()

        apply_freq_filter(st, self.filtertype, self.filterfreq)
        st.merge(fill_value=np.nan)
        st.trim(starttime, endtime, pad=True, fill_value=np.nan)
        tr = st[0]

        # initialise dataframe
        feature_data = []
        feature_idx = []

        # loop through data in interval blocks
        for trint in trace_window_data(tr, self.interval, min_len=0.8):
            absolute = np.absolute(trint.data)  # absolute value
            trint.data = absolute  # assign back to trace
            mean = trint.data.mean()
            # convert to nanometres so dealing with whole numbers
            feature_data.append(mean / 1e-9)
            feature_idx.append(trint.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S"))

        xda = xr.DataArray(
            feature_data, coords=[pd.DatetimeIndex(feature_idx)], dims="datetime"
        )
        xdf = xr.Dataset({"rsam": xda})
        xdf.attrs["starttime"] = trace.stats.starttime.isoformat()
        xdf.attrs["endtime"] = trace.stats.endtime.isoformat()
        xdf.attrs["station"] = trace.stats.station
        if self.reindex:
            starttime = UTCDateTime(str(xda.datetime.data[0]))
            starttime = round_time(starttime, self.interval)
            endtime = UTCDateTime(str(xda.datetime.data[-1]))
            endtime = round_time(endtime, self.interval)
            new_index = pd.date_range(
                starttime.datetime, endtime.datetime, freq="%dS" % int(self.interval)
            )
            xdf = xdf.reindex(dict(datetime=new_index), method="nearest")
        self.feature = xdf
        return self.feature


class EnergyExplainedByRSAM(RSAM):
    """
    The proportion of signal energy in the RSAM
    bandwidth, relative to a 0.5-10 Hz bandwidth.
    """

    features = ["rsam_energy_prop"]

    def __init__(
        self,
        interval=600.0,
        filtertype=None,
        filterfreq=(None, None),
        filtertype_wb="bandpass",
        filterfreq_wb=(0.5, 10.0),
        configfile=None,
    ):
        super(EnergyExplainedByRSAM, self).__init__(
            interval=interval,
            filtertype=filtertype,
            filterfreq=filterfreq,
            configfile=configfile,
        )
        self.filtertype_wb = filtertype_wb
        self.filterfreq_wb = filterfreq_wb

        # base params already read from configfile during superclass init
        # reopen and extract wideband filter params if any
        if configfile is not None:
            try:
                with open(configfile, "r") as fh:
                    c = yaml.safe_load(fh)
            except OSError:
                c = yaml.safe_load(configfile)
            cr = c.get("rsam")
            if cr is not None:
                self.filtertype_wb = cr.get("filtertype_wb", filtertype_wb)
                freq = cr.get("filterfreq_wb")
                if freq is not None:
                    self.filterfreq_wb = (freq["low"], freq["high"])

    def compute(self, trace):
        """
        :param trace: The seismic waveform data
        :type trace: :class:`obspy.Trace`
        """
        if len(trace) < 1:
            msg = "Trace is empty."
            raise ValueError(msg)

        logger.info(
            "Computing Energy explained by RSAM for {} between {} and {}.".format(
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
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime

        # handle gaps (i.e. NaNs)
        mdata = np.ma.masked_invalid(tr.data)
        tr.data = mdata

        st_rsam = tr.split()
        st_wb = st_rsam.copy()

        apply_freq_filter(st_rsam, self.filtertype, self.filterfreq)
        st_rsam.merge(fill_value=np.nan)
        st_rsam.trim(starttime, endtime, pad=True, fill_value=np.nan)
        tr_rsam = st_rsam[0]

        # Get trace in wide bandwidth
        apply_freq_filter(st_wb, self.filtertype_wb, self.filterfreq_wb)
        st_wb.merge(fill_value=np.nan)
        st_wb.trim(starttime, endtime, pad=True, fill_value=np.nan)
        tr_wb = st_wb[0]

        # initialise dataframe
        feature_data = []
        feature_idx = []

        # loop through data in interval blocks
        for t0, t1 in trace_window_times(tr_rsam, self.interval, min_len=0.8):
            energy_ratio = rsam_wideband_energy_ratio(
                tr_rsam.slice(t0, t1, nearest_sample=False).data,
                tr_wb.slice(t0, t1, nearest_sample=False).data,
            )
            feature_data.append(energy_ratio)
            feature_idx.append(t0.strftime("%Y-%m-%dT%H:%M:%S"))

        xdf = pd.DataFrame(
            data=feature_data,
            index=pd.DatetimeIndex(feature_idx, name="datetime"),
            columns=["rsam_energy_prop"],
            dtype=float,
        ).to_xarray()

        xdf.attrs["starttime"] = trace.stats.starttime.isoformat()
        xdf.attrs["endtime"] = trace.stats.endtime.isoformat()
        xdf.attrs["station"] = trace.stats.station

        self.feature = xdf
        self.trace = tr
        return self.feature
