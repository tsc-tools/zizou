"""
Compute the displacement seismic amplitude ratio (DSAR).
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from obspy import UTCDateTime

from .feature_base import FeatureBaseClass
from .util import round_time, trace_window_times

logger = logging.getLogger(__name__)


class DSAR(FeatureBaseClass):
    """
    DSAR is the ratio of
    low frequency to high frequency amplitudes, as suggested in:
    https://doi.org/10.1130/G46107.1

    The time-domain displacement signal is bandpass filtered in two
    frequency bands, 4.5 to 8 Hz and 8 to 16 Hz, and DSAR is defined
    as the median ratio of the absolute signals. The quantity is
    interpreted in the publication as a change in seismic attenuation
    in the vicinity of the seismic station.
    """

    features = ["dsar"]

    def __init__(
        self,
        interval=600.0,
        filtertype="bandpass",
        lowerfreqband=(4.5, 8),
        higherfreqband=(8, 16),
        configfile=None,
        reindex=True,
    ):
        super(DSAR, self).__init__()
        self.interval = float(interval)
        self.filtertype = filtertype
        self.lowerfreqband = lowerfreqband
        self.higherfreqband = higherfreqband
        self.reindex = reindex

        if configfile is not None:
            try:
                with open(configfile, "r") as fh:
                    c = yaml.safe_load(fh)
            except OSError:
                c = yaml.safe_load(configfile)
            self.interval = c["default"].get("interval", interval)
            cd = c.get("dsar")
            if cd is not None:
                self.reindex = cd.get("reindex", reindex)
                self.filtertype = cd.get("filtertype", filtertype)
                lfreq = cd.get("lowerfreqband")
                if lfreq is not None:
                    self.lowerfreqband = (lfreq["low"], lfreq["high"])
                hfreq = cd.get("higherfreqband")
                if hfreq is not None:
                    self.higherfreqband = (hfreq["low"], hfreq["high"])

    def compute(self, trace):
        """
        Compute the displacement seismic amplitude ratio (DSAR).

        # TODO:
        Publication is quite vague and doesn't have enough detail
        to understand the method. Unclear whether median is every
        10 minutes or every day. This affects the bootstrap calc-
        ulation too, but this might not be necessary..

        :param stream: The SLCN code for the seismic stream
        :type stream: str
        :param startdate: Datetime of the first sample.
        :type startdate: :class:`obspy.UTCDateTime`
        :param enddate: Datetime of the last sample.
        :type enddate: :class:`obspy.UTCDateTime`
        :param interval: Length of the time interval in seconds over
                         which to compute DSAR (default 10 mins)
        :type interval: int, optional
        """
        if len(trace) < 1:
            raise ValueError("Trace is empty.")

        logger.info(
            "Computing DSAR for {} between {} and {}.".format(
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

        tr_lf = trace.copy()
        # handle gaps (i.e. NaNs)
        mdata = np.ma.masked_invalid(tr_lf.data)
        tr_lf.data = mdata
        # Get second trace for alternative filtering
        tr_hf = tr_lf.copy()

        # Filter settings
        lf_f1, lf_f2 = self.lowerfreqband
        hf_f1, hf_f2 = self.higherfreqband
        st_lf = tr_lf.split()
        st_hf = tr_hf.split()
        # Integrate to displacement
        st_lf.integrate("cumtrapz")
        st_hf.integrate("cumtrapz")

        st_lf.filter(
            self.filtertype, freqmin=lf_f1, freqmax=lf_f2, corners=4, zerophase=False
        )
        st_hf.filter(
            self.filtertype, freqmin=hf_f1, freqmax=hf_f2, corners=4, zerophase=False
        )

        starttime = trace.stats.starttime
        endtime = trace.stats.endtime
        st_lf.merge(fill_value=np.nan)
        st_lf.trim(starttime, endtime, pad=True, fill_value=np.nan)
        st_hf.merge(fill_value=np.nan)
        st_hf.trim(starttime, endtime, pad=True, fill_value=np.nan)
        tr_lf = st_lf[0]
        tr_hf = st_hf[0]

        # DSAR calculation
        feature_data = []
        feature_idx = []

        # loop through data in interval blocks
        for t0, t1 in trace_window_times(tr_lf, self.interval, min_len=0.8):
            tr_lf_int = tr_lf.slice(t0, t1)
            tr_hf_int = tr_hf.slice(t0, t1)

            absolute_lf = np.absolute(tr_lf_int.data)
            absolute_hf = np.absolute(tr_hf_int.data)

            # calculate ratio
            if np.isnan(absolute_lf).any() or np.isnan(absolute_hf).any():
                median = np.nan
            else:
                m = absolute_hf > 0
                ratio = absolute_lf[m] / absolute_hf[m]
                median = np.median(ratio)

            feature_data.append(median)
            feature_idx.append(t0.strftime("%Y-%m-%dT%H:%M:%S"))

        xda = xr.DataArray(
            feature_data, coords=[pd.DatetimeIndex(feature_idx)], dims="datetime"
        )
        xdf = xr.Dataset({"dsar": xda})
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
