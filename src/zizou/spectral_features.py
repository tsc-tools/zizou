import logging

import numpy as np
import obspy.signal.freqattributes
import obspy.signal.hoctavbands
import obspy.signal.util
import pandas as pd
import scipy.fftpack
import scipy.integrate
import yaml

from zizou import FeatureBaseClass
from zizou.util import apply_freq_filter, trace_window_data

logger = logging.getLogger(__name__)


class SpectralFeatures(FeatureBaseClass):
    features = ["central_freq", "bandwidth", "predom_freq"]

    def __init__(
        self,
        interval=600.0,
        filtertype="highpass",
        filterfreq=(0.5, None),
        ncep=8,
        configfile=None,
    ):
        super(SpectralFeatures, self).__init__()
        self.interval = float(interval)
        self.filtertype = filtertype
        self.filterfreq = filterfreq
        self.ncep = int(ncep)
        if configfile is not None:
            try:
                with open(configfile, "r") as fh:
                    c = yaml.safe_load(fh)
            except OSError:
                c = yaml.safe_load(configfile)
            self.interval = c["default"].get("interval", interval)
            cs = c.get("spectral_features")
            if cs is not None:
                self.filtertype = cs.get("filtertype", filtertype)
                freqs = cs.get("filterfreq")
                if freqs is not None:
                    self.filterfreq = (freqs["low"], freqs["high"])
                self.ncep = cs.get("num_cepstral", ncep)

    def compute(self, trace):
        """
        :param trace: The seismic waveform data
        :type trace: :class:`obspy.Trace`
        :return:
        DataFrame containing
        - central frequency (acceleration) in Hz
        - bandwidth parameter (acceleration)
        - predominant frequency (acceleration) in Hz
        - eight cepstral coefficients
        """
        if len(trace) < 1:
            msg = "Trace is empty."
            raise ValueError(msg)

        logger.info(
            "Computing spectral features for {} between {} and {}.".format(
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

        fs = trace.stats.sampling_rate

        keys = ["central_freq", "bandwidth", "predom_freq"]
        # keys += [f'cepstral_coef_{ind}' for ind in np.arange(self.ncep)]

        # initialise dataframe
        feature_data = []
        feature_idx = []

        # loop through data in interval blocks
        for tr_int in trace_window_data(trace, self.interval, min_len=0.8):
            if np.any(np.isnan(tr_int.data)):
                vals = np.full(len(keys), np.nan)
            else:
                tr_int.detrend("constant")
                tr_int.detrend("linear")
                apply_freq_filter(tr_int, self.filtertype, self.filterfreq)
                tr_int_diff = tr_int.differentiate()

                # Calculate psdf
                nfft = obspy.signal.util.next_pow_2(tr_int_diff.stats.npts)
                frequency = np.linspace(0, fs, nfft + 1)
                frequency = frequency[0 : nfft // 2]
                data_f = scipy.fftpack.fft(tr_int_diff.data, nfft)
                data_psd = np.abs(data_f[0 : nfft // 2]) ** 2

                # Interpolate onto logspace frequency vector
                f_ls = np.logspace(-2, 2, 401)  # These constants should be accessible
                data_psd_raw = np.interp(f_ls, frequency, data_psd)
                data_psd_ls = obspy.signal.util.smooth(data_psd_raw, 3)

                central_freq = self.centralFrequency(data_psd_ls, f_ls)
                bandwidth = self.bandwidth(data_psd_ls, f_ls)
                predom_freq = self.predominantFrequency(data_psd_ls.data, f_ls)
                # cepstral_coef = self.cepstralCoefficients(
                #     tr_int.data, fs, self.ncep
                # )
                vals = [central_freq, bandwidth, predom_freq]
                # vals = np.concatenate((vals, cepstral_coef, hob), axis=None)

            feature_data.append(vals)
            feature_idx.append(tr_int.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S"))

        xdf = pd.DataFrame(
            data=feature_data,
            index=pd.DatetimeIndex(feature_idx, name="datetime"),
            columns=keys,
            dtype=float,
        ).to_xarray()
        xdf.attrs["starttime"] = trace.stats.starttime.isoformat()
        xdf.attrs["endtime"] = trace.stats.endtime.isoformat()
        xdf.attrs["station"] = trace.stats.station

        self.feature = xdf
        self.trace = trace
        return self.feature

    @staticmethod
    def predominantFrequency(psd, freq):
        """
        The peak frequency of the acceleration power spectral
        density function, in Hz.
        """
        predom_freq = freq[np.argmax(psd)]
        return predom_freq

    @staticmethod
    def centralFrequency(psd, freq):
        """
        The 'central' frequency of the acceleration power
        spectral density function (Hz). Defined as the
        square root of the ratio between the second and
        zero-th spectral moments. The spectral moments
        are calculated with log-spaced frequencies.
        """
        m0 = 2 * scipy.integrate.simpson(freq**0 * psd)
        m2 = 2 * scipy.integrate.simpson(freq**2 * psd)
        return np.sqrt(m2 / m0)

    @staticmethod
    def bandwidth(psd, freq):
        """
        The Vanmarcke (1970) bandwidth parameter, q where
        0 <= q <= 1. Calculated using the spectral moments
        of the acceleration power spectral density function.
        0 is a Dirac delta PSDF (sine wave), 1 is a flat PSDF
        (white signal).
        """
        m0 = 2 * scipy.integrate.simpson(freq**0 * psd)
        m1 = 2 * scipy.integrate.simpson(freq**1 * psd)
        m2 = 2 * scipy.integrate.simpson(freq**2 * psd)
        return np.sqrt(1 - m1**2 / m2 / m0)

    @staticmethod
    def cepstralCoefficients(data, fs, ncep=8, nfilters=10, window="Hamming"):
        cep = obspy.signal.freqattributes.log_cepstrum(
            data=np.expand_dims(data, axis=0),
            fs=fs,
            nc=ncep,
            p=nfilters,
            n=None,
            w=window,
        )
        return cep
