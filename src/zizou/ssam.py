import logging
import math as M

import numpy as np
import obspy
import pandas as pd
import xarray as xr
import yaml

from zizou import FeatureBaseClass
from zizou.util import window_array

logger = logging.getLogger(__name__)


class SSAM(FeatureBaseClass):
    """
    SSAM is basically the same as computing a spectrogram. Data are split into
    interval length segments and the spectrum of each section is
    computed. Before computing the spectrum, the mean is removed and
    a hanning window is applied to each segment. The percentage of
    overlap of each segment can be specified with per_lap and Welch-type
    smoothing can be applied.

    :param interval: Segment length in seconds.
    :type interval: int
    :param per_lap: If the value is less than 1 it is treated as the percentage
                    of segment overlap; else it is the step size in sample points
    :type per_lap: float or int
    :param scale_by_freq: Divide by the sampling frequency so that density
                          function has units of dB/Hz and can be integrated
                          by the frequency values.
    :type scale_by_freq: boolean
    :param smooth: The number of windows over which to
                   average the spectrogram.
    :type smooth: int
    :param frequencies: Frequencies at which to return the spectrogram. This
                        uses linear interpolation to compute the spectrogram
                        at the given frequencies from the original spectrogram.
    :type frequencies: :class:`numpy.ndarray`
    :param timestamp: Can be either 'center' or 'start'. If 'center', the
                      timestamp of each spectrum is the center of the windows
                      from which the spectrum was computed. If 'start' it is
                      the timestamp of the first sample of the first window.
    :type timestamp: str
    :param resample_int: Interval to upsample the dataset using linear
                         interpolation over the first interval and then
                         downsample using the mean over the second interval.
    :type resample_int: (str, str)
    :param configfile: Configuration as an .ini file or a dictionary.
    :type configfile: str or dict

    >>> from zizou.ssam import SSAM, test_signal
    >>> import numpy as np
    >>> tr = test_signal(gaps=True)
    >>> s = SSAM(interval=60, per_lap=.9, smooth=10,
                 frequencies=np.arange(0, 25.1, .1))

    """

    features = ["ssam", "filterbank", "sonogram"]

    def __init__(
        self,
        interval=10,
        per_lap=0,
        scale_by_freq=True,
        smooth=None,
        frequencies=np.linspace(0, 25, 251),
        fbfilters=16,
        nhob=8,
        timestamp="center",
        resample_int=None,
        configfile=None,
    ):
        super(SSAM, self).__init__()
        self.interval = interval
        self.per_lap = per_lap
        self.scale_by_freq = scale_by_freq
        self.smooth = smooth
        self.frequencies = frequencies
        self.timestamp = timestamp
        self.resample_int = resample_int
        self.fbfilters = fbfilters
        self.nhob = nhob

        self.feature = None

        if configfile is not None:
            try:
                with open(configfile, "r") as fh:
                    c = yaml.safe_load(fh)
            except OSError:
                c = yaml.safe_load(configfile)
            cs = c.get("ssam")
            if cs is not None:
                self.interval = cs.get("interval", interval)
                self.per_lap = cs.get("per_lap", per_lap)
                self.scale_by_freq = cs.get("scale_by_freq", scale_by_freq)
                self.smooth = cs.get("smooth", smooth)
                self.timestamp = cs.get("timestamp", timestamp)
                self.nhob = cs.get("nhob", nhob)
                freqs = cs.get("frequencies", frequencies)
                if freqs is not None:
                    self.frequencies = np.arange(
                        freqs["start"], freqs["end"], freqs["step"]
                    )

                self.fbfilters = cs.get("fbfilters", fbfilters)
                ri = cs.get("resample_int", resample_int)
                if ri is not None:
                    self.resample_int = (ri["upsample"], ri["downsample"])

    @staticmethod
    def _nearest_pow_2(x):
        """
        Find power of two nearest to x

        >>> _nearest_pow_2(3)
        2.0
        >>> _nearest_pow_2(15)
        16.0

        :param x: Number
        :type x: float
        :return: Nearest power of 2 to x
        :rtype: Int
        """
        a = M.pow(2, M.ceil(np.log2(x)))
        b = M.pow(2, M.floor(np.log2(x)))
        if abs(a - x) < abs(b - x):
            return a
        else:
            return b

    def _average(self, spec, t):
        """
        Average spectrogram over nwin windows to produce a smoother
        spectrogram. If used with overlapping windows it is known
        as the Welch algorithm

        :param spec: Input spectrogram
        :type spec: :class:`numpy.ndarray`
        :param t: Time axis of spectrogram.
        :type t: :class:`numpy.ndarray`
        :param nwin: Number of windows over which to average
        :type nwin: int
        :return: smoothed_spec, new-time
        :rtype: tuple of :class:`~numpy.ndarray`
        """
        start = 0
        end = 0
        ntimes = spec.shape[1]
        av_mat = []
        nt = []
        while end < ntimes:
            av_arr = np.zeros(spec.shape[1])
            end = start + self.smooth
            if end > ntimes:
                end = ntimes
            av_arr[start:end] = 1.0
            av_mat.append(av_arr)
            if self.timestamp == "center":
                nt.append(t[start : end : self.smooth - 1].mean())
            elif self.timestamp == "start":
                nt.append(t[start])
            start = end

        av_mat = np.array(av_mat).T
        nt = np.array(nt)
        # find all columns with NaNs
        colidx = np.unique(np.where(np.isnan(spec))[1])
        # set corresponding rows in averaging matrix to NaN
        av_mat[colidx, :] = 0
        factor = np.sum(av_mat, axis=0)
        av_mat = np.divide(av_mat, factor, where=av_mat > 0)
        mspec = np.ma.masked_invalid(spec)
        mav_mat = np.ma.masked_invalid(av_mat)
        nspec = np.ma.dot(mspec, mav_mat)
        return nspec.data, nt

    @staticmethod
    def _resample(xdf, resample_int=("1min", "10min")):
        """
        Resample spectrograms to the desired sampling interval using xarray
        functionality.

        :param xdf: An xarray dataset as returned by :class:`zizou.ssam.SSAM`
        :type xdf: :class:`xarray.Dataset`
        :param resample_int: Intervals to upsample the dataset using linear
                             interpolation over the first interval and then
                             downsample using the mean over the second
                             interval.
        :type resample_int: (str, str)
        :return: Stacked dataset.
        :rtype: :class:`~xarray.Dataset`
        """
        upsample_int, downsample_int = resample_int
        attrs = xdf.attrs
        if upsample_int is not None:
            xdf = xdf.resample(datetime=upsample_int).interpolate("linear")
        if downsample_int is not None:
            xdf = xdf.resample(datetime=downsample_int).mean()
        # Resample transposes the dimensions so we need to revert this
        xdf["ssam"] = xdf.ssam.transpose("frequency", "datetime")
        xdf["filterbank"] = xdf.filterbank.transpose("fbfrequency", "datetime")
        xdf["sonogram"] = xdf.sonogram.transpose("sonofrequency", "datetime")
        xdf.attrs = attrs
        return xdf

    @staticmethod
    def filterbank(nfilters, nfft, fs):
        """
        Computes the integral in an array of bandpass filters that
        separates the input signal into multiple components.
        Each component consists of a single frequency sub-band
        of the original signal.

        :param nfilters: Number of filters in the filter bank.
        :type nfitlers: int
        :param nfft: Number of samples
        :type nfft: int
        :param fs: Sampling frquency
        :type fs: float

        """
        fl = fs / nfft
        fh = fs / 2
        freqs = np.logspace(np.log10(fl), np.log10(fh), nfilters + 2)
        bins = np.floor((nfft + 1) * freqs / fs)
        fbank = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
        for m in range(1, nfilters + 1):
            f_m_minus = int(bins[m - 1])  # left
            f_m = int(bins[m])  # center
            f_m_plus = int(bins[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

        return freqs[1:-1], fbank

    @staticmethod
    def sonogram(data, fs, fc1, nofb=8):
        """
        Computes the sonogram of the given data which can be windowed or not.
        The sonogram is determined by the power in half octave bands of the given
        data.

        If data are windowed the analytic signal and the envelope of each window
        is returned.

        :param data: Data to make envelope of.
        :type data: :class:`~numpy.ndarray`
        :param fs: Sampling frequency in Hz.
        :param fc1: Center frequency of lowest half octave band.
        :param nofb: Number of half octave bands.
        :param no_win: Number of data windows.
        :return: Central frequencies, Half octave bands.
        """
        fc = float(fc1) * 1.5 ** np.arange(nofb)
        fmin = fc / np.sqrt(5.0 / 3.0)
        fmax = fc * np.sqrt(5.0 / 3.0)
        z_tot = np.sum(np.abs(data) ** 2, axis=0)
        nfft = data.shape[0]
        start = np.around(fmin * nfft / fs, 0).astype(np.int_) - 1
        end = np.around(fmax * nfft / fs, 0).astype(np.int_)
        z = np.zeros([nofb, data.shape[1]])
        for i in range(nofb):
            z[i, :] = np.sum(np.abs(data[start[i] : end[i], :]) ** 2, axis=0)
        hob = np.log(z / z_tot[np.newaxis, :])
        return fc, hob

    def compute(self, trace):
        """
        Compute SSAM for the given trace.

        :param trace: The seismic waveform data.
        :type trace: :class:`obspy.Trace`
        :return: spectrogram, frequencies, times
        :rtype: tuple
        """
        if len(trace) < 1:
            msg = "Trace is empty."
            raise ValueError(msg)

        logger.info(
            "Computing spectrograms for {} between {} and {}.".format(
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
        x = tr.data
        npts = tr.stats.npts
        Fs = tr.stats.sampling_rate
        interval = min(self.interval * Fs, npts)

        NFFT = int(self._nearest_pow_2(interval))
        if NFFT > npts:
            NFFT = int(self._nearest_pow_2(npts / 8.0))

        if self.per_lap < 1.0:
            noverlap = int(NFFT * float(self.per_lap))
        else:
            noverlap = NFFT - self.per_lap

        scaling_factor = 2.0
        result, windowVals = window_array(x, NFFT, noverlap)
        result = np.fft.rfft(result, n=NFFT, axis=0)
        freqs = np.fft.rfftfreq(NFFT, 1 / Fs)

        result = np.conj(result) * result
        # Also include scaling factors for one-sided densities and dividing by
        # the sampling frequency, if desired. Scale everything, except the DC
        # component and the NFFT/2 component:
        slc = slice(1, -1, None)

        result[slc] *= scaling_factor

        # MATLAB divides by the sampling frequency so that density function
        # has units of dB/Hz and can be integrated by the plotted frequency
        # values. Perform the same scaling here.
        if self.scale_by_freq:
            result /= Fs
            # Scale the spectrum by the norm of the window to compensate for
            # windowing loss; see Bendat & Piersol Sec 11.5.2.
            result /= (np.abs(windowVals) ** 2).sum()
        else:
            # In this case, preserve power in the segment, not amplitude
            result /= np.abs(windowVals).sum() ** 2

        result = result.real
        steplength = NFFT - noverlap
        steps = np.arange(result.shape[1]) * steplength
        if self.timestamp == "center":
            t = (steps + NFFT / 2) / Fs
        elif self.timestamp == "start":
            t = steps / Fs
        # smooth the spectrogram using the Welch algorithm
        if self.smooth is not None:
            result, t = self._average(result, t)

        features = {}
        dates = [(trace.stats.starttime + _t).datetime for _t in t]

        fb_freqs, fb = self.filterbank(self.fbfilters, NFFT, Fs)
        result_fb = np.dot(fb, result)
        filterbank = xr.DataArray(
            result_fb,
            coords=[fb_freqs, pd.to_datetime(dates)],
            dims=["fbfrequency", "datetime"],
        )
        features["filterbank"] = filterbank

        sono_freq, sono_amp = self.sonogram(result, Fs, 0.68, self.nhob)
        sono = xr.DataArray(
            sono_amp,
            coords=[sono_freq, pd.to_datetime(dates)],
            dims=["sonofrequency", "datetime"],
        )
        features["sonogram"] = sono

        # Linear interpolation onto a new set of frequencies
        def myinterp(a):
            return np.interp(self.frequencies, freqs, a)

        result = np.apply_along_axis(myinterp, 0, result)
        freqs = self.frequencies

        ssam = xr.DataArray(
            result,
            coords=[freqs, pd.to_datetime(dates)],
            dims=["frequency", "datetime"],
        )
        features["ssam"] = ssam
        self.feature = xr.Dataset(features)
        self.feature.attrs["starttime"] = trace.stats.starttime.isoformat()
        self.feature.attrs["endtime"] = trace.stats.endtime.isoformat()
        self.feature.attrs["station"] = trace.stats.station

        if self.resample_int is not None:
            self.feature = self._resample(self.feature, self.resample_int)
        return self.feature
