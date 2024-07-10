#!/usr/bin/env python
"""
Complex trace Analysis

"""
from configparser import ConfigParser

from scipy import signal
import scipy.integrate
from obspy.signal import util

import numpy as np
from obspy.core import UTCDateTime
import pandas as pd

from zizou import FeatureBaseClass


class Inst_Band(FeatureBaseClass):
    """
    Computes the instantaneous bandwidth of the given data which can be
    windowed or not. The instantaneous bandwidth is determined by the time
    derivative of the envelope normalized by the envelope of the input
    data.

    :param fil_coef: Filter coefficients for computing time derivative.
    :type fil_coef: :class:`~numpy.ndarray
    :return: **DataFrame** - Time,Instantaneous bandwidth of input data
    """

    def __init__(self, interval=600., fil_coef=(None, None, None),
                 configfile=None):
        super(Inst_Band, self).__init__()
        self.interval = interval
        self.fil_coef = fil_coef
        if configfile is not None:
            c = ConfigParser()
            c.read(configfile)
            self.interval = float(c['DEFAULT']['interval'])
            self.fil_coef = c['Insta_Band']['fil_coef']
            fc1, fc2, fc3 = c['Insta_Band']['fil_coef'].split(',')
            self.fil_coef = (float(fc1), float(fc2), float(fc3))

    def compute(self, trace):
        """
        :param trace: The seismic waveform data
        :type trace: :class:`obspy.Trace`
        """
        self.trace = trace
        self.data = self.trace.data
        self.fs = self.trace.stats.sampling_rate

        # initialise dataframe
        instan_band = pd.DataFrame()

        if len(self.trace) < 1:
            msg = "Trace is empty."
            raise ValueError(msg)

        ti = self.trace.stats.starttime
        index = 0
        # loop through data in interval blocks
        while ti < self.trace.stats.endtime:
            trint = self.trace.slice(ti, ti + self.interval)
            duration = trint.stats.npts * trint.stats.delta
            if duration >= 500:
                if duration < self.interval:
                    trint = self.trace.slice(
                        self.trace.stats.endtime - self.interval,
                        self.trace.stats.endtime
                    )

                # Calculate the envelope for interval blocks

                nfft = util.next_pow_2(trint.data.shape[-1])
                a_cpx = np.zeros((trint.data.shape), dtype=np.complex64)
                a_abs = np.zeros((trint.data.shape), dtype=np.float64)

                a_cpx = signal.hilbert(trint.data, nfft)
                a_abs = abs(signal.hilbert(trint.data, nfft))

                x = (a_cpx, a_abs)

                # Calculate Instantaneous bandwidth of input data
                row = x[1]

                # faster alternative to calculate A_win_add
                a_win_add = np.hstack(
                    ([row[0]] * (np.size(self.fil_coef) // 2), row,
                     [row[np.size(row) - 1]] * (np.size(self.fil_coef) // 2)))
                t = signal.lfilter(self.fil_coef, 1, a_win_add)
                # correct start and end values
                t = t[np.size(self.fil_coef) - 1:np.size(t)]
                sigma = abs((t * self.fs) / (x[1] * 2 * np.pi))

                # Create a DataFrame for output
                inst_b = pd.DataFrame(sigma)
                inst_b.drop(inst_b.tail(-trint.stats.npts).index, inplace=True)

                data1 = inst_b.mean()
                tstr = pd.to_datetime(
                    UTCDateTime.strftime(ti, '%Y-%m-%dT%H:%M:%S')
                )
                df = pd.DataFrame(data1.values, index=[tstr])
                instan_band = instan_band.append(df)
                index += 1
            ti += self.interval
        instan_band.columns = ['inst_band']

        self.feature = instan_band
        self.trace = trace
        return instan_band


class Inst_Freq(FeatureBaseClass):
    """
    Computes the instantaneous frequency of the given data which can be
    windowed or not. The instantaneous frequency is determined by the time
    derivative of the analytic signal of the input data.

    :param fil_coef: Filter coefficients for computing time derivative.
    :type fil_coef: :class:`~numpy.ndarray
    :return: **DataFrame** - Time, Instantaneous frequency of input data
    .
    """

    def __init__(self, interval=600., fil_coef=(None, None, None),
                 configfile=None):
        super(Inst_Freq, self).__init__()
        self.interval = interval
        self.fil_coef = fil_coef
        if configfile is not None:
            c = ConfigParser()
            c.read(configfile)
            self.interval = float(c['DEFAULT']['interval'])
            self.fil_coef = c['Insta_Freq']['fil_coef']
            fc1, fc2, fc3 = c['Insta_Freq']['fil_coef'].split(',')
            self.fil_coef = (float(fc1), float(fc2), float(fc3))

    def compute(self, trace):
        """
        :param trace: The seismic waveform data
        :type trace: :class:`obspy.Trace`
        """
        self.trace = trace
        self.data = self.trace.data
        self.fs = self.trace.stats.sampling_rate

        # initialise dataframe
        instan_freq = pd.DataFrame()

        if len(self.trace) < 1:
            msg = "Trace is empty."
            raise ValueError(msg)

        ti = self.trace.stats.starttime
        index = 0
        # loop through data in interval blocks
        while ti < self.trace.stats.endtime:
            trint = self.trace.slice(ti, ti + self.interval)
            duration = trint.stats.npts * trint.stats.delta
            if duration >= 500:
                if duration < self.interval:
                    trint = self.trace.slice(
                        self.trace.stats.endtime - self.interval,
                        self.trace.stats.endtime
                    )

                # Calculate the envelope for interval blocks

                nfft = util.next_pow_2(trint.data.shape[-1])
                a_cpx = np.zeros((trint.data.shape), dtype=np.complex64)
                a_abs = np.zeros((trint.data.shape), dtype=np.float64)

                a_cpx = signal.hilbert(trint.data, nfft)
                a_abs = abs(signal.hilbert(trint.data, nfft))

                x = (a_cpx, a_abs)

                # Calculate Instantaneous bandwidth of input data
                omega = np.zeros(np.size(x[0]), dtype=np.float64)
                f = np.real(x[0])
                h = np.imag(x[0])
                # faster alternative to calculate f_add
                f_add = np.hstack(
                    ([f[0]] * (np.size(self.fil_coef) // 2), f,
                     [f[np.size(f) - 1]] * (np.size(self.fil_coef) // 2)))
                fd = signal.lfilter(self.fil_coef, 1, f_add)
                # correct start and end values of time derivative
                fd = fd[np.size(self.fil_coef) - 1:np.size(fd)]
                # faster alternative to calculate h_add
                h_add = np.hstack(
                    ([h[0]] * (np.size(self.fil_coef) // 2), h,
                     [h[np.size(h) - 1]] * (np.size(self.fil_coef) // 2)))
                hd = signal.lfilter(self.fil_coef, 1, h_add)
                # correct start and end values of time derivative
                hd = hd[np.size(self.fil_coef) - 1:np.size(hd)]
                omega = abs(((f * hd - fd * h) / (f * f + h * h)) *
                            self.fs / 2 / np.pi)

                # Create a DataFrame for output

                inst_f = pd.DataFrame(omega)
                inst_f.drop(inst_f.tail(-trint.stats.npts).index, inplace=True)

                data1 = inst_f.mean()
                tstr = pd.to_datetime(
                    UTCDateTime.strftime(ti, '%Y-%m-%dT%H:%M:%S')
                )
                df = pd.DataFrame(data1.values, index=[tstr])
                instan_freq = instan_freq.append(df)
                index += 1
            ti += self.interval
        instan_freq.columns = ['inst_freq']

        self.feature = instan_freq
        self.trace = trace
        return instan_freq

  
class Norm_Env(FeatureBaseClass):
    """
    Computes the normalized envelope of the given data which can be
    windowed or not. In order to obtain a normalized measure of the signal
    envelope the instantaneous bandwidth of the smoothed envelope is
    normalized by the Nyquist frequency and is integrated afterwards.

    The time derivative of the normalized envelope is returned if input
    data are windowed only.

      :param fil_coef: Filter coefficients for computing time derivative.
      :type fil_coef: :class:`~numpy.ndarray
      :return: **DataFrame** - Time , Normalized envelope of input data
    """
    def __init__(self, window=10, interval=600., fil_coef=(None, None, None),
                 configfile=None):
        super(Norm_Env, self).__init__()
        self.interval = interval
        self.fil_coef = fil_coef
        self.window = window
        if configfile is not None:
            c = ConfigParser()
            c.read(configfile)
            self.interval = float(c['DEFAULT']['interval'])
            self.window = c['Norm_Env']['window']
            self.fil_coef = c['Norm_Env']['fil_coef']
            fc1, fc2, fc3 = c['Norm_Env']['fil_coef'].split(',')
            self.fil_coef = (float(fc1), float(fc2), float(fc3))

    def compute(self, trace):
        """
        :param trace: The seismic waveform data
        :type trace: :class:`obspy.Trace`
        """
        self.trace = trace
        self.data = self.trace.data
        self.fs = self.trace.stats.sampling_rate

        # initialise dataframe
        norm_env = pd.DataFrame()

        if len(self.trace) < 1:
            msg = "Trace is empty."
            raise ValueError(msg)

        ti = self.trace.stats.starttime
        index = 0
        # loop through data in interval blocks
        while ti < self.trace.stats.endtime:
            trint = self.trace.slice(ti, ti + self.interval)
            duration = trint.stats.npts * trint.stats.delta
            if duration >= 500:
                if duration < self.interval:
                    trint = self.trace.slice(
                        self.trace.stats.endtime - self.interval,
                        self.trace.stats.endtime
                    )

                # Calculate the envelope for interval blocks

                nfft = util.next_pow_2(trint.data.shape[-1])
                a_cpx = np.zeros((trint.data.shape), dtype=np.complex64)
                a_abs = np.zeros((trint.data.shape), dtype=np.float64)

                a_cpx = signal.hilbert(trint.data, nfft)
                a_abs = abs(signal.hilbert(trint.data, nfft))

                x = (a_cpx, a_abs)

                smoothie = int(self.window)

                # Calculate the normalize envelope
                a_win_smooth = util.smooth(x[1], smoothie)
                # Differentiation of original signal, dA/dt
                # Better, because faster, calculation of A_win_add
                a_win_add = np.hstack(
                    ([a_win_smooth[0]] * (np.size(self.fil_coef) // 2),
                     a_win_smooth, [a_win_smooth[np.size(a_win_smooth) - 1]] *
                     (np.size(self.fil_coef) // 2)))
                t = signal.lfilter(self.fil_coef, 1, a_win_add)
                # correct start and end values of time derivative
                t = t[np.size(self.fil_coef) - 1:np.size(t)]
                a_win_smooth[a_win_smooth < 1] = 1
                t_ = t / (2. * np.pi * (a_win_smooth) * (self.fs / 2.0))
                # Integral within window
                t_ = scipy.integrate.cumtrapz(t_, dx=(1.0 / self.fs))
                t_ = np.concatenate((t_[0:1], t_))
                anorm = ((np.exp(np.mean(t_))) - 1) * 100

                # Create a DataFrame for output

                data1 = {'norm_env': anorm}
                tstr = pd.to_datetime(
                    UTCDateTime.strftime(ti, '%Y-%m-%dT%H:%M:%S')
                )
                df = pd.DataFrame(data1, index=[tstr])
                norm_env = norm_env.append(df)
                index += 1
            ti += self.interval
        norm_env.columns = ['norm_env']

        self.feature = norm_env
        self.trace = trace
        return norm_env


class Centroid(FeatureBaseClass):
    """
    Computes the centroid time of the given data which can be windowed or
    not. The centroid time is determined as the time in the processed
    window where 50 per cent of the area below the envelope is reached.

    The time derivative of the centroid time is returned if input data are
    windowed only.

     :param fil_coef: Filter coefficients for computing time derivative.
     :type fil_coef: :class:`~numpy.ndarray

     :return: **DataFrame** - Time , Centroid time input data
    """
    def __init__(self, interval=600., configfile=None):
        super(Centroid, self).__init__()
        self.interval = interval
        if configfile is not None:
            c = ConfigParser()
            c.read(configfile)
            self.interval = float(c['DEFAULT']['interval'])

    def compute(self, trace):
        """
        :param trace: The seismic waveform data
        :type trace: :class:`obspy.Trace`
        """
        self.trace = trace
        self.data = self.trace.data
        self.fs = self.trace.stats.sampling_rate

        # initialise dataframe
        centroid1 = pd.DataFrame()

        if len(self.trace) < 1:
            msg = "Trace is empty."
            raise ValueError(msg)

        ti = self.trace.stats.starttime
        index = 0
        # loop through data in interval blocks
        while ti < self.trace.stats.endtime:
            trint = self.trace.slice(ti, ti + self.interval)
            duration = trint.stats.npts * trint.stats.delta
            if duration >= 500:
                if duration < self.interval:
                    trint = self.trace.slice(
                        self.trace.stats.endtime - self.interval,
                        self.trace.stats.endtime
                    )

                # Calculate the envelope
                nfft = util.next_pow_2(trint.data.shape[-1])
                a_cpx = signal.hilbert(trint.data, nfft)
                a_abs = abs(signal.hilbert(trint.data, nfft))

                x = (a_cpx, a_abs)

                # Calculate the centroid
                centro = np.zeros(1, dtype=np.float64)
                # Integral within window
                half = 0.5 * sum(x[1])
                # Estimate energy centroid
                for k in range(2, np.size(x[1])):
                    t = sum(x[1][0:k])
                    if (t >= half):
                        frac = (half - (t - sum(x[1][0:k - 1]))) / \
                               (t - (t - sum(x[1][0:k - 1])))
                        centro = \
                            (float(k) + float(frac)) / float(np.size(x[1]))

                # Create a DataFrame for output1
                data1 = {'centroid': centro}
                tstr = pd.to_datetime(
                    UTCDateTime.strftime(ti, '%Y-%m-%dT%H:%M:%S')
                )
                df = pd.DataFrame(data1, index=[tstr])
                centroid1 = centroid1.append(df)
                index += 1
            ti += self.interval
        centroid1.columns = ['centroid']
        self.feature = centroid1
        self.trace = trace
        return centroid1


class WindowCentroid(FeatureBaseClass):
    """
    Computes the centroid of a given time window,
    defined between 0 and 1.
    Deltas at t_start, t_end and t_midpoint give
    values of 0, 1 and 0.5 respectively.
    A flat signal has a centroid = 0.57.

    :param interval: Length of the time interval
                     to compute feature.
    :type interval: int, optional
    """

    def __init__(self, interval=600., configfile=None):
        super(WindowCentroid, self).__init__()
        self.interval = interval
        if configfile is not None:
            c = ConfigParser()
            c.read(configfile)
            self.interval = float(c['DEFAULT']['interval'])

    def compute(self, trace):
        if len(trace) < 1:
            msg = "Trace is empty."
            raise ValueError(msg)
        tr = trace.copy()
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime
        # Filter settings
        f1, f2 = 0.1, 20
        # handle gaps (i.e. NaNs)
        mdata = np.ma.masked_invalid(tr.data)
        tr.data = mdata
        st = tr.split()
        # Filter
        st.detrend('linear')
        st.detrend('constant')
        st.filter('bandpass', freqmin=f1, freqmax=f2, corners=4,
                  zerophase=False)
        st.merge(fill_value=np.nan)
        st.trim(starttime, endtime, pad=True, fill_value=np.nan)
        tr = st[0]
        tr.taper(0.025)
        tr.differentiate()

        # initialise dataframe
        window_centroid = pd.DataFrame()

        t = tr.stats.starttime
        index = 0
        # loop through data in interval blocks
        while t < tr.stats.endtime:
            trint = tr.slice(t, t + self.interval)
            duration = trint.stats.npts * trint.stats.delta
            if duration >= 0.8 * self.interval:
                if duration < self.interval:
                    trint = tr.slice(trace.stats.endtime - self.interval,
                                     trace.stats.endtime)

                m0 = 2 * scipy.integrate.simps(trint.times() ** 0 * np.abs(trint.data))
                m2 = 2 * scipy.integrate.simps(trint.times() ** 2 * np.abs(trint.data))
                centroid = np.sqrt(m2 / m0) / trint.times()[-1]

                data = {'window_centroid': float(format(centroid, '.4f'))}
                ts = UTCDateTime.strftime(t, '%Y-%m-%dT%H:%M:%S')
                tstr = pd.to_datetime(ts)
                window_centroid = \
                    window_centroid.append(
                        pd.DataFrame(data, index=[tstr])
                    )
                index += 1
            t += self.interval
        self.feature = window_centroid
        self.trace = tr
        return window_centroid
