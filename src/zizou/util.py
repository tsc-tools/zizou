"""
Functionality common to several modules.
"""
from datetime import datetime
import logging
import math as M
import os

import h5netcdf
from cftime import num2date, date2num, date2index
import numpy as np
from obspy import UTCDateTime, Trace
import pandas as pd
from scipy.signal import chirp, sweep_poly
import xarray as xr
from warnings import filterwarnings


logger = logging.getLogger(__name__)


def stride_windows(x, n, noverlap):
    """
    Get all windows of x with length n as a single array,
    using strides to avoid data duplication.

    :param x: 1D array or sequence containing the data.
    :type x: :class:`~numpy.ndarray`
    :param n: The number of data points in each window.
    :type n: int
    :param noverlap: The overlap between adjacent windows.
                     Default is 0 (no overlap)
    :type noverlap: int
    :return: Array with the windowed data in columns.
    :rtype: :class:`~numpy.ndarray`
    """

    if noverlap >= n:
        raise ValueError('noverlap must be less than n')
    if n < 1:
        raise ValueError('n cannot be less than 1')

    if n == 1 and noverlap == 0:
        return x[np.newaxis]
    if n > x.size:
        raise ValueError('n cannot be greater than the length of x')

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. noverlap or n. See #3845.
    noverlap = int(noverlap)
    n = int(n)

    step = n - noverlap
    shape = (n, (x.shape[-1] - noverlap) // step)
    strides = (x.strides[0], step * x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides,
                                           writeable=False)


def demean(x, axis=None):
    """
    Return x minus the mean(x).

    :param x: array or sequence containing the data can
              have any dimensionality
    :type x: :class:`~numpy.ndarray`
    :param axis: The axis along which to take the mean. See numpy.mean
                 for a description of this argument.
    :type axis: int
    :return: input minus the arithmetic mean along the specified axis.
    :rtype: :class:`~numpy.ndarray`
    """
    x = np.asarray(x)

    if axis is not None and axis + 1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    return x - np.nanmean(x, axis=axis, keepdims=True)


def apply_hanning(x, return_window=None):
    """
    Apply a hanning window to the given 1D or 2D array along
    the given axis.

    :param x: 1-D or 2-D array or sequence containing the data
    :type x: :class:`numpy.ndarray`
    :param return_window: If true, also return the 1D values of the window
                          that was applied.
    :type return_window: bool
    :return: **wtr[, wvals]** input multiplied with the hanning
             window function, hanning window function
    """
    x = np.asarray(x)

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError('only 1D or 2D arrays can be used')

    xshapetarg = x.shape[0]
    windowVals = np.hanning(xshapetarg) * np.ones(xshapetarg, dtype=x.dtype)

    if x.ndim == 1:
        if return_window:
            return windowVals * x, windowVals
        else:
            return windowVals * x

    windowVals = windowVals[:, np.newaxis]

    if return_window:
        return windowVals * x, windowVals
    else:
        return windowVals * x


def window_array(x, nwin, noverlap,
                 remove_mean=True, taper=True,
                 padval=0, return_window=True):
    """
    Take a 1-D numpy array and devide it into (overlapping)
    windows.

    :param x: 1D array containing the data.
    :type x: :class:`~numpy.ndarray`
    :param n: The number of data points in each window. If this
              is not a power of 2, the length of each window will be
              the nearest power of 2.
    :type n: int
    :param noverlap: The number of overlapping points between
                     adjacent windows.
    :type noverlap: int
    :param remove_mean: Remove the individual mean from each
                        window.
    :type remove_mean: bool
    :param taper: Apply a hanning taper to each window.
    :type taper: bool
    :param return_window: If taper is 'True', return the values
                          of the hanning taper.
    :type return_window: bool
    :param padval: If this is not 'None', padval will be used to
                   pad the input array such that its length is
                   a multiple of the window length.
    :type padval: float, class:`numpy.nan`, or None
    :return: Array with the windowed data in columns.
    :rtype: :class:`~numpy.ndarray`
    """
    x = x.astype('float')
    npts = x.size
    if nwin > npts:
        msg = "window length can't be greater than array length"
        raise ValueError(msg)

    # step length that the window is advanced by
    step = nwin - noverlap
    # compute the remainder of the trace after dividing it
    # into windows
    rest = (npts - nwin) % step

    # if the remainder is >0 pad with zeros
    if rest > 0:
        npts_pad = npts + (step - rest)
        x = np.resize(x, int(npts_pad))
        x[npts:] = padval

    result = stride_windows(x, nwin, noverlap)
    if remove_mean:
        result = demean(result, axis=0)
    if taper:
        result, windowVals = apply_hanning(result, return_window=True)
        if return_window:
            return result, windowVals
    return result


def trace_window_times(trace, window_len, min_len=0.8):
    """Subdivide an `obspy.core.trace.Trace` into windows of `window_len`
    length and iterate over window start & end times.

    :param trace: The data to be windowed.
    :type trace: :class:`obspy.core.trace.Trace`
    :param window_len: The window length in seconds.
    :type window_len: float
    :param min_len: Windows with length < `min_len * window_len` are
        skipped, defaults to 0.8
    :type min_len: float, optional
    :yield: The window start & end times
    :rtype: :class:`obspy.core.utcdatetime.UTCDateTime`,
        :class:`obspy.core.utcdatetime.UTCDateTime`,
    """

    min_len = float(min_len)
    if not 0 < min_len <= 1.0:
        raise ValueError("min_len <= 0 or > 1.")
    min_len = min_len * window_len

    if (trace.stats.endtime - trace.stats.starttime) < min_len:
        raise ValueError("window length > total trace data.")

    t0 = trace.stats.starttime
    while t0 < trace.stats.endtime:
        t1 = min(t0 + window_len, trace.stats.endtime)
        this_window_len = t1 - t0
        if this_window_len >= min_len:
            if this_window_len < window_len:
                t0 = t1 - window_len
            yield t0, t1

        t0 += window_len


def trace_window_data(trace, window_len, min_len=0.8):
    """Subdivide an `obspy.core.trace.Trace` into windows of `interval`
    length and iterate over windows.

    :param trace: The data to be windowed.
    :type trace: :class:`obspy.core.trace.Trace`
    :param window_len: The window length in seconds.
    :type window_len: float
    :param min_len: Windows with length < `min_length * window_len` are
        skipped, defaults to 0.8
    :type min_len: float, optional
    :yield: A trace data windowNone
    :rtype: `obspy.core.trace.Trace`
    """

    for t0, t1 in trace_window_times(trace, window_len, min_len):
        yield trace.slice(t0, t1, nearest_sample=False)


def apply_freq_filter(
    data, filtertype, filterfreq, corners=4, zerophase=False, **options
):
    """Apply frequency filter to seismic data.

    Filter parameters are taken from instance attributes
    `filtertype` and `filterfreq`.

    :param data: The seismic data to be filtered.
    :type data: :class:`obspy.core.trace.Trace` or
        :class:``obspy.core.stream.Stream`
    :param corners: Filter corners / order, defaults to 4.
    :type corners: int
    :param zerophase: If True, apply filter once forwards and once
        backwards. Defaults to False.
    :type zerophase: bool
    :param options: Passed to underlying obspy filter methods.
    """
    f_min, f_max = filterfreq
    options.update({"zerophase": zerophase, "corners": corners})

    if filtertype in ["lp", "lowpass"]:
        if f_max is None:
            raise ValueError(
                "Define upper frequency cutoff for low-pass filter."
            )
        data.filter("lowpass", freq=f_max, **options)

    elif filtertype in ["hp", "highpass"]:
        if f_min is None:
            raise ValueError("Define lower frequency cutoff for high-pass filter.")
        data.filter("highpass", freq=f_min, **options)

    elif filtertype in ["bp", "bandpass"]:
        if f_min is None or f_max is None:
            raise ValueError(
                "Define upper and lower requency cutoff for band-pass filter."
            )
        data.filter("bandpass", freqmin=f_min, freqmax=f_max, **options)

    elif filtertype is not None:
        raise ValueError(f"Unrecognised filtertype '{filtertype}'")


def round_time(time, interval):
    """
    Find closest multiple of interval to time.

    :param time: Time to round
    :type time: :class:`obspy.UTCDateTime`
    :param interval: Interval length in seconds
    :type interval: int
    :returns: The new time that is a multiple of interval
    :rtype: :class:`obspy.UtCDateTime`
    """
    tmp = (time + 0.5 * interval).timestamp % interval
    ntime = (time + 0.5 * interval).timestamp - tmp
    return UTCDateTime(ntime)


def xarray2hdf5(xArray, fdir, rootGroupName="original", timedim="datetime"):
    """
    Store an xarray dataset as an HDF5 file.

    :param xArray: Data to store. 
    :type xArray: :class:`xarray.Dataset`
    :param fdir: Directory to store data under.
    :type fdir: str
    :param rootGroupName: Hdf5 group name.
    :type rootGroupName: str
    :param timedim: Name of time dimension.
    :type timedim: str
    """
    filterwarnings(action='ignore', category=DeprecationWarning,
               message='`np.bool` is a deprecated alias')

    for featureName in list(xArray.data_vars.keys()):
        h5file = os.path.join(fdir, featureName +'.nc')

        mode = 'a' if os.path.isfile(h5file) else 'w'
        
        with h5netcdf.File(h5file, mode) as h5f:
            try:
                rootGrp = _create_h5_Structure(rootGroupName, featureName, h5f, xArray)
            except ValueError: # group already exists, append
                rootGrp = h5f[rootGroupName]

            # determine indices
            new_time = date2num(xArray[timedim].values.astype('datetime64[us]').astype(datetime),
                                units=rootGrp[timedim].attrs['units'],
                                calendar=rootGrp[timedim].attrs['calendar'])
            dt = (np.diff(xArray['datetime'])/np.timedelta64(1, 'h'))[0]
            t0 = date2num(np.datetime64(rootGrp.attrs['starttime']).astype('datetime64[us]').astype(datetime),
                          units=rootGrp[timedim].attrs['units'],
                          calendar=rootGrp[timedim].attrs['calendar'])
            indices = np.rint((new_time - t0)/dt).astype(int)
            assert np.all(indices >= 0)
            times = rootGrp[timedim]
            newsize = indices[-1] + 1
            if newsize > times.shape[0]:
                rootGrp.resize_dimension(timedim, newsize)
            times[:] = t0 + np.arange(times.shape[0])*dt
            data = rootGrp[featureName]
            if len(data.shape) > 1:
                data[:, indices] = xArray[featureName].values
            else:
                data[indices] = xArray[featureName].values
            rootGrp.attrs['endtime'] = str(num2date(times[-1], units=rootGrp[timedim].attrs['units'],
                                                    calendar=rootGrp[timedim].attrs['calendar']))
            try:
                _setMetaInfo(featureName, h5f, xArray)
            except KeyError as e:
                logging.warning(f"Could not set all meta info for {featureName}: {e}")


def _create_h5_Structure(defaultGroupName, featureName, h5f, xArray):
    rootGrp = h5f.create_group(defaultGroupName)
    for label, size in xArray.dims.items(): 
        _setAttributes(label, size, rootGrp, xArray)
    # Note: xArray.dims returns a dictionary of dimensions that are not necesarily
    # in the right order; xArray[featureName].dims returns a tuple with dimension
    # names in the correct order
    rootGrp.create_variable(featureName, tuple(xArray[featureName].dims), dtype=float, fillvalue=0.)
    return rootGrp


def _setAttributes(label, size, rootGrp, xArray):
    if np.issubdtype(xArray[label].dtype, np.datetime64):
        starttime = str(xArray[label].values[0].astype('datetime64[us]').astype(datetime))
        rootGrp.dimensions[label] = None
        coordinates = rootGrp.create_variable(label, (label,), float)
        coordinates.attrs['units'] = 'hours since 1970-01-01 00:00:00.0'
        coordinates.attrs['calendar'] = 'gregorian'
        rootGrp.attrs['starttime'] = starttime
    else:
        rootGrp.dimensions[label] = size 
        coordinates = rootGrp.create_variable(label, (label,), float)
        coordinates[:] = xArray[label].values
        

def _setMetaInfo(featureName, h5f, xArray):
    h5f.attrs['station'] = xArray.attrs['station']
    h5f.attrs['latitude'] = -42
    h5f.attrs['longitude'] = 168
    h5f.attrs['datatype'] = featureName
    

def generate_test_data(dim=1, ndays=30, nfreqs=10,
                       tstart=datetime.utcnow(),
                       feature_name=None,
                       freq_name=None):
    """
    Generate a 1D or 2D feature for testing.
    """
    assert dim < 3
    assert dim > 0

    nints = ndays * 6 * 24
    dates = pd.date_range(tstart.strftime('%Y-%m-%d'), freq='10min', periods=nints)
    rs = np.random.default_rng(42)
    # Random walk as test signal
    data = np.abs(np.cumsum(rs.normal(0, 8., len(dates))))
    if dim == 2:
        data = np.tile(data, (nfreqs, 1))
    # Add 10% NaNs
    idx_nan = rs.integers(0, nints-1, int(0.1*nints))
    if dim == 1:
        data[idx_nan] = np.nan
        if feature_name is None:
            feature_name = 'rsam'
        xrd = xr.Dataset({feature_name: xr.DataArray(data, coords=[dates], dims=['datetime'])})
    if dim == 2:
        data[:, idx_nan] = np.nan
        freqs = np.arange(nfreqs)
        if feature_name is None:
            feature_name = 'ssam'
        if freq_name is None:
            freq_name = 'frequency'
        xrd = xr.Dataset({feature_name: xr.DataArray(data, coords=[freqs, dates], dims=[freq_name, 'datetime'])})
    xrd.attrs['starttime'] = UTCDateTime(dates[0]).isoformat()
    xrd.attrs['endtime'] = UTCDateTime(dates[-1]).isoformat()
    xrd.attrs['station'] = 'MDR'
    return xrd


def test_signal(nsec=3600, sampling_rate=100.,
                frequencies=[0.1, 3.0, 10.0],
                amplitudes=[0.1, 1.0, 0.7],
                phases=[0.0, np.pi*0.25, np.pi],
                offsets=[0., 0., 0.],
                starttime=UTCDateTime(1970, 1, 1),
                gaps=False,  noise=True, noise_std=.5,
                sinusoid=True,addchirp=True,
                network='NZ', station='BLUB', location='',
                channel='HHZ'):
    """
    Produce a test signal for which we know where the peaks
    are in the spectrogram.

    :param nsec: Length of the trace in seconds.
    :type nsec: int
    :param sampling_rate: Sampling rate of the signal in Hz.
    :type sampling_rate: float
    :param starttime: Starttime of the trace.
    :type starttime: :class:`~obspy.UTCDateTime`
    :param gaps: If 'True' add gaps to the test signal.
    :type gaps: boolean
    :param noise_std: Standard deviation of the noise.
    :type noise_std: float
    :param sinusoid: Add a signal with a sinusoidal frequency change.
    :type sinusoid: bool
    :param station: Station name of the returned trace.
    :type station: str
    :param location: Location of the returned trace.
    :type location: str
    :return: Time series of test data
    :rtype: :class:`~obspy.Trace`
    """
    t = np.arange(0, nsec, 1 / sampling_rate)
    signals = []
    # Some constant signals with different phases
    for f, A, ph, dt in zip(frequencies, amplitudes, phases, offsets):
        _s = np.zeros(t.size)
        dt_idx = int(dt*sampling_rate)
        _s[dt_idx:]= A * np.sin(2.0 * np.pi * f * t[dt_idx:] + ph)
        signals.append(_s)
    if addchirp:
        # Frequency-swept signals
        # Chirp
        s4 = 0.8 * chirp(t, 5, t[-1], 15, method='quadratic')
        signals.append(s4)

    if sinusoid:
        vals = [6]
        for k in range(0, 7):
            vals.append(2*(-1)**k*np.power(2*np.pi/nsec, 2*k+1)/M.factorial(2*k+1))
            vals.append(0)
        p = np.poly1d(np.array(vals[::-1]))
        s5 = sweep_poly(t, p)
        signals.append(s5)

    # add some noise
    if noise:
        rs = np.random.default_rng(42)
        noise = rs.normal(loc=0., scale=noise_std, size=t.size)
        for s in signals:
            noise += s
        signal = noise
    else:
        signal = signals[0]
        for s in signals[1:]:
            signal += s
    stats = {'network': network, 'station': station, 'location': location,
             'channel': channel, 'npts': len(signal),
             'sampling_rate': sampling_rate,
             'mseed': {'dataquality': 'D'}}
    stats['starttime'] = starttime
    stats['endtime'] = stats['starttime'] + nsec
    if gaps:
        p1 = (0.27, 0.33)
        p2 = (0.69, 0.83)
        p3 = (.95, 1)
        for pmin, pmax in [p1, p2, p3]:
            idx0 = int(pmin * nsec * sampling_rate)
            idx1 = int(pmax * nsec * sampling_rate)
            signal[idx0:idx1] = np.nan
    return Trace(data=signal, header=stats)

