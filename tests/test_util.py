import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from obspy import Trace, UTCDateTime
from obspy.signal.util import enframe
from tonik.utils import generate_test_data

from zizou.util import (
    apply_freq_filter,
    apply_hanning,
    demean,
    round_time,
    stack_feature,
    stride_windows,
    trace_window_data,
    trace_window_times,
    window_array,
)


def test_stack_feature():
    """
    Test the stack_features function.
    """
    data = generate_test_data(
        dim=1, ndays=1, tstart=datetime(2021, 4, 3), add_nans=False
    )
    stack_length = "1h"
    xdf = stack_feature(data, stack_length)
    assert np.alltrue(xdf.datetime == data.datetime)

    a1 = data["rsam"].loc[dict(datetime="2021-04-03T00")].values
    a2 = xdf["rsam"].loc[dict(datetime="2021-04-03T00")].values
    assert a2[5] == (a1[:6]).sum() / len(a1[:6])
    stack_len_seconds = 3600

    num_windows = int(stack_len_seconds / pd.Timedelta(xdf.interval).seconds)
    rsam = data["rsam"]

    # Check correct values
    rolling_mean = [
        np.nanmean(rsam.data[(ind - num_windows + 1) : ind + 1])
        for ind in np.arange(num_windows, rsam.shape[0])
    ]
    np.testing.assert_array_almost_equal(
        np.array(rolling_mean), xdf.rsam.data[num_windows:], 6
    )


def test_stride_windows():
    data = np.random.randint(0, 10, size=10)
    test_array = np.vstack((data[0:5], data[3:8]))
    ds = stride_windows(data, 5, 2)
    np.testing.assert_array_equal(test_array.T, ds)
    ds = stride_windows(data[0:8], 5, 2)
    np.testing.assert_array_equal(test_array.T, ds)
    # Because the returned array is just a
    # different view on the original array,
    # you can't assign to it
    with pytest.raises(ValueError):
        ds[0, 0] = 5

    # But because ds is just a view, changing the
    # original array will change ds is, maybe,
    # unexpected ways:
    data[3] = 100
    assert ds[0, 1] == 100


def test_obspy_enframe():
    """
    Compare stride_windows to obspy's enframe which doesn't
    use 'as_strided'.
    """
    data = np.random.randint(0, 10, size=10)
    ds = window_array(data, 4, 2, taper=False, padval=None)
    ds1, nwin, nowin = enframe(data, np.ones(4), 2)
    np.testing.assert_array_equal(ds.T, ds1)


def test_demean():
    a = np.arange(9.0)
    ds = stride_windows(a, 3, 2)
    np.testing.assert_array_equal(demean(ds, axis=0)[0], -np.ones(7))
    a[1] = np.nan
    np.testing.assert_array_equal(demean(ds, axis=0)[1, 1], -0.5)


def test_taper():
    a = np.arange(9)
    ds = stride_windows(a, 3, 1)
    dsw, wvals = apply_hanning(ds, return_window=True)
    np.testing.assert_array_equal(dsw[:, 0], np.array([0, 1.0, 0]))
    np.testing.assert_array_equal(dsw[1], ds[1])
    np.testing.assert_array_equal(wvals, np.array([0, 1, 0])[:, np.newaxis])


def test_window_array():
    data = np.arange(10)
    # This will produce 4 windows with 5 elements each
    # overlapping by 3 elements; the last window will
    # contain one 0 to bring it to size 5
    res = window_array(data, 5, 3, remove_mean=False, taper=False, padval=0.0)
    assert res.shape == (5, 4)
    assert res[-1, -1] == 0
    # The returned array is just a different view of the
    # original so you can't assign to it
    with pytest.raises(ValueError):
        res[0, 0] = 100
    # Same as above but also remove the mean from each
    # column
    resm = window_array(data, 5, 3, remove_mean=True, taper=False, padval=0.0)
    np.testing.assert_array_equal(resm.sum(axis=0), np.zeros(resm.shape[1]))
    # The returned array is a new copy of the original
    # so you now can assign to it
    resm[0, 0] = 100
    assert resm[0, 0] == 100

    # Now remove the mean and also taper each window. This
    # is the default behaviour
    rest, wvals = window_array(data, 5, 3, padval=0.0)
    np.testing.assert_array_equal(rest[1], resm[1] * wvals[1])

    # This also works if we pad with np.nan
    rest, wvals = window_array(data, 5, 3, padval=np.nan)
    res_cp = res.copy()
    res_cp[-1, -1] = np.nan
    res_test = (res_cp[:4] - res_cp[:4, -1].sum() / 4) * wvals[:4]
    np.testing.assert_array_equal(rest[:4, -1], res_test[:4, -1])


def test_trace_window_data():
    tr = _dummy_trace_data(901)
    interval = 1.0
    result = list(trace_window_data(tr, interval))
    assert len(result) == 9
    np.testing.assert_array_equal(tr.data[100:201], result[1].data)
    with pytest.raises(ValueError):
        list(trace_window_times(tr, 100000))


def test_trace_window_times():
    n_samp = 901
    hz = 100.0
    tr = _dummy_trace_data(n_samp, hz)
    tr_len = (n_samp - 1) / hz

    # test simple case
    interval = 1.0
    n_windows = int(tr_len / interval)
    result = list(trace_window_times(tr, interval, interval))
    assert len(result) == n_windows
    assert tr.stats.starttime + 3 * interval == result[3][0]
    # test cases where final window len < interval
    interval = 1.05
    n_windows = int(tr_len / interval)
    result = list(trace_window_times(tr, interval, 1))
    assert len(result) == n_windows
    result = list(trace_window_times(tr, interval, 0.2))
    assert len(result) == n_windows + 1
    with pytest.raises(ValueError):
        list(trace_window_times(tr, 100000))


def test_apply_freq_filter():
    f = (2, 10)
    ftype = "bp"
    tr_untouched = _dummy_trace_data()
    tr1 = tr_untouched.copy()
    tr2 = tr_untouched.copy()
    apply_freq_filter(tr1, ftype, f)
    tr2.filter("bandpass", freqmin=f[0], freqmax=f[1], corners=4, zerophase=False)
    np.testing.assert_array_equal(tr1.data, tr2.data)


def test_round_time():
    """
    Test find closest time to interval.
    """
    time = UTCDateTime(2021, 5, 21, 13, 8)
    assert round_time(time, 600) == UTCDateTime(2021, 5, 21, 13, 10)


def _dummy_trace_data(n=2048, hz=100.0):
    t = np.linspace(0, 40 * np.pi, n) + np.linspace(0, 13 * np.pi, n)
    return Trace(
        np.sin(t), header={"starttime": "2020-01-01T12:00:00", "sampling_rate": hz}
    )


if __name__ == "__main__":
    unittest.main()
