import datetime
import os
import shutil
import tempfile
import unittest
import numpy as np

import h5py
from obspy import Trace, UTCDateTime
from obspy.signal.util import enframe
import xarray as xr

from zizou.util import (stride_windows,
                       demean,
                       apply_hanning,
                       window_array,
                       trace_window_data,
                       trace_window_times,
                       apply_freq_filter,
                       round_time,
                       test_signal)
from zizou.ssam import SSAM
from zizou.rsam import RSAM


class UtilTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.isdir(self.tempdir):
            shutil.rmtree(self.tempdir)

    def test_stride_windows(self):
        data = np.random.randint(0, 10, size=10)
        test_array = np.vstack((data[0:5], data[3:8]))
        ds = stride_windows(data, 5, 2)
        np.testing.assert_array_equal(test_array.T, ds)
        ds = stride_windows(data[0:8], 5, 2)
        np.testing.assert_array_equal(test_array.T, ds)
        # Because the returned array is just a
        # different view on the original array,
        # you can't assign to it
        with self.assertRaises(ValueError):
            ds[0, 0] = 5

        # But because ds is just a view, changing the
        # original array will change ds is, maybe,
        # unexpected ways:
        data[3] = 100
        self.assertEqual(ds[0, 1], 100)
        
    def test_obspy_enframe(self):
        """
        Compare stride_windows to obspy's enframe which doesn't
        use 'as_strided'.
        """
        data = np.random.randint(0, 10, size=10)
        ds = window_array(data, 4, 2, taper=False, padval=None)
        ds1, nwin, nowin  = enframe(data, np.ones(4), 2)
        np.testing.assert_array_equal(ds.T, ds1)

    def test_demean(self):
        a = np.arange(9.)
        ds = stride_windows(a, 3, 2)
        np.testing.assert_array_equal(demean(ds, axis=0)[0], -np.ones(7))
        a[1] = np.nan
        np.testing.assert_array_equal(demean(ds, axis=0)[1, 1], -0.5)

    def test_taper(self):
        a = np.arange(9)
        ds = stride_windows(a, 3, 1)
        dsw, wvals = apply_hanning(ds, return_window=True)
        np.testing.assert_array_equal(dsw[:, 0], np.array([0, 1., 0]))
        np.testing.assert_array_equal(dsw[1], ds[1])
        np.testing.assert_array_equal(wvals,
                                      np.array([0, 1, 0])[:, np.newaxis])

    def test_window_array(self):
        data = np.arange(10)
        # This will produce 4 windows with 5 elements each
        # overlapping by 3 elements; the last window will
        # contain one 0 to bring it to size 5
        res = window_array(data, 5, 3, remove_mean=False,
                           taper=False, padval=0.)
        self.assertEqual(res.shape, (5, 4))
        self.assertEqual(res[-1, -1], 0)
        # The returned array is just a different view of the
        # original so you can't assign to it
        with self.assertRaises(ValueError):
            res[0, 0] = 100
        # Same as above but also remove the mean from each
        # column
        resm = window_array(data, 5, 3, remove_mean=True,
                            taper=False, padval=0.)
        np.testing.assert_array_equal(resm.sum(axis=0),
                                      np.zeros(resm.shape[1]))
        # The returned array is a new copy of the original
        # so you now can assign to it
        resm[0, 0] = 100
        self.assertEqual(resm[0, 0], 100)

        # Now remove the mean and also taper each window. This
        # is the default behaviour
        rest, wvals = window_array(data, 5, 3, padval=0.)
        np.testing.assert_array_equal(rest[1], resm[1] * wvals[1])

        # This also works if we pad with np.nan
        rest, wvals = window_array(data, 5, 3, padval=np.nan)
        res_cp = res.copy()
        res_cp[-1, -1] = np.nan
        res_test = (res_cp[:4] - res_cp[:4, -1].sum()/4) * wvals[:4]
        np.testing.assert_array_equal(rest[:4, -1], res_test[:4, -1])

    def test_trace_window_data(self):
        tr = _dummy_trace_data(901)
        interval = 1.
        result = list(trace_window_data(tr, interval))
        self.assertEqual(len(result), 9)
        np.testing.assert_array_equal(tr.data[100:201], result[1].data)
        with self.assertRaises(ValueError):
            list(trace_window_times(tr, 100000))

    def test_trace_window_times(self):
        n_samp = 901
        hz = 100.
        tr = _dummy_trace_data(n_samp, hz)
        tr_len = (n_samp - 1) / hz

        # test simple case
        interval = 1.
        n_windows = int(tr_len / interval)
        result = list(trace_window_times(tr, interval, interval))
        self.assertEqual(len(result), n_windows)
        self.assertEqual(tr.stats.starttime + 3 * interval, result[3][0])
        # test cases where final window len < interval
        interval = 1.05
        n_windows = int(tr_len / interval)
        result = list(trace_window_times(tr, interval, 1))
        self.assertEqual(len(result), n_windows)
        result = list(trace_window_times(tr, interval, 0.2))
        self.assertEqual(len(result), n_windows + 1)
        with self.assertRaises(ValueError):
            list(trace_window_times(tr, 100000))

    def test_apply_freq_filter(self):
        f = (2, 10)
        ftype = "bp"
        tr_untouched = _dummy_trace_data()
        tr1 = tr_untouched.copy()
        tr2 = tr_untouched.copy()
        apply_freq_filter(tr1, ftype, f)
        tr2.filter(
            "bandpass", freqmin=f[0], freqmax=f[1], corners=4, zerophase=False
        )
        np.testing.assert_array_equal(tr1.data, tr2.data)

    def test_round_time(self):
        """
        Test find closest time to interval.
        """
        time = UTCDateTime(2021,5,21,13,8)
        self.assertEqual(round_time(time, 600),
                         UTCDateTime(2021,5,21,13,10))


def _dummy_trace_data(n=2048, hz=100.):
    t = np.linspace(0, 40 * np.pi, n) + np.linspace(0, 13 * np.pi, n)
    return Trace(
        np.sin(t),
        header={"starttime": "2020-01-01T12:00:00", "sampling_rate": hz}
    )


if __name__ == '__main__':
    unittest.main()
