import inspect
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from obspy import Trace, UTCDateTime
from zizou.cpxtrace import (
    Centroid, Inst_Band, Inst_Freq, Norm_Env, WindowCentroid
)
from zizou.data import DataSource
from zizou.util import test_signal

from zizou.cpxtrace import Inst_Band, Inst_Freq, Norm_Env, Centroid, WindowCentroid


@unittest.skip("Tested code needs re-write")
class TestCpxtrace(unittest.TestCase):

    def setUp(self):
        # set the path to test data
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        self.tr1 = self.test_signal()
        self.tr2 = self.test_signal(gaps=True)

    def test_signal(self, gaps=False):
        nsec = 6400.
        sampling_rate = 100.
        signal = -1e-9 * np.ones(int(nsec * sampling_rate))
        if gaps:
            idx0 = int(500 * sampling_rate)
            idx1 = int(1200 * sampling_rate)
            signal[idx0:idx1] = np.nan

        stats = {'network': 'NZ', 'station': 'BLUB', 'location': '',
                 'channel': 'HHZ', 'npts': len(signal),
                 'sampling_rate': sampling_rate,
                 'mseed': {'dataquality': 'D'}}
        stats['starttime'] = UTCDateTime()
        stats['endtime'] = stats['starttime'] + nsec
        tr = Trace(data=signal, header=stats)
        return tr

    def test_with_gaps(self):
        instband = Inst_Band(interval=600., fil_coef=(-1.0, 0.0, 1.0))
        df_instB = instband.compute(self.tr1)
        np.testing.assert_almost_equal(
            df_instB.values,
            np.r_[np.linspace(0.00125562, 0.00125562, 10)].reshape((10, 1)),
            8
        )
        df_instB_gaps = instband.compute(self.tr2)
        np.testing.assert_almost_equal(
            df_instB_gaps.values,
            np.r_[np.nan, np.nan,
                  np.linspace(0.00125562, 0.00125562, 8)
                  ].reshape((10, 1)),
            8
        )

        instfreq = Inst_Freq(interval=600., fil_coef=(-1.0, 0.0, 1.0))
        df_instF = instfreq.compute(self.tr1)
        np.testing.assert_almost_equal(
            df_instF.values,
            np.r_[np.linspace(0.00141813, 0.00141813, 10)].reshape((10, 1)),
            8
        )
        df_instF_gaps = instfreq.compute(self.tr2)
        np.testing.assert_almost_equal(
            df_instF_gaps.values,
            np.r_[np.nan, np.nan,
                  np.linspace(0.00141813, 0.00141813, 8)
                  ].reshape((10, 1)),
            8
        )
        normenv = Norm_Env(window=10, interval=600., fil_coef=(-1.0, 0.0, 1.0))
        df_normE = normenv.compute(self.tr1)
        np.testing.assert_almost_equal(
            df_normE.values,
            np.r_[np.linspace(8.59312621e-12, 8.59312621e-12, 10)
                  ].reshape((10, 1)),
            8
        )

        df_normE_gaps = normenv.compute(self.tr2)
        np.testing.assert_almost_equal(
            df_normE_gaps.values,
            np.r_[np.nan, np.nan,
                  np.linspace(8.59312621e-12, 8.59312621e-12, 8)
                  ].reshape((10, 1)),
            8
        )

    def test_inst_band(self):
        """
        Test instantaneous bandwidth method
        """
        stream = 'WIZ.10.HHZ.NZ'
        ds = DataSource(source='fdsn')

        startdate = UTCDateTime('2019-07-14')
        enddate = UTCDateTime('2019-07-15')

        site, loc, comp, net = str.split(stream, '.')
        instband = Inst_Band(interval=600., fil_coef=(-1.0, 0.0, 1.0))
        for tr in ds.get_waveforms(net, site, loc, comp,
                                   startdate, enddate,
                                   fill_value='interpolate'):
            omega = instband.compute(tr)

        omega_test = pd.read_csv(os.path.join(self.data_dir,
                                              'WIZ_instband.csv'),
                                 index_col=0, parse_dates=True)

        pd.testing.assert_frame_equal(omega, omega_test)

    def test_inst_freq(self):
        """
        Test instantaneous frequency method
        """
        stream = 'WIZ.10.HHZ.NZ'
        ds = DataSource(source='fdsn')

        startdate = UTCDateTime('2019-07-14')
        enddate = UTCDateTime('2019-07-15')

        site, loc, comp, net = str.split(stream, '.')
        instfreq = Inst_Freq(interval=600., fil_coef=(-1.0, 0.0, 1.0))

        for tr in ds.get_waveforms(net, site, loc, comp,
                                   startdate, enddate,
                                   fill_value='interpolate'):
            sigma = instfreq.compute(tr)

        sigma_test = pd.read_csv(os.path.join(self.data_dir,
                                              'WIZ_instfreq.csv'),
                                 index_col=0, parse_dates=True)

        pd.testing.assert_frame_equal(sigma, sigma_test)

    def test_norm_env(self):
        """
        Test Normalize envelope method
        """
        stream = 'WIZ.10.HHZ.NZ'
        ds = DataSource(source='fdsn')

        startdate = UTCDateTime('2019-08-14T00:00:01')
        enddate = startdate + 86400.

        site, loc, comp, net = str.split(stream, '.')
        normenv = Norm_Env(window=10, interval=600., fil_coef=(-1.0, 0.0, 1.0))

        for tr in ds.get_waveforms(net, site, loc, comp,
                                   startdate, enddate,
                                   fill_value='interpolate'):
            norm_env = normenv.compute(tr)

        norm_env_test = pd.read_csv(os.path.join(self.data_dir,
                                                 'WIZ_norm_env.csv'),
                                    index_col=0, parse_dates=True)

        pd.testing.assert_frame_equal(norm_env, norm_env_test)

    @unittest.skip('Slow test')
    def test_centroid(self):
        """
        Test instantaneous frequency method
        """
        stream = 'WIZ.10.HHZ.NZ'
        ds = DataSource(source='fdsn')

        startdate = UTCDateTime('2019-08-14')
        enddate = startdate + 1000.

        site, loc, comp, net = str.split(stream, '.')

        for tr in ds.get_waveforms(net, site, loc, comp,
                                   startdate, enddate,
                                   attach_response=True):
            centro = Centroid.compute(tr)

        centro_test = pd.read_csv(os.path.join(self.data_dir,
                                               'WIZ_centroid.csv'),
                                  index_col=0, parse_dates=True)

        centro_test.columns = ['']

        pd.testing.assert_frame_equal(centro, centro_test)

@unittest.skip("Tested code needs re-write")
class WindowCentroidTestCase(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        stream = 'WIZ.10.HHZ.NZ'
        ds = DataSource(source='fdsn')
        # dates are inclusive, and complete days
        startdate = UTCDateTime('2019-08-14')
        enddate = UTCDateTime('2019-08-15')

        site, loc, comp, net = str.split(stream, '.')

        gen = ds.get_waveforms(net, site, loc, comp,
                               startdate, enddate,
                               fill_value='interpolate')
        self.trace = next(gen)

    def test_window_centroid(self):
        feat = WindowCentroid(interval=600)
        df = feat.compute(self.trace)
        df_test = pd.read_csv(os.path.join(
            self.data_dir, 'window_centroid_test.csv'),
            index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(df, df_test)

if __name__ == '__main__':
    unittest.main()
