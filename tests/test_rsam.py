import inspect
import os
import shutil
import tempfile
import unittest

import numpy as np
from obspy import Trace, UTCDateTime
import pandas as pd
import xarray as xr

from zizou.rsam import RSAM, EnergyExplainedByRSAM
from zizou.data import DataSource


class RSAMTestCase(unittest.TestCase):

    def setUp(self):
        self.tr1 = self.test_signal()
        self.tr2 = self.test_signal(gaps=True)
        self.tempdir1 = tempfile.mkdtemp()
        self.tempdir2 = tempfile.mkdtemp()
        self.tempdir3 = tempfile.mkdtemp()

    def tearDown(self):
        for _dir in [self.tempdir1, self.tempdir2, self.tempdir3]:
            if os.path.isdir(_dir):
                shutil.rmtree(_dir)

    def test_signal(self, gaps=False, starttime=None):
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
        if starttime is not None:
            stats['starttime'] = starttime
        stats['endtime'] = stats['starttime'] + nsec
        tr = Trace(data=signal, header=stats)
        return tr

    def test_with_gaps(self):
        r = RSAM(filtertype=None)
        df = r.compute(self.tr1)
        np.testing.assert_array_almost_equal(df['rsam'].values, np.ones(10), 2)
        df1 = r.compute(self.tr2)
        np.testing.assert_array_almost_equal(
            df1['rsam'].values, np.r_[np.nan, np.nan, np.ones(8)], 2
        )

    def test_append(self):
        """
        Test appending new data to existing output files.
        """

        starttime1 = UTCDateTime('2019-08-14T01:00:00')
        endtime1 = UTCDateTime('2019-08-14T01:20:00')
        starttime2 = endtime1
        endtime2 = UTCDateTime('2019-08-14T01:40:00')

        tr = self.test_signal(starttime=starttime1)
        tr1 = tr.copy().trim(starttime1, endtime1)
        tr2 = tr.copy().trim(starttime2, endtime2)
        tr2.data *= 2
        # hack to make test results consistent
        # necessary because time windows overlap
        tr1.data[-1] *= 2
        tr3 = tr1.__add__(tr2, method=1)
        r1 = RSAM(interval=600, filtertype=None)
        r2 = RSAM(interval=600, filtertype=None)
        r3 = RSAM(interval=600, filtertype=None)

        _ = r1.compute(tr1)
        r1.save(self.tempdir1)

        _ = r2.compute(tr2)
        r2.save(self.tempdir1)

        _ = r3.compute(tr3)
        r3.save(self.tempdir2)

        file1 = os.path.join(self.tempdir1, 'rsam.nc')
        file2 = os.path.join(self.tempdir2, 'rsam.nc')
        ds1 = xr.open_dataset(file1, group='original')
        ds2 = xr.open_dataset(file2, group='original')
        np.testing.assert_array_equal(ds1['datetime'],
                                      ds2['datetime'])
        np.testing.assert_array_equal(ds1['rsam'],
                                      ds2['rsam'])

        # test overlapping time windows
        tr4 = tr.copy().trim(starttime1, endtime2)
        r4 = RSAM(interval=600, filtertype=None)
        _ = r4.compute(tr4)
        r4.save(self.tempdir3)
        r2.save(self.tempdir3)
        file3 = os.path.join(self.tempdir3, 'rsam.nc')
        ds3 = xr.open_dataset(file3, group='original')
        np.testing.assert_array_equal(ds3['rsam'].values.astype(int),
                                      np.array([1, 1, 2, 2]))


class EnergyExplainedByRSAMTestCase(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        stream = 'WIZ.10.HHZ.NZ'
        ds = DataSource(sources=('fdsn',), fill_value='interpolate')
        # dates are inclusive, and complete days
        startdate = UTCDateTime('2019-08-14')
        enddate = UTCDateTime('2019-08-15')

        site, loc, comp, net = str.split(stream, '.')

        gen = ds.get_waveforms(net, site, loc, comp,
                               startdate, enddate)
        self.trace = next(gen)

    def test_energy(self):
        feat = EnergyExplainedByRSAM(
            interval=600, filtertype='bp', filterfreq=(2, 5)
        )
        df = feat.compute(self.trace).to_dataframe()
        fname = os.path.join(self.data_dir, 'rsam_energy_test.csv')
        df_test = pd.read_csv(
            fname, index_col="datetime", parse_dates=True
        )
        pd.testing.assert_frame_equal(df, df_test)

