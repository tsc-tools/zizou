import inspect
import os
import unittest

import numpy as np
from obspy import UTCDateTime
import pandas as pd

from zizou.dsar import DSAR
from zizou.data import DataSource


class DSARTestCase(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    @unittest.skip("Tested code needs re-write as it's testing agains live FDSN server")
    def test_daily_ratio(self):
        dsar = DSAR(filtertype='bandpass', lowerfreqband=(4.5, 8),
                    higherfreqband=(8, 16))
        stream = 'KRVZ.10.EHZ.NZ'
        # dates are inclusive, and complete days
        startdate = UTCDateTime('2012-11-01')
        enddate = UTCDateTime('2012-11-03')
        ds = DataSource(sources=('fdsn',), fill_value='interpolate')
        site, loc, comp, net = stream.split('.')
        for tr in ds.get_waveforms(net, site, loc, comp,
                                   startdate, enddate):
            df = dsar.compute(tr)
        df_test = pd.read_csv(os.path.join(self.data_dir,
                                           'dsar_test.csv'),
                              index_col=0, parse_dates=True).to_xarray()
        np.testing.assert_array_almost_equal(df['dsar'].values, df_test['dsar'].values)

