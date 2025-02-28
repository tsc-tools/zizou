import inspect
import os

import numpy as np
from obspy import UTCDateTime
import pandas as pd
import pytest

from zizou.dsar import DSAR
from zizou.data import DataSource

@pytest.fixture(scope="session")
def dsar_setup():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "data")
    return data_dir

def test_daily_ratio(dsar_setup, setup_sds):
    data_dir = dsar_setup
    sds_dir, ds = setup_sds
    dsar = DSAR(filtertype='bandpass', lowerfreqband=(4.5, 8),
                higherfreqband=(8, 16))
    stream = 'KRVZ.10.EHZ.NZ'
    # dates are inclusive, and complete days
    startdate = UTCDateTime('2012-11-01')
    enddate = UTCDateTime('2012-11-03')
    site, loc, comp, net = stream.split('.')
    for tr in ds.get_waveforms(net, site, loc, comp,
                                startdate, enddate):
        df = dsar.compute(tr)
    df_test = pd.read_csv(os.path.join(data_dir,
                                        'dsar_test.csv'),
                            index_col=0, parse_dates=True).to_xarray()
    np.testing.assert_array_almost_equal(df['dsar'].values,
                                         df_test['dsar'].values, 3)

