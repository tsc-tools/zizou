import inspect
import os

import numpy as np
from obspy import UTCDateTime
import pandas as pd
import pytest

from zizou.rsam import RSAM
from zizou.util import test_signal
    

def original_rsam(trace):
    """
    RSAM implementation by Steve Sherburn
    """
    f1 = 1
    f2 = 4
    data = np.zeros(145)
    t = trace.stats.starttime
    nans = np.isnan(trace.data)
    indices = np.arange(trace.stats.npts)
    trace.data[nans] = np.interp(indices[nans], indices[~nans], trace.data[~nans]) 
    index = 0
    #loop through data in 600sec (10 min) blocks
    while t < trace.stats.endtime:
        tr_10m = trace.slice(t, t + 600)

        duration = tr_10m.stats.npts * tr_10m.stats.delta
        if duration >= 500:
            if duration < 600:
                tr_10m = trace.slice(trace.stats.endtime - 600, trace.stats.endtime) 
            tr_10m.detrend(type='constant')
            tr_10m.filter('bandpass', freqmin=f1, freqmax=f2, corners=4, zerophase=False)
            absolute = np.absolute(tr_10m.data)	#absolute value
            tr_10m.data = absolute	#assign back to trace
            mean = tr_10m.data.mean()
            mean = mean / 1e-9	#convert to nanometres so dealing with whole numbers
            data[index] = mean
            index += 1
        t += 600
    data = np.resize(data, index)
    return data


@pytest.fixture(scope="session")
def rsam_setup():
    amplitudes = [0.1, 1.]
    offsets = [1000, 2000]
    frequencies = [.1, 3.]
    phases = [0., np.pi * .25]
    starttime = UTCDateTime(1978, 7, 18, 8, 8, 24)
    tr1 = test_signal(sinusoid=False, addchirp=False,
                      starttime=starttime,
                      offsets=offsets,
                      amplitudes=amplitudes,
                      phases=phases, frequencies=frequencies)
    tr1.data *= 1e-9
    tr2 = test_signal(sinusoid=False, addchirp=False,
                      starttime=starttime,
                      offsets=offsets, amplitudes=amplitudes,
                      phases=phases, frequencies=frequencies,
                      gaps=True)
    tr2.data *= 1e-9
    data_dir = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "data")

    return tr1, tr2, data_dir

def test_with_gaps(rsam_setup, setup_sds):
    tr1, tr2, data_dir = rsam_setup
    r = RSAM()
    xdf = r.compute(tr1)
    max_err = np.abs((xdf['rsam'].values - original_rsam(tr1.copy()))).max() 
    assert max_err < 0.01 

    xdf1 = r.compute(tr2)
    max_err = np.abs((xdf1['rsam'].values - original_rsam(tr2.copy()))).max() 
    assert max_err < 0.01 
    assert UTCDateTime(str(xdf1.datetime.values[0])) == UTCDateTime(xdf1.attrs['starttime'])
    assert UTCDateTime(str(xdf1.datetime.values[-1])) == UTCDateTime(xdf1.attrs['starttime']) + 5. * 600.

    sds_dir, ds = setup_sds
    r = RSAM()
    startdate = UTCDateTime('2019-08-14')
    enddate = UTCDateTime('2019-08-15')
    gen = ds.get_waveforms('NZ', 'WIZ', '10', 'HHZ',
                            startdate, enddate)
    tr3 = next(gen)
    xdf3 = r.compute(tr3)

    fname = os.path.join(data_dir, 'rsam_energy_test.csv')
    df_test = pd.read_csv(
        fname, index_col="datetime", parse_dates=True
    )
    max_err = (xdf3.rsam_energy_prob.values - df_test.values.squeeze()).max()
    assert max_err < 0.01
