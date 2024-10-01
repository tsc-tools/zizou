import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
from obspy import Trace, UTCDateTime
from scipy.signal import find_peaks
from tonik import Storage
import xarray as xr

from zizou.ssam import SSAM
from zizou.util import test_signal


class SSAMTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        data = np.random.randint(0, 10, size=10)
        self.tr = Trace(data=data, header=dict(sampling_rate=1))
        self.outdir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)

    def test_ssam(self):
        test_peaks = np.array([1., 3., 10.])
        s = SSAM(interval=60, per_lap=.9, smooth=10)
        spec = s.compute(test_signal(sinusoid=False, addchirp=False,
                                     frequencies=[1., 3., 10.]))
        peaks = find_peaks(spec['ssam'].data[:, 0], height=.1)[0]
        np.testing.assert_array_almost_equal(spec['frequency'].values[peaks],
                                             test_peaks, 2)
        
    def test_sonogram(self):
        s = SSAM(interval=60, per_lap=.9, smooth=10)
        test_trace = test_signal(offsets=[1000., 2000., 3000.],
                                 frequencies=[1., 2., 5.],
                                 amplitudes=[0.1, 1., 5.],
                                 sinusoid=False,
                                 addchirp=False)
        spec = s.compute(test_trace)
        idx1 = (int(np.abs(spec.datetime - 
                np.datetime64((test_trace.stats.starttime + 1500.)
                .datetime)).argmin()))
        idx2 = (int(np.abs(spec.datetime - 
                np.datetime64((test_trace.stats.starttime + 2600.)
                .datetime)).argmin()))
        idx3 = (int(np.abs(spec.datetime - 
                np.datetime64((test_trace.stats.starttime + 3500.)
                .datetime)).argmin()))
        self.assertAlmostEqual(float(spec.sonofrequency[spec['sonogram']
                                .isel(dict(datetime=idx1)).argmax()]), 2.295, 3)
        self.assertAlmostEqual(float(spec.sonofrequency[spec['sonogram']
                                .isel(dict(datetime=idx2)).argmax()]), 5.164, 3)
        self.assertAlmostEqual(float(spec.sonofrequency[spec['sonogram']
                                .isel(dict(datetime=idx3)).argmax()]), 11.62, 2)

    def test_filterbank(self):
        s = SSAM(interval=60, per_lap=.9, smooth=10)
        test_trace = test_signal(offsets=[1000., 2000., 3000.],
                                 frequencies=[0.1, 2., 5.],
                                 amplitudes=[0.1, 1., 5.],
                                 sinusoid=False,
                                 addchirp=False)
        spec = s.compute(test_trace)
        idx1 = (int(np.abs(spec.datetime - 
                np.datetime64((test_trace.stats.starttime + 1500.)
                .datetime)).argmin()))
        idx2 = (int(np.abs(spec.datetime - 
                np.datetime64((test_trace.stats.starttime + 2600.)
                .datetime)).argmin()))
        idx3 = (int(np.abs(spec.datetime - 
                np.datetime64((test_trace.stats.starttime + 3500.)
                .datetime)).argmin()))
        self.assertAlmostEqual(float(spec.fbfrequency[spec['filterbank']
                                .isel(dict(datetime=idx1)).argmax()]), 2.295, 3)
        self.assertAlmostEqual(float(spec.fbfrequency[spec['filterbank']
                                .isel(dict(datetime=idx2)).argmax()]), 5.164, 3)
        self.assertAlmostEqual(float(spec.fbfrequency[spec['filterbank']
                                .isel(dict(datetime=idx3)).argmax()]), 11.62, 2)


    def test_ssam_with_gaps(self):
        linfreqs = np.arange(0, 25.1, 0.1)
        test_peaks = np.array([0.1, 3., 5, 6.1, 10.])
        s = SSAM(interval=60, per_lap=.9, smooth=10,
                 frequencies=linfreqs)
        spec = s.compute(test_signal(gaps=True))
        peaks = find_peaks(spec['ssam'].data[:, 0], height=1e-1)[0]
        np.testing.assert_array_almost_equal(linfreqs[peaks], test_peaks, 2)

    def test_realtime(self):
        """
        Test with 1 hour windows as in the real-time computation
        """
        linfreqs = np.arange(0, 25.1, 0.1)
        testpeaks = np.array([0.1, 3., 5., 6., 10.])
        s = SSAM(interval=60, frequencies=linfreqs,
                 resample_int=(None, '10min'), timestamp='start')
        starttime = UTCDateTime(2020, 4, 21, 0, 0, 0)
        nsec = 86500 # Slightly longer than 24 h to avoid rounding issues
        step = 3600
        tr = test_signal(nsec=nsec, starttime=starttime, gaps=True)
        sg = Storage('test', self.outdir, starttime=starttime.datetime,
                          endtime=(starttime + nsec).datetime)
        store = sg.get_substore('WIZ', '00', 'HHZ')
        startwin = starttime
        while True:
            endwin = startwin + step
            if endwin > tr.stats.endtime:
                break
            ntr = tr.slice(startwin, endwin)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.save(s.compute(ntr))
            startwin = endwin

        ssam = store('ssam')
        self.assertEqual(ssam.shape, (251, 144))
        peaks = find_peaks(ssam.data[:, 0], height=1e-1)[0]
        np.testing.assert_array_almost_equal(linfreqs[peaks],
                                             testpeaks, 2)

    def test_filterbank(self):
        s = SSAM()
        _ = s.filterbank(10, 512, 100.)

    def test_config(self):
        """
        Test parsing of config parameters.
        """
        config = """
        default:
          interval: 600
        ssam:
          interval: 60
          per_lap: .9
          scale_by_freq: true
          timestamp: start
          frequencies:
            start: 0
            end: 25.1
            step: 0.1
          resample_int:
            upsample: null
            downsample: 10min
        """ 
        s = SSAM(configfile=config)
        self.assertEqual(s.resample_int, (None, '10min'))
        np.testing.assert_equal(s.frequencies, np.arange(0, 25.1, .1))


