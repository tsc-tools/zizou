import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import pytest
from obspy import UTCDateTime
from tonik import Storage

from zizou.spectral_features import SpectralFeatures
from zizou.util import test_signal


class SpectralTestCase(unittest.TestCase):
    def setUp(self):
        self.outdir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)

    sup = np.testing.suppress_warnings()
    sup.filter(module=np.matrixlib.defmatrix)  # module must match exactly

    def test_spectral(self):
        sf = SpectralFeatures(filtertype="bandpass", filterfreq=(0.5, 25))
        test_trace = test_signal(
            addchirp=False,
            frequencies=[5.0],
            amplitudes=[1.0],
            phases=[0.0],
            sinusoid=False,
            noise=False,
        )
        xds = sf.compute(test_trace)
        self.assertAlmostEqual(xds.predom_freq.data[0], 5, 0)
        self.assertAlmostEqual(xds.central_freq.data[0], 5, 0)
        self.assertAlmostEqual(xds.bandwidth.data[0], 0.062, 3)

    def test_spectral_with_gaps(self):
        sf = SpectralFeatures(filtertype="bandpass", filterfreq=(0.5, 25))
        test_trace = test_signal(
            addchirp=False,
            frequencies=[5.0],
            amplitudes=[1.0],
            phases=[0.0],
            sinusoid=False,
            noise=False,
            gaps=True,
        )
        xds = sf.compute(test_trace)
        self.assertAlmostEqual(xds.predom_freq.data[0], 5, 0)
        self.assertAlmostEqual(xds.central_freq.data[0], 5, 0)
        self.assertAlmostEqual(xds.bandwidth.data[0], 0.062, 3)
        self.assertEqual(np.isnan(xds.predom_freq.values).sum(), 3)

    @pytest.mark.slow
    def test_realtime(self):
        """
        Test with 1 hour windows as in the real-time computation
        """
        sf = SpectralFeatures(filtertype="highpass", filterfreq=(0.5, None))
        starttime = UTCDateTime(2020, 4, 21, 0, 0, 0)
        nsec = 86500  # Slightly longer than 24 h to avoid rounding issues
        step = 3600
        tr = test_signal(nsec=nsec, starttime=starttime, gaps=True)

        startwin = starttime
        sg = Storage("test", self.outdir)
        sg.starttime = starttime.datetime
        sg.endtime = (starttime + nsec).datetime
        st = sg.get_substore()
        while True:
            endwin = startwin + step
            if endwin > tr.stats.endtime:
                break
            ntr = tr.slice(startwin, endwin)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.save(sf.compute(ntr))
            startwin = endwin
        for feature in ["bandwidth", "central_freq", "predom_freq"]:
            self.assertEqual(st(feature).data.shape, (144,))
