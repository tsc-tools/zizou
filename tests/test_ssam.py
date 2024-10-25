import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import pytest
from obspy import Trace, UTCDateTime
from scipy.signal import find_peaks
from tonik import Storage

from zizou.ssam import SSAM
from zizou.util import test_signal


@pytest.fixture(scope="module")
def setup_spectrogram():
    trace_ssam = test_signal(
        sinusoid=False, addchirp=False, frequencies=[1.0, 3.0, 10.0]
    )
    trace_ssam_gaps = test_signal(gaps=True)
    trace_sono = test_signal(
        offsets=[1000.0, 2000.0, 3000.0],
        frequencies=[1.0, 2.0, 5.0],
        amplitudes=[0.1, 1.0, 5.0],
        sinusoid=False,
        addchirp=False,
    )

    return trace_ssam, trace_sono, trace_ssam_gaps


def test_ssam(setup_spectrogram):
    trace, _, _ = setup_spectrogram
    test_peaks = np.array([1.0, 3.0, 10.0])
    s = SSAM(interval=60, per_lap=0.9, smooth=10)
    spec = s.compute(trace.copy())
    peaks = find_peaks(spec["ssam"].data[:, 0], height=0.1)[0]
    np.testing.assert_array_almost_equal(spec["frequency"].values[peaks], test_peaks, 2)


def test_sonogram(setup_spectrogram):
    _, trace, _ = setup_spectrogram
    s = SSAM(interval=60, per_lap=0.9, smooth=10)
    spec = s.compute(trace.copy())
    idx1 = int(
        np.abs(
            spec.datetime - np.datetime64((trace.stats.starttime + 1500.0).datetime)
        ).argmin()
    )
    idx2 = int(
        np.abs(
            spec.datetime - np.datetime64((trace.stats.starttime + 2600.0).datetime)
        ).argmin()
    )
    idx3 = int(
        np.abs(
            spec.datetime - np.datetime64((trace.stats.starttime + 3500.0).datetime)
        ).argmin()
    )
    assert (
        abs(
            float(
                spec.sonofrequency[spec["sonogram"].isel(dict(datetime=idx1)).argmax()]
            )
            - 2.295
        )
        < 1e-3
    )
    assert (
        abs(
            float(
                spec.sonofrequency[spec["sonogram"].isel(dict(datetime=idx2)).argmax()]
            )
            - 5.164
        )
        < 1e-3
    )
    assert (
        abs(
            float(
                spec.sonofrequency[spec["sonogram"].isel(dict(datetime=idx3)).argmax()]
            )
            - 11.62
        )
        < 1e-2
    )


def test_filterbank(setup_spectrogram):
    _, trace, _ = setup_spectrogram
    s = SSAM(interval=60, per_lap=0.9, smooth=10)
    spec = s.compute(trace.copy())
    idx1 = int(
        np.abs(
            spec.datetime - np.datetime64((trace.stats.starttime + 1500.0).datetime)
        ).argmin()
    )
    idx2 = int(
        np.abs(
            spec.datetime - np.datetime64((trace.stats.starttime + 2600.0).datetime)
        ).argmin()
    )
    idx3 = int(
        np.abs(
            spec.datetime - np.datetime64((trace.stats.starttime + 3500.0).datetime)
        ).argmin()
    )

    assert (
        abs(
            float(
                spec.fbfrequency[spec["filterbank"].isel(dict(datetime=idx1)).argmax()]
            )
            - 31.929
        )
        < 1e-3
    )
    assert (
        abs(
            float(
                spec.fbfrequency[spec["filterbank"].isel(dict(datetime=idx2)).argmax()]
            )
            - 2.165
        )
        < 1e-3
    )
    assert (
        abs(
            float(
                spec.fbfrequency[spec["filterbank"].isel(dict(datetime=idx3)).argmax()]
            )
            - 5.309
        )
        < 1e-3
    )


def test_ssam_with_gaps(setup_spectrogram):
    _, _, trace = setup_spectrogram
    linfreqs = np.arange(0, 25.1, 0.1)
    test_peaks = np.array([0.1, 3.0, 5, 6.1, 10.0])
    s = SSAM(interval=60, per_lap=0.9, smooth=10, frequencies=linfreqs)
    spec = s.compute(trace.copy())
    peaks = find_peaks(spec["ssam"].data[:, 0], height=1e-1)[0]
    np.testing.assert_array_almost_equal(linfreqs[peaks], test_peaks, 2)


@pytest.mark.slow
def test_realtime(tmp_path_factory):
    """
    Test with 1 hour windows as in the real-time computation
    """
    savedir = tmp_path_factory.mktemp("realtime")
    linfreqs = np.arange(0, 25.1, 0.1)
    testpeaks = np.array([0.1, 3.0, 5.0, 6.0, 10.0])
    s = SSAM(
        interval=60,
        frequencies=linfreqs,
        resample_int=(None, "10min"),
        timestamp="start",
    )
    starttime = UTCDateTime(2020, 4, 21, 0, 0, 0)
    nsec = 86500  # Slightly longer than 24 h to avoid rounding issues
    step = 3600
    tr = test_signal(nsec=nsec, starttime=starttime, gaps=True)
    sg = Storage(
        "test",
        savedir,
        starttime=starttime.datetime,
        endtime=(starttime + nsec).datetime,
    )
    store = sg.get_substore("WIZ", "00", "HHZ")
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

    ssam = store("ssam")
    assert ssam.shape == (251, 144)
    peaks = find_peaks(ssam.data[:, 0], height=1e-1)[0]
    np.testing.assert_array_almost_equal(linfreqs[peaks], testpeaks, 2)


def test_config():
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
    assert s.resample_int[0] is None
    assert s.resample_int[1] == "10min"
    np.testing.assert_equal(s.frequencies, np.arange(0, 25.1, 0.1))
