import os
import tarfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr
from tonik import Storage
from tonik.utils import generate_test_data

from zizou.data import DataSource, SDSWaveforms


def pytest_addoption(parser):
    parser.addoption(
        "--runwebservice",
        action="store_true",
        default=False,
        help="run webservice tests",
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "webservice: mark tests that request data from webservices"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runwebservice"):
        skip_webservice = pytest.mark.skip(reason="need --runwebservice option to run")
        for item in items:
            if "webservice" in item.keywords:
                item.add_marker(skip_webservice)
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


tstart = datetime(2023, 1, 1, 0, 0, 0)
ndays = 10


@pytest.fixture(scope="package")
def setup(tmp_path_factory):
    features1D = [
        "rsam",
        "dsar",
        "central_freq",
        "predom_freq",
        "bandwidth",
        "rsam_energy_prop",
    ]
    features2D = ["sonogram", "ssam", "filterbank"]
    features2D_freqs = ["sonofrequency", "frequency", "fbfrequency"]

    savedir = tmp_path_factory.mktemp("zizou_test_tmp", numbered=True)
    sgw = Storage("volcanoes", savedir)
    s_wiz = sgw.get_substore("WIZ", "00", "HHZ")
    s_wiz.station = "NZ.WIZ.00.HHZ"
    s_mdr = sgw.get_substore("MDR", "00", "BHZ")
    s_mdr.station = "NZ.MDR.00.BHZ"
    s_mavz = sgw.get_substore("MAVZ", "00", "EHZ")
    s_mavz.station = "NZ.MAVZ.00.EHZ"
    s_mms = sgw.get_substore("MMS", "00", "BHZ")
    s_mms.station = "NZ.MMS.00.BHZ"
    # Generate some fake data
    feat = generate_test_data(tstart=tstart, feature_names=features1D, ndays=ndays)
    for _s in [s_wiz, s_mdr, s_mavz, s_mms]:
        feat["datetime"] = feat.datetime.dt.round("10min")
        _s.save(feat)
    feat = generate_test_data(
        tstart=tstart,
        feature_names=features2D,
        freq_names=features2D_freqs,
        ndays=ndays,
        nfreqs=8,
        dim=2,
    )
    for _s in [s_wiz, s_mdr, s_mavz, s_mms]:
        feat["datetime"] = feat.datetime.dt.round("10min")
        _s.save(feat)

    alg = generate_test_data(
        tstart=tstart,
        feature_names=["autoencoder"],
        ndays=ndays,
        nfreqs=5,
        freq_names=["cluster"],
        dim=2,
    )
    s_mdr.save(alg)
    return savedir


@pytest.fixture(scope="module")
def setup_dataset(tmp_path_factory):
    npts = 3
    npts2 = 4
    vals = np.array(
        [
            [1.0, np.nan, np.nan, 3.0],
            [4.0, 5.0, np.nan, 7.0],
            [np.nan, 9.0, 10.0, 11.0],
        ]
    )
    xds_2D = xr.Dataset(
        {"ssam": (["frequency", "datetime"], vals)},
        coords={
            "frequency": np.arange(npts),
            "datetime": pd.date_range(start=tstart, periods=npts2, freq="10min"),
        },
    )

    xds_1D = xr.Dataset(
        {"rsam": (["datetime"], vals[1, :])},
        coords={
            "datetime": pd.date_range(start=tstart, periods=npts2, freq="10min"),
        },
    )
    xds_1D.attrs["resolution"] = 1 / 6.0
    xds_2D.attrs["resolution"] = 1 / 6.0

    savedir = tmp_path_factory.mktemp("zizou_dataset_test_tmp", numbered=True)
    sgw = Storage(
        "volcanoes",
        savedir,
        starttime=tstart,
        endtime=tstart + timedelta(minutes=10 * npts2),
    )
    s_wiz = sgw.get_substore("WIZ", "00", "HHZ")
    # Generate some fake data
    s_wiz.save(xds_1D, archive_starttime=tstart)
    s_wiz.save(xds_2D, archive_starttime=tstart)
    return s_wiz


@pytest.fixture(scope="module")
def setup_ac(setup):
    savedir = setup
    sg = Storage(
        "volcanoes",
        rootdir=savedir,
        starttime=tstart,
        endtime=tstart + timedelta(days=ndays),
    )
    s_wiz = sg.get_substore("WIZ", "00", "HHZ")
    s_wiz.station = "NZ.WIZ.00.HHZ"
    s_wsrz = sg.get_substore("WSRZ", "00", "HHZ")
    s_wsrz.station = "NZ.WSRZ.00.HHZ"
    config = """ 
    autoencoder:
      layers: [2000,500,200,6]
      epochs: 5
      patience: 10
    """
    return savedir, s_wiz, s_wsrz, config


def download_and_extract_tar(url, extract_path):
    # Download the tar file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("file.tar", "wb") as f:
            f.write(response.raw.read())

        # Open the tar file
        tar = tarfile.open("file.tar")

        # Extract it
        tar.extractall(path=extract_path)
        tar.close()
    else:
        print(f"Failed to download {url}")


@pytest.fixture(scope="module")
def setup_sds():
    test_data_dir = os.path.join(os.environ["HOME"], "zizou_test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(test_data_dir, "sds_test")):
        url = "https://zenodo.org/records/13377159/files/zizou_test_data.tar.gz"
        download_and_extract_tar(url, test_data_dir)
    sds_dir = os.path.join(test_data_dir, "sds_test")
    fdsn_urls = ("https://service.geonet.org.nz", "https://service-nrt.geonet.org.nz")
    sdsc = SDSWaveforms(
        sds_dir=sds_dir, fdsn_urls=fdsn_urls, staxml_dir=sds_dir, fill_value=np.nan
    )
    ds = DataSource(clients=[sdsc])
    return sds_dir, ds
