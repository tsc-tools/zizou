import tarfile
from datetime import datetime, timedelta
import os

import numpy as np
import pytest
import requests
from tonik import Storage
from zizou.data import SDSWaveforms, DataSource
import yaml

from zizou.util import generate_test_data

def pytest_addoption(parser):
    parser.addoption(
        "--runwebservice", action="store_true", default=False, help="run webservice tests"
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "webservice: mark tests that request data from webservices")


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
   

tstart = datetime(2023,1,1,0,0,0)
ndays = 10

@pytest.fixture(scope='package')
def setup(tmp_path_factory):
    features1D = ['rsam',
                  'dsar',
                  'central_freq',
                  'predom_freq',
                  'bandwidth',
                  'rsam_energy_prop']
    features2D = [('sonogram', 'sonofrequency'),
                  ('ssam', 'frequency'),
                  ('filterbank', 'fbfrequency')]

    savedir = tmp_path_factory.mktemp('zizou_test_tmp', numbered=True)
    sgw = Storage('volcanoes', savedir)
    s_wiz = sgw.get_substore('WIZ', '00', 'HHZ') 
    s_mdr = sgw.get_substore('MDR', '00', 'BHZ') 
    s_mavz = sgw.get_substore('MAVZ', '00', 'EHZ') 
    s_mms = sgw.get_substore('MMS', '00', 'BHZ') 
   # Generate some fake data
    for _f in features1D:
        feat = generate_test_data(tstart=tstart,
                                    feature_name=_f,
                                    ndays=ndays)
        for _s in [s_wiz, s_mdr, s_mavz, s_mms]:
            _s.save(feat)
    for _n, _f in features2D:
        feat = generate_test_data(tstart=tstart,
                                  feature_name=_n,
                                  ndays=ndays,
                                  nfreqs=8,
                                  freq_name=_f,
                                  dim=2)
        for _s in [s_wiz, s_mdr, s_mavz, s_mms]:
            _s.save(feat)

    alg = generate_test_data(tstart=tstart,
                             feature_name='autoencoder',
                             ndays=ndays,
                             nfreqs=5,
                             freq_name='cluster',
                             dim=2)
    s_mdr.save(alg)
    return savedir

@pytest.fixture(scope='module')
def setup_ac(setup):
    savedir = setup
    sg = Storage('volcanoes', rootdir=savedir,
                      starttime=tstart, 
                      endtime=tstart + timedelta(days=ndays))
    s_wiz = sg.get_substore('WIZ', '00', 'HHZ')
    s_wsrz = sg.get_substore('WSRZ', '00', 'HHZ')
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
        with open('file.tar', 'wb') as f:
            f.write(response.raw.read())

        # Open the tar file
        tar = tarfile.open('file.tar')

        # Extract it
        tar.extractall(path=extract_path)
        tar.close()
    else:
        print(f"Failed to download {url}")


@pytest.fixture(scope='module')
def setup_sds():
    test_data_dir = os.path.join(os.environ['HOME'], 'zizou_test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(test_data_dir, 'sds_test')):
        url = "https://zenodo.org/records/13377159/files/zizou_test_data.tar.gz"
        download_and_extract_tar(url, test_data_dir)
    sds_dir = os.path.join(test_data_dir, "sds_test")
    fdsn_urls=('https://service.geonet.org.nz',
                'https://service-nrt.geonet.org.nz')
    sdsc = SDSWaveforms(sds_dir=sds_dir, fdsn_urls=fdsn_urls,
                        staxml_dir=sds_dir, fill_value=np.nan)
    ds = DataSource(clients=[sdsc])
    return sds_dir, ds 

