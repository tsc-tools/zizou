from datetime import datetime, timedelta
import inspect
import os
from subprocess import Popen

import pytest
from tonik import StorageGroup
import yaml

from zizou.util import generate_test_data, xarray2hdf5

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
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
    sgw = StorageGroup('volcanoes', savedir)
    s_wiz = sgw.get_store('WIZ', '00', 'HHZ') 
    s_mdr = sgw.get_store('MDR', '00', 'BHZ') 
    s_mavz = sgw.get_store('MAVZ', '00', 'EHZ') 
    s_mms = sgw.get_store('MMS', '00', 'BHZ') 
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
    sg = StorageGroup('volcanoes', rootdir=savedir,
                      starttime=tstart, 
                      endtime=tstart + timedelta(days=ndays))
    s_wiz = sg.get_store('WIZ', '00', 'HHZ')
    config = """ 
    autoencoder:
      layers: [2000,500,200,6]
      epochs: 5
      patience: 10
    """
    return savedir, s_wiz, config