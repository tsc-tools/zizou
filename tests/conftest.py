from datetime import datetime, timedelta
import inspect
import os
from subprocess import Popen

import pytest
import yaml

from zizou.data import FeatureRequest
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
    widir = os.path.join(savedir,
                         'features',
                         'Whakaari',
                         'WIZ',
                         'HHZ')
    doomdir = os.path.join(savedir,
                           'features',
                           'Mt Doom',
                           'MDR',
                           'BHZ')
    ruadir = os.path.join(savedir,
                         'features',
                         'Ruapehu',
                         'MAVZ',
                         'EHZ')
    mmdir = os.path.join(savedir,
                         'features',
                         'Misty Mountain',
                         'MMS',
                         'BHZ')
    for _dir in [widir, doomdir, ruadir, mmdir]:
        os.makedirs(_dir)
   # Generate some fake data
    for _f in features1D:
        feat = generate_test_data(tstart=tstart,
                                    feature_name=_f,
                                    ndays=ndays)
        for _dir in [widir, doomdir, ruadir]:
            xarray2hdf5(feat, _dir)
    for _n, _f in features2D:
        feat = generate_test_data(tstart=tstart,
                                  feature_name=_n,
                                  ndays=ndays,
                                  nfreqs=8,
                                  freq_name=_f,
                                  dim=2)
        for _dir in [widir, doomdir, ruadir]:
            xarray2hdf5(feat, _dir)

    alg = generate_test_data(tstart=tstart,
                             feature_name='autoencoder',
                             ndays=ndays,
                             nfreqs=5,
                             freq_name='cluster',
                             dim=2)
    xarray2hdf5(alg, doomdir)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(
                            inspect.getfile(inspect.currentframe()))), "data")
    with open(os.path.join(data_dir, 'config_integration_test.yml')) as fh:
        cfg = yaml.safe_load(fh)
    cfg['default'] = {'outdir': os.path.join(savedir, 'features')}
    fn_cfg = os.path.join(savedir, 'tmp.yml')
    with open(fn_cfg, 'w') as configfile:
        yaml.dump(cfg, configfile)
    return savedir

@pytest.fixture(scope='module')
def setup_ac(setup):
    savedir = setup
    fq = FeatureRequest(rootdir=os.path.join(savedir, 'features'),
                        volcano='Whakaari',
                        site='WIZ',
                        channel='HHZ',
                        starttime=tstart,
                        endtime=tstart + timedelta(days=ndays))
    config = """ 
    autoencoder:
      layers: [2000,500,200,6]
      epochs: 5
      patience: 10
    """
    return savedir, fq, config


@pytest.fixture(scope='module')
def setup_orchestrate(tmp_path_factory):
    savedir = tmp_path_factory.mktemp('zizou_test_tmp', numbered=True)
    test_dir = os.path.dirname(os.path.abspath(
                            inspect.getfile(inspect.currentframe())))
    data_dir = os.path.join(test_dir, "data")
    with open(os.path.join(data_dir, 'config_integration_test.yml')) as fh:
        cfg = yaml.safe_load(fh)
    cfg['default']['sds_dir'] =  os.path.join(savedir, 'sds')
    feature_dir = os.path.join(savedir, 'features')
    os.makedirs(feature_dir, exist_ok=True)
    cfg['default']['staxml_dir'] = data_dir
    cfg['default']['outdir'] = feature_dir
    cfg['default']['logdir'] = str(savedir)
    server = Popen(["random_walk_model"])
    fn_cfg = os.path.join(savedir, 'tmp.yml')
    with open(fn_cfg, 'w') as configfile:
        yaml.dump(cfg, configfile)
    os.environ['SAMCONFIG'] = fn_cfg
    yield feature_dir
    server.terminate()
 