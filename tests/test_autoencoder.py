from configparser import ConfigParser
from datetime import datetime
import logging
import logging.config

import pytest

from zizou.autoencoder import AutoEncoder
from zizou import get_data
from zizou import zizou_logging


logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_fit(setup_ac):
    rootdir, store, config = setup_ac
    ac = AutoEncoder(config)
    ac.fit(store)
    assert float(ac.loss_) - 1 < 0.5


@pytest.mark.slow
def test_transform(setup_ac):
    savedir, store, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    store.starttime = starttime
    store.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(store, cluster=False)
    assert ed['autoencoder'].shape == (6, 145)
 

@pytest.mark.slow
def test_cluster(setup_ac):
    savedir, store, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    store.starttime = starttime
    store.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(store)
    assert ed['autoencoder'].shape == (5, 145)
 
@pytest.mark.slow
def test_early_stopping(setup_ac):
    savedir, store, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    store.starttime = starttime
    store.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(store, cluster=False)
    assert ed['autoencoder'].shape == (6, 145)
 