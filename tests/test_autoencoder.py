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
    rootdir, store1, store2, config = setup_ac
    ac = AutoEncoder(config)
    ac.fit(store1)
    assert float(ac.loss_) - 1 < 0.5


@pytest.mark.slow
def test_transform(setup_ac):
    savedir, store1, store2, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    store1.starttime = starttime
    store1.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(store1, cluster=False)
    store1.save(ed)
    assert ed['autoencoder'].shape == (6, 145)
 

@pytest.mark.slow
def test_cluster(setup_ac):
    savedir, store1, store2, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    store1.starttime = starttime
    store1.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(store1)
    store1.save(ed)
    assert ed['autoencoder_cluster'].shape == (5, 145)
 
@pytest.mark.slow
def test_early_stopping(setup_ac):
    savedir, store1, store2, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    store1.starttime = starttime
    store1.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(store1, cluster=False)
    assert ed['autoencoder'].shape == (6, 145)


def test_exceptions(setup_ac):
    rootdir, store1, store2, config = setup_ac
    ac = AutoEncoder(config)
    with pytest.raises(AssertionError):
        ac.transform(store2)


 