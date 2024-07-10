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
    rootdir, fq, config = setup_ac
    ac = AutoEncoder(config)
    ac.fit(fq)
    assert float(ac.loss_) - 1 < 0.5


@pytest.mark.slow
def test_transform(setup_ac):
    savedir, fq, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    fq.starttime = starttime
    fq.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(fq, cluster=False)
    assert ed['autoencoder'].shape == (6, 144)
 

@pytest.mark.slow
def test_cluster(setup_ac):
    savedir, fq, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    fq.starttime = starttime
    fq.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(fq)
    assert ed['autoencoder'].shape == (5, 144)
 
@pytest.mark.slow
def test_early_stopping(setup_ac):
    savedir, fq, config = setup_ac
    starttime = datetime(2023,1,8,0,0,0)
    endtime = datetime(2023,1,9,0,0,0)
    fq.starttime = starttime
    fq.endtime = endtime
    ac = AutoEncoder(config)
    ed = ac.fit_transform(fq, cluster=False)
    assert ed['autoencoder'].shape == (6, 144)
    ac.save(savedir)
 