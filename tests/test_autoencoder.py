import logging
import logging.config
from configparser import ConfigParser
from datetime import datetime

import pytest

from zizou import get_data, zizou_logging
from zizou.autoencoder import AutoEncoder

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_fit(setup_ac):
    rootdir, store1, store2, config = setup_ac
    ac = AutoEncoder(store1, configfile=config, batch_size=64)
    ac.clear()
    ac.fit()
    assert float(ac.loss_) - 1 < 0.5


@pytest.mark.slow
def test_transform(setup_ac):
    savedir, store1, store2, config = setup_ac
    starttime = datetime(2023, 1, 8, 0, 0, 0)
    endtime = datetime(2023, 1, 9, 0, 0, 0)
    store1.starttime = starttime
    store1.endtime = endtime
    ac = AutoEncoder(store1, configfile=config, batch_size=64)
    ac.clear()
    ac.fit_transform(starttime, endtime)
    store1.starttime = starttime
    store1.endtime = endtime
    ae = store1("autoencoder_embedding")
    acl = store1("autoencoder_cluster")
    acloss = store1("autoencoder_loss")
    assert ae.shape == (6, 144)
    assert acl.shape == (5, 144)
    assert acloss.shape == (144,)


@pytest.mark.slow
def test_early_stopping(setup_ac):
    savedir, store1, store2, config = setup_ac
    starttime = datetime(2023, 1, 8, 0, 0, 0)
    endtime = datetime(2023, 1, 9, 0, 0, 0)
    store1.starttime = starttime
    store1.endtime = endtime
    ac = AutoEncoder(store1, configfile=config, batch_size=64)
    ac.clear()
    ac.fit_transform(starttime, endtime)
    store1.starttime = starttime
    store1.endtime = endtime
    ae = store1("autoencoder_embedding")
    acl = store1("autoencoder_cluster")
    acloss = store1("autoencoder_loss")
    assert ae.shape == (6, 144)
    assert acl.shape == (5, 144)
    assert acloss.shape == (144,)


@pytest.mark.slow
def test_spectrogram(setup_ac):
    rootdir, store1, store2, config = setup_ac
    starttime = datetime(2023, 1, 8, 0, 0, 0)
    endtime = datetime(2023, 1, 9, 0, 0, 0)
    ac = AutoEncoder(store1, configfile=config, features=["filterbank"], batch_size=64)
    ac.clear()
    ac.fit_transform(starttime, endtime)
    assert float(ac.loss_) - 1 < 0.5
