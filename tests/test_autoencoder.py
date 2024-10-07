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
    ac = AutoEncoder(configfile=config)
    ac.fit(store1)
    assert float(ac.loss_) - 1 < 0.5


# @pytest.mark.slow
def test_transform(setup_ac):
    savedir, store1, store2, config = setup_ac
    starttime = datetime(2023, 1, 8, 0, 0, 0)
    endtime = datetime(2023, 1, 9, 0, 0, 0)
    store1.starttime = starttime
    store1.endtime = endtime
    ac = AutoEncoder(configfile=config)
    ed = ac.fit_transform(store1)
    store1.save(ed, mode="w")
    assert ed["autoencoder_embedding"].shape == (6, 145)
    assert ed["autoencoder_cluster"].shape == (5, 145)
    assert ed["autoencoder_loss"].shape == (145,)


@pytest.mark.slow
def test_early_stopping(setup_ac):
    savedir, store1, store2, config = setup_ac
    starttime = datetime(2023, 1, 8, 0, 0, 0)
    endtime = datetime(2023, 1, 9, 0, 0, 0)
    store1.starttime = starttime
    store1.endtime = endtime
    ac = AutoEncoder(configfile=config)
    ed = ac.fit_transform(store1)
    assert ed["autoencoder_embedding"].shape == (6, 145)
    assert ed["autoencoder_cluster"].shape == (5, 145)
    assert ed["autoencoder_loss"].shape == (145,)


def test_exceptions(setup_ac):
    rootdir, store1, store2, config = setup_ac
    ac = AutoEncoder(configfile=config)
    with pytest.raises(AssertionError):
        ac.transform(store2)
