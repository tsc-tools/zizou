from datetime import datetime, timedelta
import os
import shutil
import tempfile
import unittest

from tonik import Storage
import xarray as xr

from zizou import AnomalyDetectionBaseClass
from zizou.util import generate_test_data


def test_get_features(setup_ac):
    savedir, sg, _, config = setup_ac
    transform_dict = {'rsam': 'log'}
    stack_dict = {'dsar': '2D',
                  'central_freq': '1H',
                  'predom_freq': '1H',
                  'variance': '1H',
                  'bandwidth': '1H'}
    ab = AnomalyDetectionBaseClass(['rsam', 'sonogram', 'dsar'], stacks=stack_dict,
                                    transforms=transform_dict)
    featurefile = os.path.join(savedir, 'pca_features.nc')
    feats = ab.get_features(sg, featurefile)
    assert feats.shape == (10*144, 10)
    # Run again to test reading features from file
    feats1 = ab.get_features(sg, featurefile)
    xr.testing.assert_allclose(feats, feats1)
