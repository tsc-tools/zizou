from datetime import datetime, timedelta
import shutil
import tempfile
import unittest

import os

import xarray as xr

from zizou import AnomalyDetectionBaseClass
from zizou.util import generate_test_data, xarray2hdf5
from zizou.data import FeatureRequest


class AnomalyBaseClassTestCase(unittest.TestCase):

    def setUp(self):
        self.savedir = tempfile.mkdtemp()
        self.rootdir = os.path.join(self.savedir,
                                    'Mt_Doom',
                                    'MDR',
                                    'HHZ')
        os.makedirs(self.rootdir)
        tstart = datetime(2023,1,1,0,0,0)
        self.fq = FeatureRequest(rootdir=self.savedir,
                                 volcano='Mt_Doom',
                                 site='MDR',
                                 channel='HHZ',
                                 starttime=tstart,
                                 endtime=tstart + timedelta(days=30))
        # Generate some fake data
        rsam = generate_test_data(tstart=tstart,
                                  feature_name='rsam',
                                  ndays=30)
        xarray2hdf5(rsam, self.rootdir)
        dsar = generate_test_data(tstart=tstart,
                                  feature_name='dsar',
                                  ndays=30)
        xarray2hdf5(dsar, self.rootdir)
        sonogram = generate_test_data(tstart=tstart, feature_name='sonogram',
                                      freq_name='sonofrequency', nfreqs=8,
                                      dim=2,
                                      ndays=30)
        xarray2hdf5(sonogram, self.rootdir)
        
    def tearDown(self):
        if os.path.isdir(self.savedir):
            shutil.rmtree(self.savedir)

    def test_get_features(self):
        transform_dict = {'rsam': 'log'}
        stack_dict = {'dsar': '2D',
                      'central_freq': '1H',
                      'predom_freq': '1H',
                      'variance': '1H',
                      'bandwidth': '1H'}
        ab = AnomalyDetectionBaseClass(['rsam', 'sonogram', 'dsar'], stacks=stack_dict,
                                       transforms=transform_dict)
        featurefile = os.path.join(self.savedir, 'pca_features.nc')
        feats = ab.get_features(self.fq, featurefile)
        self.assertEqual(feats.shape, (30*144, 10))
        # Run again to test reading features from file
        feats1 = ab.get_features(self.fq, featurefile)
        xr.testing.assert_allclose(feats, feats1)


if __name__ == '__main__':
    unittest.main()

        

