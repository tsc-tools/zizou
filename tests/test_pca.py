import datetime
import inspect
import tempfile
import unittest
from unittest import TestCase

import numpy as np
import xarray as xr
import os
import pandas as pd

from obspy import UTCDateTime
from zizou import get_data
from zizou.pca import PCA, runPca, runPcaCore


@unittest.skip("Tested code needs re-write")
class PrincipalComponentTestCase(unittest.TestCase):

    def setUp(self):
        filename = inspect.getfile(inspect.currentframe())
        filedir = os.path.dirname(os.path.abspath(filename))
        self.testdata_dir = os.path.join(filedir, "data")

        class DummyFQTrain:
            def __init__(self):
                rng = np.random.RandomState(42)
                self.x, self.y = rng.multivariate_normal([10, 3],
                                                         cov=[[2, 1], [1, 2]],
                                                         size=10000).T
                self.times = pd.date_range('2021-05-25', freq='10min', periods=10000)

            def __call__(self, feature, stack_length=None):
                if feature == 'rsam':
                    return xr.DataArray(data=self.x,
                                        coords=[self.times],
                                        dims=['datetime'],
                                        name='rsam',
                                        )
                elif feature == 'dsar':
                    return xr.DataArray(data=self.x,
                                        coords=[self.times],
                                        dims=['datetime'],
                                        name='dsar')

        self.dfq_train = DummyFQTrain()

        class DummyFQTransform:
            def __init__(self):
                self.x = np.array([1., np.nan, 3., np.nan])
                self.y = np.array([1., np.nan, 3.])

                self.starttime = datetime.datetime(2020, 1, 1, 0, 0, 0)
                self.nantime = datetime.datetime(2020, 1, 1, 0, 10, 0)
                self.endtime = datetime.datetime(2020, 1, 1, 0, 20, 0)
                self.extraTime = datetime.datetime(2020, 1, 1, 0, 30, 0)
                self.times = [self.starttime, self.nantime, self.endtime]
                self.site = 'RIZ'

            def __call__(self, feature, stack_length=None):
                if feature == 'rsam':
                    return xr.DataArray(data=self.x,
                                        coords=[self.times+[self.extraTime]],
                                        dims=['datetime'],
                                        name="x")
                elif feature == 'dsar':
                    return xr.DataArray(data=self.y,
                                        coords=[self.times],
                                        dims=['datetime'],
                                        name='y')

        self.dfq_transform = DummyFQTransform()
        self.testfile = tempfile.mktemp()

    def tearDown(self):
        try:
            os.unlink(self.testfile)
        except FileNotFoundError:
            pass

    def test_normalise_and_fill(self):
        print("test_normalise_and_fill")
        pca = PCA(self.testfile,
                  n_components=2,
                  pca_features=['rsam', 'dsar'])

        feat1 = self.dfq_train(pca.pca_features[0]) #stack_dict[pca.pca_features[0]])

        mn, norm_feat_fill_nan, std = pca.normaliseAndFill(feat1)
        self.assertEqual(mn, 9.9900799153757021)
        self.assertEqual(std, 1.4216981693126631)
        array_sum = np.sum(norm_feat_fill_nan)
        array_has_nan = np.isnan(array_sum)
        self.assertFalse(array_has_nan)

        feat2 = self.dfq_transform(pca.pca_features[0], stack_dict[pca.pca_features[0]])
        mn, norm_feat_fill_nan, std = pca.normaliseAndFill(feat2)
        self.assertEqual(mn, 2.0)
        self.assertEqual(std, 1.0)

    def test_pca_train(self):
        print("test_pca_train")
        pca = PCA(self.testfile, n_components=2,
                  pca_features=['rsam', 'dsar'])
        pca.train(self.dfq_train)
        tval = 1/np.sqrt(2)
        np.testing.assert_array_almost_equal(np.abs(pca.pca_model.components_),
                                             np.array([[tval, tval],
                                                       [tval, tval]]), 8)
        self.assertEqual(np.abs(np.sign(pca.pca_model.components_[0]).sum()),
                         2)
        self.assertEqual(np.abs(np.sign(pca.pca_model.components_[1]).sum()),
                         0)

    def test_pca_transform(self):
        print("test_pca_transform")
        pca = PCA(self.testfile, n_components=2,
                  pca_features=['rsam', 'dsar'])
        pca.train(self.dfq_train)
        xdf = pca.infer(self.dfq_transform)
        self.assertTrue(np.all(np.isclose(xdf.pca.loc[:, 'pc1'].values, 0)))

        # test caching
        pca1 = PCA(self.testfile, n_components=2,
                   pca_features=['rsam', 'dsar'])
        pca1.train('dummy')
        xdf1 = pca1.infer(self.dfq_transform)
        self.assertTrue(np.all(np.isclose(xdf1.pca.loc[:, 'pc1'].values, 0)))
        tval = np.sqrt(2)
        np.testing.assert_array_almost_equal(xdf1.pca.loc[:, 'pc0'].values,
                                             np.array([tval, 0, -tval]), 8)

    # def test_pcaRun(self):
    #
    #     runPca()

    # def test_runPcaCore(self):
    #     print("test_runPcaCore")
    #     start = UTCDateTime(2021, 3, 13, 6, 0, 0)
    #     end = round_times(start, 600)
    #     start = end-7200
    #
    #     config_file = os.path.join(self.testdata_dir, "config_tests.ini")
    #     meta_file = get_data('package_data/metadata.json')
    #
    #     outDir = tempfile.gettempdir()
    #     runPcaCore(config_file=config_file, metadata_file=meta_file,
    #                end_time=end.datetime, start_time=start.datetime,
    #                out_dir=outDir)


if __name__ == '__main__':
    unittest.main()
