import shutil
import tempfile
import unittest

import numpy as np
import os

import pandas as pd
import xarray as xr

from zizou import FeatureBaseClass

dt1 = '2020-01-03 00:10:00'
dt2 = '2020-01-03 00:20:00'
dt3 = '2020-01-03 00:30:00'
dt4 = '2020-01-03 00:40:00'
dt5 = '2020-01-04 00:10:00'
dt6 = '2020-01-04 00:30:00'
dt7 = '2020-01-04 00:50:00'


class BaseClassTestCase(unittest.TestCase):

    def setUp(self):
        self.savedir = tempfile.mkdtemp() 

    def tearDown(self):
        if os.path.isdir(self.savedir):
            shutil.rmtree(self.savedir)

    def test_exceptions(self):
        """
        Test that the right exceptions are raised.
        """
        bc = FeatureBaseClass()
        with self.assertRaises(NotImplementedError):
            bc.compute()

        with self.assertRaises(TypeError):
            bc.save()

    def test_append(self):
        """
        Test appending existing HDF5 files.
        """
        featureName = '1DFeature'
        xdf1 = self.make1DDataSet(dt1, dt3, [1, 2], featureName)
        bc1 = FeatureBaseClass()
        bc1.feature = xdf1
        bc1.save(self.savedir) #here we overwrite the file first to make sure it's created from scratch

        testfile = os.path.join(self.savedir, featureName+'.nc')
        with xr.load_dataset(testfile, group='original') as xds1:
            self.assertEqual(np.datetime64(xds1.attrs['starttime']).astype('datetime64[ms]'),
                             np.datetime64(dt1).astype('datetime64[ms]'))
            self.assertEqual(np.datetime64(xds1.attrs['endtime']).astype('datetime64[ms]'),
                             np.datetime64(dt3).astype('datetime64[ms]'))

        xdf2 = self.make1DDataSet(dt3, dt4, [3, 4], featureName)
        bc2 = FeatureBaseClass()
        bc2.feature = xdf2
        bc2.save(self.savedir)

        with xr.load_dataset(testfile, group='original') as xds1:
            self.assertEqual(np.datetime64(xds1.attrs['starttime']).astype('datetime64[ms]'),
                             np.datetime64(dt1).astype('datetime64[ms]'))
            self.assertEqual(np.datetime64(xds1.attrs['endtime']).astype('datetime64[ms]'),
                             np.datetime64(dt4).astype('datetime64[ms]'))
            np.testing.assert_array_almost_equal(xds1['1DFeature'].values,
                                                 np.array([1., 2., 3., 4.]), 5)

    def test_2D_append(self):
        """
        Test appending when the feature is an xarray Dataset with 2D
        data.
        """
        xdf1, xdf2 = self.make2DDataSets()

        bc1 = FeatureBaseClass()
        bc1.feature = xdf1
        bc1.save(self.savedir)

        bc2 = FeatureBaseClass()
        bc2.feature = xdf2
        bc2.save(self.savedir)

        testfile = os.path.join(self.savedir, '2Dfeature.nc')
        with xr.open_dataset(testfile, group='original') as xds:
            self.assertEqual(np.datetime64(xds.attrs['starttime']).astype('datetime64[ms]'),
                             np.datetime64(dt5).astype('datetime64[ms]'))
            self.assertEqual(np.datetime64(xds.attrs['endtime']).astype('datetime64[ms]'),
                             np.datetime64(dt7).astype('datetime64[ms]'))

            np.testing.assert_array_almost_equal(xds['2Dfeature'].values,
                                                 np.array([[1., 2., 6., 7., 8.],
                                                           [4., 5., 9., 10., 11.]]))

    def make1DDataSet(self, dt1, dt2, values, featureName):
        t = pd.date_range(dt1, dt2,periods=2)
        xdf = xr.DataArray(values, coords=dict(datetime=t), dims=['datetime'])
        xdf = xr.Dataset({featureName: xdf})
        xdf.attrs['station'] = 'WIZ'
        return xdf

    def make2DDataSets(self):
        t1 = pd.date_range(dt5, dt6, freq='10min')
        values = np.arange(1, 7).reshape((2, 3))
        foo = xr.DataArray(values,
                           coords=[np.array([10, 20]), t1],
                           dims=['frequency', 'datetime'])
        xdf1 = xr.Dataset({'2Dfeature': foo})
        xdf1.attrs['starttime'] = t1[0].isoformat()
        xdf1.attrs['endtime'] = t1[-1].isoformat()
        xdf1.attrs['station'] = 'WIZ'

        # Note that t1 and t2 overlap so the final result should have the
        # first row of the second dataset overwritten by the last row of
        # the first dataset
        t2 = pd.date_range(dt6, dt7, freq='10min')
        values = np.arange(6, 12).reshape((2, 3))
        bar = xr.DataArray(values,
                           coords=[np.array([10, 20]), t2],
                           dims=['frequency', 'datetime'])

        xdf2 = xr.Dataset({'2Dfeature': bar})
        xdf2.attrs['starttime'] = t2[0].isoformat()
        xdf2.attrs['endtime'] = t2[-1].isoformat()
        xdf2.attrs['station'] = 'WIZ'

        return xdf1, xdf2


if __name__ == '__main__':
    unittest.main()
