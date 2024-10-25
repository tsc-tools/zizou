import os

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader, random_split

from zizou import (
    AnomalyDetectionBaseClass,
    SliceBatchSampler,
    ZizouDataset,
    min_max_scale_netcdf,
)


def test_dataset(setup_dataset):
    sg = setup_dataset
    zs = ZizouDataset(sg, ["rsam", "ssam"])
    shapes = zs.get_shapes()
    assert shapes[1] == 4
    assert zs.resolution == 1 / 6.0
    assert zs[:][0].shape == (4, 4)
    np.testing.assert_array_equal(zs[0:2][0], zs[[0, 1]][0])
    np.testing.assert_array_equal(zs[(1, 3)][0], zs[np.array([1, 3])][0])
    assert zs.get_dates()[0] == pd.Timestamp("2023-01-01 00:30:00")


def test_dataloader(setup_dataset):
    sg = setup_dataset
    zd = ZizouDataset(sg, ["rsam", "ssam"])
    train_size = 3
    test_size = 1
    train_dataset, test_dataset = random_split(zd, [train_size, test_size])
    assert len(train_dataset) == 3
    assert len(test_dataset) == 1
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for batch, (X, y) in enumerate(train_loader):
        if batch == 0:
            assert X.shape == (2, 4)
        elif batch == 1:
            assert X.shape == (4,)


def test_SliceBatchSampler():
    slices = list(SliceBatchSampler(range(10), 4, False))
    assert slices == [slice(0, 4, None), slice(4, 8, None), slice(8, 10, None)]
    slices = list(SliceBatchSampler(range(5, 10), 4, False))
    assert slices == [slice(5, 9, None), slice(9, 10, None)]


def test_min_max_scaling(setup_dataset):
    sg = setup_dataset
    for feature in ["rsam"]:
        fin = sg.feature_path(feature)
        fout = os.path.join(sg.path, f"{feature}_scaling_params.nc")
        scaling_params = min_max_scale_netcdf(fin, feature, fout)
        assert scaling_params["min"] == 4.0
        assert scaling_params["max"] == 7.0
        assert scaling_params["median"] == 5.0

    for feature in ["ssam"]:
        fin = sg.feature_path(feature)
        fout = os.path.join(sg.path, f"{feature}_scaling_params.nc")
        scaling_params = min_max_scale_netcdf(fin, feature, fout)
        assert scaling_params["min"] == [1.0, 4.0, 9.0]
        assert scaling_params["max"] == [3.0, 7.0, 11.0]
        assert scaling_params["median"] == [2.0, 5.0, 10.0]


def test_scaling(setup_dataset):
    sg = setup_dataset
    zs = ZizouDataset(sg, ["rsam", "ssam"])
    xda_1D = zs.load_and_scale_feature("rsam")
    np.testing.assert_array_equal(xda_1D.values, np.array([0.0, 1 / 3.0, 1 / 3.0, 1.0]))
    xda_2D = zs.load_and_scale_feature("ssam")
    np.testing.assert_array_equal(
        xda_2D.values,
        np.array(
            [
                [0.0, 1 / 2.0, 1 / 2.0, 1.0],
                [0.0, 1 / 3.0, 1 / 3.0, 1.0],
                [1 / 2.0, 0.0, 1 / 2.0, 1.0],
            ]
        ),
    )
