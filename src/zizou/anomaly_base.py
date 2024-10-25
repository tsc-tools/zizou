import copy
import datetime
import json
import logging
import os

import h5netcdf
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import BatchSampler, Dataset

logger = logging.getLogger(__name__)


def min_max_scale_netcdf(
    input_file, variable_name, scaling_params_file, chunk_size=144
):
    if os.path.isfile(scaling_params_file):
        with open(scaling_params_file, "r") as f:
            scaling_params = json.load(f)
        return scaling_params

    with xr.open_dataset(input_file, group="original") as ds:
        data = ds[variable_name]
        if len(data.sizes) > 1:
            min_value = list(data.min(axis=1).values)
            max_value = list(data.max(axis=1).values)
            median_value = list(data.median(axis=1).values)
        else:
            min_value = float(data.min())
            max_value = float(data.max())
            median_value = float(data.median())
    scaling_params = {"min": min_value, "max": max_value, "median": median_value}
    with open(scaling_params_file, "w") as f:
        json.dump(scaling_params, f)

    return scaling_params


class SliceBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # Generate all indices from the sampler
        indices = list(self.sampler)

        # Yield slices instead of list of indices
        for i in range(indices[0], indices[-1] + 1, self.batch_size):
            if self.drop_last and i + self.batch_size > indices[-1] + 1:
                break
            yield slice(i, min(i + self.batch_size, indices[-1] + 1))


class ZizouDataset(Dataset):
    def __init__(self, store, features, target_transform=None, feature_transforms=None):
        self.target_transform = target_transform
        self.feature_transforms = feature_transforms
        self.store = copy.copy(store)
        self.features = features
        self.resolution = None
        self.starttime = store.starttime
        self.endtime = store.endtime
        self.init_dataset()
        self.npts = int(
            (self.endtime - self.starttime) / datetime.timedelta(hours=self.resolution)
        )
        self.nfeatures = self.get_shapes()[1]
        self.dates = None

    def get_dates(self):
        return pd.date_range(
            start=self.store.starttime,
            end=self.store.endtime,
            freq=datetime.timedelta(hours=self.resolution),
        )

    def init_dataset(self):
        self.scaling_params = {}
        for feature in self.features:
            scaling_params_file = os.path.join(
                self.store.path, f"scaling_params_{feature}.json"
            )
            fin = self.store.feature_path(feature)
            self.scaling_params[feature] = min_max_scale_netcdf(
                fin, feature, scaling_params_file
            )
            self.get_resolution(fin)

    def get_resolution(self, fin):
        with h5netcdf.File(fin, "r") as f:
            _resolution = f["original"].attrs["resolution"]
            if self.resolution is None:
                self.resolution = _resolution
            else:
                if self.resolution != _resolution:
                    msg = "Resolution is not consistent.\n"
                    msg += f"Expected {self.resolution}, got {_resolution}.\n"
                    msg += f"Feature: {fin}\n"
                    raise ValueError(msg)

    def get_shapes(self):
        nsamples = None
        nfeatures = 0
        for feature in self.features:
            shape_dict = self.store.shape(feature)
            for key, value in shape_dict.items():
                if key != "datetime":
                    nfeatures += value
                else:
                    if len(shape_dict) == 1:
                        nfeatures += 1
                    if nsamples is None:
                        nsamples = value
                    else:
                        if nsamples != value:
                            msg = "Number of samples is not consistent."
                            msg += f"Expected {nsamples}, got {value}"
                            msg += f"for feature: {feature}"
                            logger.warning(msg)
        return (nsamples, nfeatures)

    def load_and_scale_feature(self, feature):
        data = self.store(feature)
        if len(data.shape) > 1:
            data = data.fillna(
                np.array(self.scaling_params[feature]["median"])[:, np.newaxis]
            )
            data = (
                data - np.array(self.scaling_params[feature]["min"])[:, np.newaxis]
            ) / (
                np.array(self.scaling_params[feature]["max"])[:, np.newaxis]
                - np.array(self.scaling_params[feature]["min"])[:, np.newaxis]
            )
        else:
            data = data.fillna(self.scaling_params[feature]["median"])
            data = (data - self.scaling_params[feature]["min"]) / (
                self.scaling_params[feature]["max"]
                - self.scaling_params[feature]["min"]
            )
        return data

    def load_data(self, starttime, endtime):
        features = []
        var_cnt = 0
        for _f in self.features:
            logger.debug(f"Reading feature {_f} between {starttime} and {endtime}.")
            self.store.starttime = starttime
            self.store.endtime = endtime
            feat = self.load_and_scale_feature(_f)
            if len(feat.shape) > 1:
                for c in range(feat.shape[0]):
                    da = feat[c].reset_coords(drop=True)
                    feat_name = f"variable_{var_cnt}"
                    var_cnt += 1
                    features.append(da.rename(feat_name))
            else:
                if self.feature_transforms is not None:
                    features.append(
                        self.feature_transforms[_f](feat.rename(f"variable_{var_cnt}"))
                    )
                else:
                    features.append(feat.rename(f"variable_{var_cnt}"))
                var_cnt += 1
        feats = xr.merge(features).to_array()
        self.dates = feats.datetime.values
        return feats.transpose("datetime", "variable").values

    def __len__(self):
        return self.npts

    def __getitems__(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.int64):
            start = [int(idx)]
            stop = [int(idx)]
        elif isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            if start is None:
                start = 0
            else:
                start = idx.start
            if stop is None:
                stop = self.npts - 1
            elif stop >= self.npts:
                stop = self.npts - 1
            else:
                stop = idx.stop - 1
            assert stop >= start, "Stop index must be greater than start index."
            start = [start]
            stop = [stop]

        elif isinstance(idx, list) or isinstance(idx, tuple):
            start = list(idx)
            stop = list(idx)

        elif isinstance(idx, np.ndarray):
            start = idx.tolist()
            stop = idx.tolist()

        # Handle PyTorch tensor of indices
        elif isinstance(idx, torch.Tensor):
            if (
                idx.dtype == torch.long or idx.dtype == torch.int
            ):  # Ensure it's an index tensor
                start = idx.tolist()
                stop = idx.tolist()
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        all_data = []
        for _st, _sp in zip(start, stop):
            date_start = self.starttime + datetime.timedelta(
                hours=_st * self.resolution
            )
            date_end = self.starttime + datetime.timedelta(hours=_sp * self.resolution)
            _data = self.load_data(date_start, date_end)
            all_data.append(_data)
        sample = np.array(all_data).squeeze()
        target = np.array(all_data).squeeze()
        if self.target_transform is not None:
            sample = self.target_transform(sample)
            target = self.target_transform(target)
        return sample, target


class AnomalyDetectionBaseClass(object):
    def __init__(self, features, stacks=None, transforms=None):
        self.features = features
        self.stacks = stacks if stacks is not None else dict()
        self.transforms = transforms if transforms is not None else dict()

    def fit(self, starttime, endtime):
        """
        Train the model.
        """
        msg = "'training' is not implemented yet"
        raise NotImplementedError(msg)

    def infer(self, starttime, endtime):
        """
        Use the model to infer.
        """
        msg = "'inference' is not implemented yet"
        raise NotImplementedError(msg)

    def write_model_parameters(self):
        """
        Write principal components or
        model weights to file
        """
        msg = "'write_model_parameters' is not implemented yet"
        raise NotImplementedError(msg)

    def read_model_parameters(self):
        """
        Read principal components or
        model weights from file
        """
        msg = "'read_model_parameters' is not implemented yet"
        raise NotImplementedError(msg)

    def write_hyperparameters(self):
        """
        Store things like random seed,
        learning rate, number of principal
        components etc.
        """
        msg = "'write_hyperparameters' is not implemented yet"
        raise NotImplementedError(msg)

    def compute_anomaly_index(self):
        """
        A scalar value between 0 and 1 indicating
        the degree of anomaly (1=most anomaluous).
        """
        msg = "'compute_anomaly_index' is not implemented yet"
        raise NotImplementedError(msg)
