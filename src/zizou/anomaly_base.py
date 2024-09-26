import logging
import os

import numpy as np
from obspy import UTCDateTime
import pandas as pd
from tqdm import tqdm
import xarray as xr


logger = logging.getLogger(__name__)


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

    def get_features(self, data, featurefile=None):
        try:
            feats = xr.open_dataarray(featurefile)
        except (FileNotFoundError, ValueError):
            features = []
            for _f in self.features:
                logger.info(f"Reading feature {_f}.") 
                feat = data(_f, self.stacks.get(_f, None))
                if _f == 'sonogram':
                    for c in range(feat.shape[0]):
                        da = feat[c].reset_coords(names=['sonofrequency'], drop=True)
                        feat_name = 'sonogram_{:d}'.format(c)
                        features.append(da.rename(feat_name))
                else:
                    if self.transforms.get(_f, None) == 'log':
                        features.append(np.log10(feat))
                    else:
                        features.append(feat)
                
            feats = xr.merge(features).to_array()
            if featurefile is not None:
                feats.to_netcdf(featurefile)
        return feats.transpose('datetime', 'variable')
