"""
Train an autoencoder to reduce dimensionality of input features.
"""

import logging
import math
import os
import pickle
from collections.abc import Sequence
from typing import List

import numpy as np
import skfuzzy as fuzz
import tonik
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split

from zizou import AnomalyDetectionBaseClass, SliceBatchSampler, ZizouDataset

logger = logging.getLogger(__name__)


class TiedAutoEncoder(nn.Module):
    """
    Applies an tied-weight autoencoder to the incoming data.
    From https://gist.github.com/northanapon/375e17fb395391c144deff20914e51df
    Parameters
    -------------
    in_features : int
        size of each input sample.
    h_features : List[int]
        a list of size of each layer for the encoder and
        the reverse for the decoder.
    activation : function, optional
        an activation function to apply for each layer
        (the default is torch.tanh).
    """

    def __init__(self, in_features: int, h_features: List[int], activation=F.relu):
        """Create an autoencoder."""
        super().__init__()
        self.in_features = in_features
        self.h_features = h_features
        self.encoded_features = h_features[-1]
        self.activation = activation
        self.weights = nn.ParameterList([])
        self.enc_biases = nn.ParameterList([])
        self.dec_biases = nn.ParameterList([])
        in_dim = in_features
        for h in h_features:
            self.weights.append(nn.Parameter(torch.DoubleTensor(h, in_dim)))
            self.enc_biases.append(nn.Parameter(torch.DoubleTensor(h)))
            in_dim = h
        for h in reversed(h_features[:-1]):
            self.dec_biases.append(nn.Parameter(torch.DoubleTensor(h)))
        self.dec_biases.append(nn.Parameter(torch.DoubleTensor(in_features)))
        self.reset_parameters()
        self.loss = torch.nn.MSELoss(reduction="none")

    def forward(self, x: torch.DoubleTensor):
        """Return result of encoding and decoding."""
        dec = self.encode(x)
        return self.decode(dec)

    def encode(self, x: torch.DoubleTensor):
        """Return encoded data."""
        o = x
        for w, b in zip(self.weights[:-1], self.enc_biases[:-1]):
            o = F.linear(o, w, b)
            o = self.activation(o)
        return F.linear(o, self.weights[-1], self.enc_biases[-1])

    def decode(self, o: torch.DoubleTensor):
        """Return decoded data."""
        r_weights = list(reversed(self.weights))
        for w, b in zip(r_weights[:-1], self.dec_biases[:-1]):
            o = F.linear(o, w.t(), b)
            o = self.activation(o)
        return F.linear(o, r_weights[-1].t(), self.dec_biases[-1])

    def reset_parameters(self):
        """Reset linear module parameters (from nn.Linear)."""
        for w, b in zip(self.weights, self.enc_biases):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if b is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(b, -bound, bound)
        for w, b in zip(self.weights, self.dec_biases):
            if b is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w.t())
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(b, -bound, bound)


class FuzzyCluster(TransformerMixin, BaseEstimator):
    def __init__(self, n_clusters=5, error=0.005, maxiter=1000, init=None):
        self.n_clusters_ = n_clusters
        self.error_ = error
        self.maxiter_ = maxiter
        self.init_ = init
        self.clusterscaler = MinMaxScaler()
        self._is_fitted = False

    def fit(self, data, scalerfile, clustercenterfile):
        try:
            with open(scalerfile, "rb") as fh:
                self.clusterscaler = pickle.load(fh)
            self.cntr_ = np.load(clustercenterfile)
            self._is_fitted = True
            logging.info("Loaded cluster model.")
        except FileNotFoundError:
            data_scaled = self.clusterscaler.fit_transform(data)
            self.cntr_, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data_scaled.T, self.n_clusters_, 2, error=0.005, maxiter=1000, init=None
            )
            with open(scalerfile, "wb") as fh:
                pickle.dump(self.clusterscaler, fh)
            np.save(clustercenterfile, self.cntr_)
            self._is_fitted = True

    def transform(self, data):
        assert self._is_fitted
        data_scaled = self.clusterscaler.transform(data)
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            data_scaled.T, self.cntr_, 2, error=0.005, maxiter=1000
        )
        return u

    def fit_transform(self, data, scalerfile, clustercenterfile):
        self.fit(data, scalerfile, clustercenterfile)
        return self.transform(data)


class AutoEncoder(AnomalyDetectionBaseClass):
    features = [
        "rsam",
        "dsar",
        "central_freq",
        "predom_freq",
        "bandwidth",
        "rsam_energy_prop",
        "sonogram",
    ]

    transform_dict = {"rsam": "log"}

    stack_dict = {
        "dsar": "2D",
        "central_freq": "1h",
        "predom_freq": "1h",
        "variance": "1h",
        "bandwidth": "1h",
    }

    files = dict(
        modelfile="autoencoder_weights.pth",
        scalerfile="cluster_scaler.pkl",
        clustercenterfile="cluster_center.npy",
    )

    def __init__(
        self,
        store: tonik.Storage,
        features=None,
        layers: Sequence = [2000, 500, 200, 6],
        epochs: int = 5,
        batch_size: int = 1024,
        patience: int = 10,
        device: str = "cpu",
        n_clusters: int = 5,
        configfile: str = None,
    ):
        self.store = store
        if features is not None:
            self.features = features
        self.layers_ = layers
        self.epochs_ = epochs
        self.batch_size_ = batch_size
        self.patience_ = patience
        self.device_ = device
        if configfile is not None:
            try:
                with open(configfile, "r") as fh:
                    c = yaml.safe_load(fh)
            except OSError:
                c = yaml.safe_load(configfile)
            self.layers_ = c["autoencoder"].get("layers", layers)
            self.epochs_ = c["autoencoder"].get("epochs", epochs)
            self.patience_ = c["autoencoder"].get("patience", patience)
            self.device_ = c["autoencoder"].get("device", device)

        super(AutoEncoder, self).__init__(
            self.features, self.stack_dict, self.transform_dict
        )
        self.device_ = (
            self.device_
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using {self.device_} device")
        self.cluster_ = FuzzyCluster(n_clusters=n_clusters)
        self.classifications = None
        self.dataset = ZizouDataset(
            store, self.features, target_transform=torch.from_numpy
        )
        autoencoder_ = TiedAutoEncoder(self.dataset.nfeatures, self.layers_)
        self.model_ = autoencoder_.to(self.device_)
        logger.info(self.model_)

    def clear(self):
        for _f in self.files.values():
            try:
                os.remove(os.path.join(self.store.path, _f))
            except FileNotFoundError:
                pass

    def checkpoint(self, model, filename):
        torch.save(model.state_dict(), filename)

    def resume(self, model, filename):
        model.load_state_dict(torch.load(filename))

    def train(self, dataloader, optimizer):
        size = len(dataloader.dataset)
        self.model_.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device_), y.to(self.device_)

            # Compute prediction error
            pred = self.model_(X)
            loss = self.model_.loss(pred, y).sum(axis=1).mean()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                logger.info(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, loss_fn):
        num_batches = len(dataloader)
        self.model_.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device_), y.to(self.device_)
                pred = self.model_(X)
                test_loss += loss_fn(pred, y).sum(axis=1).mean()
        test_loss /= num_batches
        self.loss_ = test_loss
        logger.info(f"Test loss: {test_loss:>8f} \n")
        return test_loss

    def _is_fitted(self, store):
        try:
            self.model_.load_state_dict(
                torch.load(os.path.join(store.path, self.files["modelfile"]))
            )
            logging.info("Loaded pretrained model.")
        except FileNotFoundError:
            return False
        return True

    def fit(self, new=False):
        if self._is_fitted(self.store) and not new:
            return
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=1e-3)
        logger.info("Splitting dataset...")
        train_size = int(0.8 * len(self.dataset))
        train_sampler = SliceBatchSampler(range(train_size), self.batch_size_, False)
        test_sampler = SliceBatchSampler(
            range(train_size, len(self.dataset)), self.batch_size_, False
        )
        train_dataloader = DataLoader(self.dataset, batch_sampler=train_sampler)
        test_dataloader = DataLoader(self.dataset, batch_sampler=test_sampler)
        logger.info("Starting training...")
        best_loss = 1e30
        best_epoch = 0
        for i in range(self.epochs_):
            logger.info(f"Epoch {i+1}")
            self.train(train_dataloader, optimizer)
            loss = self.test(test_dataloader, self.model_.loss)
            if loss < best_loss:
                best_loss = loss
                best_epoch = i
                self.checkpoint(
                    self.model_, os.path.join(self.store.path, self.files["modelfile"])
                )
            elif i - best_epoch > self.patience_:
                logger.info(f"Early stopped training at epoch {i+1}")
                break

    def transform(self, starttime, endtime):
        assert self._is_fitted(self.store)

        logger.info("Starting transform process.")
        self.store.starttime = starttime
        self.store.endtime = endtime
        zd = ZizouDataset(self.store, self.features, target_transform=torch.from_numpy)
        sampler = SliceBatchSampler(range(len(zd)), self.batch_size_, False)
        dataloader = DataLoader(zd, batch_sampler=sampler)
        logger.info("Transforming data.")
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device_)
                encoded_data = self.model_.encode(X)
                X_hat = self.model_.decode(encoded_data)

                encoded_data = encoded_data.cpu().numpy()

                output = {}
                dates = zd.dates
                logger.info("Calculating loss between input and decoded output.")
                loss = self.model_.loss(X, X_hat).sum(axis=1)
                output["autoencoder_loss"] = xr.DataArray(
                    loss.cpu().numpy(), coords=[dates], dims=["datetime"]
                )
                logger.info(
                    f"Clustering embeddings into {self.cluster_.n_clusters_} cluster."
                )
                clustered_data = self.cluster_.fit_transform(
                    encoded_data,
                    os.path.join(self.store.path, self.files["scalerfile"]),
                    os.path.join(self.store.path, self.files["clustercenterfile"]),
                )
                cluster_names = list(range(self.cluster_.n_clusters_))
                output["autoencoder_cluster"] = xr.DataArray(
                    clustered_data,
                    coords=[cluster_names, dates],
                    dims=["cluster", "datetime"],
                )
                output["autoencoder_embedding"] = xr.DataArray(
                    encoded_data.T,
                    coords=[list(range(encoded_data.shape[1])), dates],
                    dims=["autodim", "datetime"],
                )
                xds = xr.Dataset(output)
                self.store.save(xds)

    def fit_transform(self, starttime, endtime):
        self.fit()
        self.classifications = self.transform(starttime, endtime)
        return self.classifications
