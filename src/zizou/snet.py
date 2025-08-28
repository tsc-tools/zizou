import logging
from obspy import UTCDateTime
from zizou import DataSource, S3Waveforms
from zizou.util import window_array
from scatseisnet import ScatteringNetwork
import numpy as np
from obspy import Stream
import mlflow
import xarray as xr
import pandas as pd
from tonik.xarray2zarr import xarray2zarr
from zizou import FeatureBaseClass
from tqdm import tqdm  # progress bar

logger = logging.getLogger(__name__)

# New helper to configure logging
def configure_logging(log_file: str, level: int = logging.INFO):
    """Configure logging to file and limit console output.

    Parameters
    ----------
    log_file : str
        Path to the log file where detailed logs will be written.
    level : int
        Logging level for the file handler.
    """
    root = logging.getLogger()
    # Avoid duplicate handlers if reconfigured
    if getattr(root, "_snet_logging_configured", False):
        return
    root.setLevel(logging.DEBUG)

    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)

    # Console handler (only errors so progress bar remains clean)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)
    root._snet_logging_configured = True
    logger.debug("Logging configured. File: %s", log_file)

class SNet(FeatureBaseClass):
    def __init__(self):
        super(SNet, self).__init__()
        self.segment_duration_seconds = 1200.0
        self.sampling_rate_hertz = 25.0
        self.overlap = 0.5
        self.steplength = int(self.segment_duration_seconds*self.sampling_rate_hertz) - int(self.segment_duration_seconds*self.sampling_rate_hertz * self.overlap)
        self.samples_per_segment = int(self.segment_duration_seconds * self.sampling_rate_hertz)
        self.bank_keyword_arguments = (
            {"octaves": 6, "resolution": 4, "quality": 2},
            {"octaves": 6, "resolution": 1, "quality": 2},
        )

    def compute(self, st: Stream):
        window_len = int(self.segment_duration_seconds * self.sampling_rate_hertz)
        noverlap = int(self.segment_duration_seconds * self.sampling_rate_hertz * self.overlap)
        resultZ, windowValsZ = window_array(st[0].data, window_len, noverlap)
        result1, windowVals1 = window_array(st[1].data, window_len, noverlap)
        result2, windowVals2 = window_array(st[2].data, window_len, noverlap)
        segments = np.stack([resultZ.T, result1.T, result2.T], axis=1)
        timestamps = [(st[0].stats.starttime + _t).datetime for _t in np.arange(resultZ.shape[1])*self.steplength / self.sampling_rate_hertz]
        network = ScatteringNetwork(
            *self.bank_keyword_arguments,
            bins=self.samples_per_segment,
            sampling_rate=self.sampling_rate_hertz,
        )
        scattering_coefficients = network.transform(segments, reduce_type=np.median)
        coords_order1 = {'datetime': pd.to_datetime(timestamps), 'channel': np.arange(scattering_coefficients[0].shape[1]),
                        'frequency': network.banks[0].centers}
        coords_order2 = coords_order1.copy()
        coords_order2['order2 frequency'] = network.banks[1].centers 
        xda_order1 = xr.DataArray(scattering_coefficients[0], dims=['datetime', 'channel', 'frequency'],
                                coords=coords_order1)
        xda_order2 = xr.DataArray(scattering_coefficients[1], dims=['datetime', 'channel', 'frequency', 'order2 frequency'],
                                coords=coords_order2)
        self.feature = xr.Dataset({'order1': xda_order1, 'order2': xda_order2})
        self.feature.attrs["starttime"] = st[0].stats.starttime.isoformat()
        self.feature.attrs["endtime"] = st[0].stats.endtime.isoformat()
        self.feature.attrs["station"] = st[0].stats.station
        return self.feature

def main(argv=None):
    from argparse import ArgumentParser
    from zizou.visualise import plot_ssam_mpl
    parse = ArgumentParser()
    parse.add_argument('-s', '--starttime',
                      help="Define starttime in ISO860 format, e.g. 2022-01-01T00:00:00",
                      default="2022-01-01T00:00:00")
    parse.add_argument('-e', '--endtime',
                      help="Define endtime in ISO860 format, e.g. 2022-01-02T00:00:00",
                      default="2022-01-02T00:00:00")
    parse.add_argument('-d', '--data_dir',
                      help="Define data directory, e.g. /home/zizou/data/robust_forecasting",
                      default="/home/zizou/data/robust_forecasting")
    parse.add_argument('-c', '--chunks',
                      help="Define chunk size for Zarr storage.",
                      default=144, type=int) 
    parse.add_argument('--experiment',
                      help="Define MLflow experiment name.",
                      default="scatseisnet")
    parse.add_argument('--mlflow_tracking_uri',
                      help="Define MLflow tracking URI.",
                      default=None)
    parse.add_argument('--mlflow_run_name',
                      help="Define MLflow run name.",
                      default=None)
    parse.add_argument('--plot-only', \
                      help="Only plot the feature without computing it.", \
                      action='store_true')
    parse.add_argument('--log-file', \
                      help="Log file path.", \
                      default='snet-log.txt')
    args = parse.parse_args(argv)

    # Configure logging
    configure_logging(args.log_file)
    logger.info("Starting snet with args: %s", args)

    s3w = S3Waveforms(staxml_dir=args.data_dir) 
    ds = DataSource(clients=[s3w], chunk_size=86400)
    starttime = UTCDateTime(args.starttime)
    endtime = UTCDateTime(args.endtime)
    zarr_path = f"{args.data_dir}/{args.experiment}"
    if not args.plot_only:
        logger.info(f"Fetching data from {starttime} to {endtime}")
        # Progress bar over waveform chunks
        for st in tqdm(ds.get_waveforms("NZ", "MAVZ", "10", "HH?", starttime, endtime, cache=True),
                       desc="Waveform chunks", unit="chunk"):
            if len(st) < 3:
                continue
            for tr in st:
                isn = np.isnan(tr.data)
                if isn.any(): 
                    idx = np.where(~isn, np.arange(len(tr.data)), 0)
                    if np.isnan(tr.data[0]):
                        tr.data[0] = tr.data[np.where(~isn)[0][0]]
                    np.maximum.accumulate(idx, out=idx)
                    tr.data[isn] = tr.data[idx[isn]]
                tr.detrend("linear")
                tr.interpolate(sampling_rate=25)
            s = SNet()
            feature = s.compute(st)
            feature['datetime'] = feature.datetime.dt.round("600s")
            logger.info(f"Computed feature with shape {feature['order1'].shape} and {feature['order2'].shape}")
            logger.info(f"Feature starttime: {feature.attrs['starttime']}, endtime: {feature.attrs['endtime']}")
            logger.info(f"Feature station: {feature.attrs['station']}")
            xarray2zarr(feature, zarr_path, chunks=args.chunks)
            logger.info(f"Feature saved to {zarr_path}")
    xds = xr.open_zarr(f"{zarr_path}/order1.zarr", group='original')
    # Log to MLflow
    fig = plot_ssam_mpl(xds.order1.loc[dict(channel=0)].sel(frequency=slice(None, None, -1)).transpose('frequency', 'datetime'),
                        dbscale=True, canvas_dim=None, log=True)
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        experiment = mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name) as run:
            mlflow.log_param("starttime", str(starttime))
            mlflow.log_param("endtime", str(endtime))
            mlflow.log_figure(fig, artifact_file="plots/scattering_coefficients_plot.png")
            mlflow.log_artifact("snet-log.txt", artifact_path="logs")
    else:
        fig.savefig("scattering_coefficients_plot.png")

if __name__ == "__main__":
    main()
