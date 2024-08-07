"""
The main PCA script
"""
import logging
import os
import pickle
from argparse import ArgumentParser
from configparser import ConfigParser

from obspy import UTCDateTime
import sklearn.impute
import sklearn.decomposition
import xarray as xr

from zizou import AnomalyDetectionBaseClass, get_data
from zizou.data import VolcanoMetadata


logger = logging.getLogger(__name__)


class ModelException(Exception):
    pass


class PCA(AnomalyDetectionBaseClass):

    def __init__(self, modelfile='pcaModel.pkl', pca_features=None, interval=600,
                 n_components=4):
        self.pca_features = pca_features
        self.modelfile = modelfile
        self.interval = interval
        self.n_components = n_components
        self.__pca_model = None
        self.__mean = None
        self.__std = None
        self.__feature = None
        self.__training_run = False

    @property
    def pca_model(self):
        return self.__pca_model

    @property
    def mean(self):
        return self.__mean

    @property
    def std(self):
        return self.__std

    @property
    def feature(self):
        return self.__feature

    def fit(self, data):
        if os.path.isfile(self.modelfile):
            self._read_model_parameters(self.modelfile)
        else:
            pca = sklearn.decomposition.PCA(n_components=self.n_components)
            data = self.get_features(self.fq).values
            pca.fit(data)
            self.__pca_model = pca
            self.__mean = mns
            self.__std = stds
            self._write_model_parameters(self.modelfile)
        self.__training_run = True

    def _write_model_parameters(self, modelfile):
        dirName = os.path.dirname(modelfile)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        with open(modelfile, 'wb') as fh:
            pickle.dump([self.__pca_model, self.__mean, self.__std], fh)

    def _read_model_parameters(self, modelfile):
        with open(modelfile, 'rb') as fh:
            self.__pca_model, self.__mean, self.__std = pickle.load(fh)

    def infer(self, fq, savedir=None, overwrite=False):
        '''
        :param fq: Object to request features.
        :type fq: :class:`FeatureRequest`
        :param savedir:
        :param overwrite:
        :return:
        '''
        if not self.__training_run:
            msg = "Run 'training' first."
            raise ModelException(msg)

        feats, mns, stds = self.featureStack(fq)
        vals = self.pca_model.transform(feats.values.T)
        keys = ['pc%d' % i for i in range(self.n_components)]
        foo = xr.DataArray(vals, coords=[feats.datetime.values,
                                         keys],
                           dims=['datetime', 'pca_component'])
        xdf = xr.Dataset({'pca': foo})
        xdf.attrs['starttime'] = fq.starttime.isoformat()
        xdf.attrs['endtime'] = fq.endtime.isoformat()
        xdf.attrs['station'] = fq.site
        if savedir is not None:
            self.save_model_output(xdf, savedir, overwrite=overwrite)
        return xdf


def runPca(argv=None):
    parser = ArgumentParser(prog='compute_pca',
                            description=__doc__.strip())
    parser.add_argument('-c', '--config', type=str,
                        default=get_data('package_data/config.ini'),
                        help='Path to config file.')
    parser.add_argument('-m', '--metadata', type=str,
                        default=get_data('package_data/metadata.json'),
                        help='Volcano metadata json')
    parser.add_argument('-o', '--out_dir', type=str,
                        default='/tmp',
                        help='Output dir')
    parser.add_argument('-s', '--starttime', type=str,
                        help='Start of the raw data window')
    parser.add_argument('-e', '--endtime', type=str,
                        help='End of the raw data window')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files instead merging.')

    args = parser.parse_args(argv)
    endtime = UTCDateTime.now()
    starttime = endtime-3600.
    if args.starttime:
        starttime = UTCDateTime(args.starttime)
    if args.endtime:
        endtime = UTCDateTime(args.endtime)

    runPcaCore(args.config, args.metadata, args.out_dir,
               endtime.datetime, starttime.datetime,
               overwrite=args.overwrite)


def runPcaCore(config_file, metadata_file, out_dir,
               end_time=UTCDateTime.now(),
               start_time=None, overwrite=False):
    config = ConfigParser()
    config.read(config_file)
    volcanoes = config['DEFAULT']['volcanoes'].split(',')
    # List of features to use in PCA
    pca_features = config['PCA']['features'].split(',')
    vm = VolcanoMetadata(file=metadata_file)
    for volcano in volcanoes:
        vm.set_volcano_name(volcano)
        streams = vm.get_seismic_network_streams()
        for stream in streams:
            net, site, loc, comp = stream[0].split('.')
            st_stream = UTCDateTime(stream[1])
            # Check that input dates are valid for channel
            if start_time is None:
                if end_time > st_stream:
                    start_time = UTCDateTime(st_stream.year,
                                             st_stream.month,
                                             st_stream.day+1, 0, 0, 0)
            fq = FeatureRequest(volcano=volcano, site=site, channel=comp,
                                starttime=start_time, endtime=end_time)
            msg = "Compute PCA for: {}, {}, {}, {}, {}"
            msg = msg.format(volcano, site, comp, start_time, end_time)
            logging.info(msg)
            savedir = os.path.join(out_dir, volcano, site, comp, 'PCA')
            modelfile = os.path.join(savedir, "pca.pkl")
            pca = PCA(modelfile=modelfile, pca_features=pca_features)
            pca.fit(fq)
            pca.infer(fq, savedir=savedir, overwrite=overwrite)

if __name__ == '__main__':
    runPca()
