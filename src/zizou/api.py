from argparse import ArgumentParser
from datetime import timedelta
from typing import Iterable

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from obspy import UTCDateTime
import pandas as pd
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import tukey
import uvicorn
from tonik.api import TonikAPI

from zizou import DataSource, S3Waveforms, FDSNWaveforms


class ZizouAPI(TonikAPI):
    def __init__(self, rootdir: str, backend: str = "netcdf",
                 fdsn_urls: Iterable[str] = ("https://service.geonet.org.nz", "https://service-nrt.geonet.org.nz")):
        super(ZizouAPI, self).__init__(rootdir, backend)
        datasources = []
        datasources.append(S3Waveforms(staxml_dir=rootdir))
        for url in fdsn_urls:
            datasources.append(FDSNWaveforms(url=url))
        self.datasource = DataSource(datasources)
        self.app.get("/waveform")(self.waveform)
        self.app.get("/spectrogram")(self.spectrogram)

    def get_time_difference(self, starttime, endtime):
            t0 = UTCDateTime(starttime)
            t1 = UTCDateTime(endtime)
            time_difference_seconds = abs(t1 - t0)
            return t0,t1,time_difference_seconds

    def waveform(self,
                 network: str,
                 station: str,
                 location: str,
                 channel: str,
                 starttime: str,
                 duration: int,
                 maxResLimit: float = 1.0,
                 resolution: str = "full"):
        """
        Get instrument-corrected waveforms.
        """
        _st = self.preprocess_datetime(starttime)
        _et = _st + timedelta(seconds=duration) 
        t0, t1, time_difference_seconds = self.get_time_difference(_st, _et)

        if time_difference_seconds <= 3600*maxResLimit:
            for tr in self.datasource.get_waveforms(network, station, location, channel, t0, t1):
                if not tr:
                    raise HTTPException(status_code=404, detail=f"No data in period {_st} - {_et} for {network}.{station}.{location}.{channel}")
                tr.normalize()
                tr.detrend('constant')
                tr.resample(25)
                tr.filter('bandpass', freqmin=0.5, freqmax=12, corners=4, zerophase=True)
                dates = [dt.isoformat() for dt in tr.times("utcdatetime")]
                df = pd.DataFrame({'feature':tr.data, 'dates': dates})
            output = df.to_csv(index=False, columns=['dates', 'feature'])
        else:
            raise HTTPException(status_code=404, detail=f"Request for {network}.{station}.{location}.{channel} exceeds maxResLimit of {maxResLimit} hours")
            
        if output is not None:
            return StreamingResponse(iter([output]),
                                    media_type='text/csv',
                                    headers={"Content-Disposition":
                                             "attachment;filename=<feature>.csv",
                                             "Content-Length": str(len(output))})

    def spectrogram(self,
                network: str,
                station: str,
                location: str,
                channel: str,
                starttime: str,
                duration: int,
                maxResLimit: float = 1.0,
                resolution: str = "full"):

        output = None
        _st = self.preprocess_datetime(starttime)
        _et = _st + timedelta(seconds=duration) 
        t0, t1, time_difference_seconds = self.get_time_difference(_st, _et)

        if time_difference_seconds <= 3600*maxResLimit:
            for tr in self.datasource.get_waveforms(network, station, location, channel, t0, t1):
                if not tr:
                    raise HTTPException(status_code=404, detail=f"No data in period {_st} - {_et} for {network}.{station}.{location}.{channel}")
                tr.detrend('constant')
                tr.resample(25)
                df = self.getSpectrogramDataFrame(tr)
                df['feature'] = df['feature'].abs()
                output = df.to_csv(index=False, columns=['dates', 'freqs', 'feature'])
                
            if output is not None:
                return StreamingResponse(iter([output]),
                                            media_type='text/csv',
                                            headers={"Content-Disposition":
                                                    "attachment;filename=<VUMT_feature>.csv",
                                                    'Content-Length': str(len(output))})

    def getSpectrogramDataFrame(self, tr, nperseg=256, noverlap=128):
            win = tukey(nperseg, alpha=0.25)
            stft = ShortTimeFFT(win, hop=noverlap, fs=tr.stats.sampling_rate)
            Sxx = stft.spectrogram(tr.data)
            (t1, t2, f1, f2) = stft.extent(tr.stats.npts)
            frequencies = np.linspace(f1, f2, Sxx.shape[0])
            times = np.linspace(t1, t2, Sxx.shape[1])
            spectrogram_df = pd.DataFrame(10 * np.log10(Sxx), columns=times, index=frequencies)
            spectrogram_df = spectrogram_df.unstack().reset_index()
            spectrogram_df.columns = ['dates', 'freqs', 'feature']
            desired_start_date = pd.to_datetime(tr.stats.starttime.isoformat())
            spectrogram_df['dates'] = pd.to_datetime(spectrogram_df['dates'], unit='s')
            time_difference = desired_start_date - spectrogram_df['dates'].min()
            spectrogram_df['dates'] =spectrogram_df['dates'] + time_difference
            spectrogram_df = spectrogram_df.dropna()
            return spectrogram_df

 

def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--rootdir", default='/tmp')
    parser.add_argument("--backend", default='netcdf')
    parser.add_argument("-p", "--port", default=8003, type=int)
    parser.add_argument("--host", default='0.0.0.0')
    args = parser.parse_args(argv)
    za = ZizouAPI(args.rootdir, backend=args.backend)
    uvicorn.run(za.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
