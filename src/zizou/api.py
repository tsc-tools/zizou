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
import xarray as xr

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
                resolution: str = '1m',
                verticalres: int = 20):

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
                xda = self.compute_spectrogram(tr)
                nfreqs = xda.shape[0] 
                dates = xda.coords['datetime'].values
                freq, dates, spec = self.aggregate_feature(resolution, verticalres, xda, nfreqs, dates)
                vals = spec.ravel(order='C')
                vals = (vals - np.nanmin(vals)) / \
                    (np.nanmax(vals) - np.nanmin(vals))
                freqs = freq.repeat(dates.size)
                dates = np.tile(dates, freq.size)
                df = pd.DataFrame(
                    {'dates': dates, 'freqs': freqs, 'feature': vals})
                df['dates'] = pd.to_datetime(df.dates.values).strftime('%Y-%m-%dT%H:%M:%SZ')
                output = df.to_csv(index=False, columns=['dates', 'freqs', 'feature']) 
            if output is not None:
                return StreamingResponse(iter([output]),
                                            media_type='text/csv',
                                            headers={"Content-Disposition":
                                                    "attachment;filename=<VUMT_feature>.csv",
                                                    'Content-Length': str(len(output))})

    def compute_spectrogram(self, tr, nperseg=256, noverlap=128):
            win = tukey(nperseg, alpha=0.25)
            stft = ShortTimeFFT(win, hop=noverlap, fs=tr.stats.sampling_rate)
            Sxx = stft.spectrogram(tr.data)
            (t1, t2, f1, f2) = stft.extent(tr.stats.npts)
            frequencies = np.linspace(f1, f2, Sxx.shape[0])
            desired_start_date = pd.to_datetime(tr.stats.starttime.isoformat())
            times = np.linspace(t1, t2, Sxx.shape[1])
            datetimes = pd.to_datetime(times, unit='s')
            time_difference = desired_start_date - datetimes.min()
            datetimes = datetimes + time_difference
            xda = xr.DataArray(10 * np.log10(Sxx), coords=[frequencies, datetimes],
                               dims=['frequency', 'datetime'])
            return xda 

 

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
