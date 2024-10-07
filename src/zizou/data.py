import inspect
import io
import logging
import os
from datetime import date, datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd
import requests
import yaml
from botocore import UNSIGNED
from botocore.config import Config
from obspy import Inventory, Stream, Trace, UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.filesystem.sds import Client as SDS_Client

from zizou.util import test_signal

logger = logging.getLogger(__name__)


class PostProcessException(Exception):
    pass


class PostProcess:
    """
    Class for checking waveforms and station metadata for valid values,
    filling waveform data gaps and removing instrument sensitivity
    """

    def __init__(
        self,
        st=Stream(),
        inv=Inventory(),
        inv_dt=Inventory(),
        startdate=None,
        enddate=None,
        loc="*",
        comp="*Z",
        fill_value=np.nan,
    ):
        self.st = st
        self.inv = inv
        self.inv_dt = inv_dt
        self.startdate = startdate
        self.enddate = enddate
        self.loc = loc
        self.comp = comp
        self.fill_value = fill_value

        # Get check registry
        self.checklist = [
            x
            for x in inspect.getmembers(self)
            if inspect.ismethod(x[1]) and x[0].startswith("check")
        ]
        # Get matching post-processing functions
        self.pp_functions = []
        for check in self.checklist:
            self.pp_functions.append(
                [
                    x
                    for x in inspect.getmembers(self)
                    if x[0] == check[0].replace("check", "output")
                ][0]
            )

        # Run checks
        self.res = None

    def run_post_processing(self):
        self.res = self.run_checks()
        func = [a[1] for (a, b) in zip(self.pp_functions, self.res) if b][0]
        trace = func()
        return trace

    def run_checks(self):
        self.res = [check[1]() for check in self.checklist]
        if sum(self.res) == 0:
            raise NotImplementedError(
                "No post processing not set up for this case:\n" "{}".format(self.st)
            )
        elif sum(self.res) > 1:
            msg = (
                "More than one check is true: {}. Processing workflow "
                "is unclear.".format(
                    ", ".join([a[0] for (a, b) in zip(self.checklist, self.res) if b])
                )
            )
            raise PostProcessException(msg)
        else:
            return self.res

    def check_case_no_data(self):
        """
        Checks if either no stream, or no station metadata exist.
        """
        if len(self.st) < 1 or len(self.inv_dt) < 1:
            return True
        elif (
            len(self.inv_dt[0][0]) == 1
            and self.inv_dt[0][0][0].start_date == self.enddate
        ):
            return True
        elif (
            len(self.inv_dt[0][0]) == 1
            and self.inv_dt[0][0][0].end_date == self.startdate
        ):
            return True
        else:
            return False

    def output_case_no_data(self):
        raise PostProcessException

    def check_case_multiple_channel_periods(self):
        """
        Checks if there are multiple channel operation periods in a given
        time period. If so, different response removal is required.
        """
        if len(self.inv_dt) < 1 or len(self.st) < 1:
            return False
        elif len(self.inv_dt[0][0]) > 1:
            return True
        else:
            return False

    def output_case_multiple_channel_periods(self):
        """
        Stitches two response periods together to make a single merged
        trace.
        """
        if len(self.inv_dt[0][0]) > 2:
            raise NotImplementedError(
                "Current post processing function only valid for a "
                "maximum of two response periods."
            )
        st1 = self.st.copy()
        st1.merge(fill_value=self.fill_value)
        st2 = st1.copy()
        st1.trim(
            starttime=self.startdate,
            endtime=self.inv_dt[0][0][0].end_date,
            nearest_sample=False,
        )
        st2.trim(
            starttime=self.inv_dt[0][0][1].start_date,
            endtime=self.enddate,
            nearest_sample=False,
        )
        st1.attach_response(self.inv_dt)
        st1.remove_sensitivity()
        st2.attach_response(self.inv_dt)
        st2.remove_sensitivity()
        st1 += st2
        st1.merge(fill_value=self.fill_value)
        st1.trim(
            self.startdate,
            self.enddate,
            nearest_sample=False,
            pad=True,
            fill_value=self.fill_value,
        )
        return st1[0]

    def check_case_incomplete_station_metadata(self):
        """
        The case where a a channel period start/stops during the
        selected time window. Some parts of the trace in the stream
        might not correspond to a valid period according to station
        metadata.
        """
        if len(self.inv_dt) < 1:
            return False
        elif len(self.inv_dt[0][0]) != 1 or len(self.st) < 1:
            return False
        elif (self.startdate < self.inv_dt[0][0][0].start_date < self.enddate) or (
            self.startdate < self.inv_dt[0][0][0].end_date < self.enddate
        ):
            return True
        else:
            return False

    def output_case_incomplete_station_metadata(self):
        if self.inv_dt[0][0][0].start_date > self.startdate:
            self.st.trim(
                self.inv_dt[0][0][0].start_date, self.enddate, nearest_sample=False
            )
            self.st.attach_response(self.inv_dt)
            self.st.remove_sensitivity()
            self.st.trim(
                self.startdate,
                self.enddate,
                nearest_sample=False,
                pad=True,
                fill_value=self.fill_value,
            )
            return self.st[0]
        elif self.inv_dt[0][0][0].end_date < self.enddate:
            self.st.trim(
                self.startdate, self.inv_dt[0][0][0].end_date, nearest_sample=False
            )
            self.st.attach_response(self.inv_dt)
            self.st.remove_sensitivity()
            self.st.trim(
                self.startdate,
                self.enddate,
                nearest_sample=False,
                pad=True,
                fill_value=self.fill_value,
            )
            return self.st[0]
        else:
            raise PostProcessException(
                "Post-procesing function not equipped to" "handle:\n" "{}".format(
                    self.st
                )
            )

    def check_case_no_station_issues(self):
        if len(self.inv_dt) < 1:
            return False
        elif len(self.inv_dt[0][0]) != 1 or len(self.st) < 1:
            return False
        elif (
            len(self.inv_dt[0][0]) == 1
            and self.inv_dt[0][0][0].end_date == self.startdate
        ):
            return False
        elif (
            len(self.inv_dt[0][0]) == 1
            and self.inv_dt[0][0][0].start_date == self.enddate
        ):
            return False
        elif (self.startdate < self.inv_dt[0][0][0].start_date < self.enddate) or (
            self.startdate < self.inv_dt[0][0][0].end_date < self.enddate
        ):
            return False
        else:
            return True

    def output_case_no_station_issues(self):
        self.st.merge(fill_value=self.fill_value)
        self.st.trim(
            starttime=self.startdate,
            endtime=self.enddate,
            nearest_sample=False,
            pad=True,
            fill_value=self.fill_value,
        )
        self.st.attach_response(self.inv_dt)
        self.st.remove_sensitivity()
        return self.st[0]


def get_instrument_response(
    net,
    site,
    loc,
    staxml_dir="./instrument_response",
    fdsn_urls=("https://service.geonet.org.nz"),
):
    """
    Retrieve instrument response file in STATIONXML format.
    """
    fn = "{:s}.xml".format(site)
    fn = os.path.join(staxml_dir, fn)
    try:
        inv = read_inventory(fn, format="STATIONXML")
    except FileNotFoundError:
        try:
            os.makedirs(staxml_dir)
        except FileExistsError:
            pass
        for url in fdsn_urls:
            try:
                client = FDSN_Client(base_url=url)
                inv = client.get_stations(
                    network=net,
                    location=loc,
                    station=site,
                    channel="*",
                    level="response",
                )
            except Exception as e:
                logger.info(e)
            else:
                break
        inv.write(fn, format="STATIONXML")
    return inv


class WaveformBaseclass:
    pass


class SDSWaveforms(WaveformBaseclass):
    def __init__(self, sds_dir, fdsn_urls, staxml_dir, fill_value=np.nan):
        """
        Get seismic waveforms from a local SDS archive.
        """
        self.client = SDS_Client(sds_dir)
        self.fdsn_urls = fdsn_urls
        self.staxml_dir = staxml_dir
        self.fill_value = fill_value

    def get_waveforms(self, net, site, loc, comp, startdate, enddate):
        """
        :param fill_value: Default value used to fill gaps and pad traces to
                           cover the whole time period defined by `startdate`
                           and `enddate`. If 'interpolate' is chosen, the
                           traces will not be padded.
        :type fill_value: int, float, 'interpolate', np.nan, or None
        """
        inv = get_instrument_response(
            net, site, loc, staxml_dir=self.staxml_dir, fdsn_urls=self.fdsn_urls
        )

        st = self.client.get_waveforms(
            net, site, loc, comp, startdate, enddate, dtype="float64"
        )

        inv_dt = inv.select(
            location=loc, channel=comp, starttime=startdate, endtime=enddate - 1
        )

        # Post-process to remove sensitivity and account for gaps
        pp = PostProcess(
            st=st,
            inv=inv,
            inv_dt=inv_dt,
            startdate=startdate,
            enddate=enddate,
            loc=loc,
            comp=comp,
            fill_value=self.fill_value,
        )
        tr = pp.run_post_processing()
        return tr


class MockSDSWaveforms(WaveformBaseclass):
    """
    Mock SDSWaveforms class for testing by creating
    synthetic data for the requested streams.
    """

    def __init__(self, sds_dir):
        """
        Get seismic waveforms from a local SDS archive.
        """
        os.makedirs(sds_dir, exist_ok=True)
        self.client = SDS_Client(sds_dir)

    def save2sds(self, trace: Trace, rootdir: str):
        """
        Save a trace to a SDS directory structure.

        Parameters
        ----------
        trace : `obspy.Trace`
            Seismic/acoustic trace to be written to disk.
        rootdir : str
            Root directory for the SDS directory structure.
        """
        sds_fmtstr = os.path.join(
            "{year}",
            "{network}",
            "{station}",
            "{channel}.{sds_type}",
            "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}",
        )

        fullpath = sds_fmtstr.format(
            year=trace.stats.starttime.year,
            doy=trace.stats.starttime.julday,
            sds_type="D",
            **trace.stats,
        )
        fullpath = os.path.join(rootdir, fullpath)
        dirname, filename = os.path.split(fullpath)
        os.makedirs(dirname, exist_ok=True)
        print("writing", fullpath)
        trace.write(fullpath, format="mseed")

    def generate_dataset(
        self,
        rootdir: str,
        net: str,
        site: str,
        loc: str,
        comp: str,
        start: str,
        end: str,
    ) -> str:
        """
        Generate a test dataset for the integration tests.

        Parameters
        ----------
        rootdir : str
            Parent directory for the test dataset.
        start : str
            Start time for the test dataset in ISO 8601 format.
        end : str
            End time for the test dataset in ISO 8601 format.
        """
        tstart = UTCDateTime(start)
        # Always generate whole day files
        _tstart = UTCDateTime(year=tstart.year, julday=tstart.julday)
        tend = UTCDateTime(end)
        tr = test_signal(
            starttime=_tstart,
            sampling_rate=10,
            nsec=86400,
            gaps=True,
            network=net,
            station=site,
            location=loc,
            channel=comp,
        )
        while _tstart < tend:
            tr.stats.starttime = _tstart
            self.save2sds(tr, rootdir)
            _tstart += 86400

    def get_waveforms(self, net, site, loc, comp, startdate, enddate):
        """
        Generate synthetic daily files that start at midnight on the
        startdate and end at midnight on the enddate. The return the requested
        timespan from these files.
        """
        self.generate_dataset(
            self.client.sds_root, net, site, loc, comp, startdate, enddate
        )
        st = self.client.get_waveforms(
            net, site, loc, comp, startdate, enddate, dtype="float64"
        )
        return st[0]


class FDSNWaveforms(WaveformBaseclass):
    def __init__(self, url, debug=False, fill_value=np.nan):
        """
        Get seismic waveforms from FDSN web service.
        """
        self.client = FDSN_Client(base_url=url, debug=debug)
        self.fill_value = fill_value

    def get_waveforms(self, net, site, loc, comp, startdate, enddate):
        st = self.client.get_waveforms(
            net, site, loc, comp, startdate, enddate, attach_response=True
        )
        # Prepare seismic time series
        st.remove_sensitivity()
        # in case stream has more than one trace
        st.merge(fill_value=self.fill_value)
        if self.fill_value != "interpolate":
            st.trim(
                startdate,
                enddate,
                pad=True,
                fill_value=self.fill_value,
                nearest_sample=False,
            )
        return st[0]


class S3Waveforms(WaveformBaseclass):
    def __init__(self, s3bucket, fdsn_urls, staxml_dir, debug=False, fill_value=np.nan):
        config = Config(signature_version=UNSIGNED)
        self.client = boto3.client("s3", config=config)
        self.s3_bucket = s3bucket
        self.fdsn_urls = fdsn_urls
        self.staxml_dir = staxml_dir
        self.debug = debug
        self.fill_value = fill_value

    def get_waveforms(self, net, site, loc, comp, start, end):
        """
        Get seismic waveforms from GeoNet's open data archive on AWS S3.
        """
        PATH_FORMAT = "waveforms/miniseed/{year}/{year}.{julday:03d}/{station}."
        PATH_FORMAT += "{network}/{year}.{julday:03d}.{station}."
        PATH_FORMAT += "{location}-{channel}.{network}.D"
        t_start = start
        t_end = UTCDateTime(
            year=t_start.year,
            julday=t_start.julday + 1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        t_end = min(t_end, end)
        st = Stream()
        while t_start < end:
            s3_file_path = PATH_FORMAT.format(
                year=t_start.year,
                julday=t_start.julday,
                station=site,
                network=net,
                location=loc,
                channel=comp,
            )
            data = io.BytesIO()
            if self.debug:
                logger.debug("Requesting {}".format(s3_file_path))
            _ = self.client.download_fileobj(
                Bucket=self.s3_bucket, Key=s3_file_path, Fileobj=data
            )
            data.seek(0)
            _st = read(data, dtype="float64")
            data.close()
            _st.trim(t_start, t_end, nearest_sample=False)
            st += _st
            t_start = t_end
            t_end = t_start + timedelta(days=1)
            t_end = min(t_end, end)

        st.merge(fill_value=self.fill_value)
        inv = get_instrument_response(
            net, site, loc, staxml_dir=self.staxml_dir, fdsn_urls=self.fdsn_urls
        )
        inv_dt = inv.select(
            location=loc, channel=comp, starttime=t_start, endtime=t_end
        )

        # Post-process to remove sensitivity and account for gaps
        pp = PostProcess(
            st=st,
            inv=inv,
            inv_dt=inv_dt,
            startdate=start,
            enddate=end,
            loc=loc,
            comp=comp,
            fill_value=self.fill_value,
        )
        tr = pp.run_post_processing()
        return tr


def tilde_request(
    domain: str,
    name: str,
    station: str,
    method: str,
    sensor: str,
    aspect: str,
    startdate: datetime,
    enddate: datetime,
) -> pd.DataFrame:
    """
    Request data from the tilde API (https://tilde.geonet.org.nz/v3/api-docs/).
    See the tilde discovery tool for more information:
    https://tilde.geonet.org.nz/ui/data-discovery/

    Parameters
    ----------
    domain : str
        The domain of the data (e.g. 'manualcollect')
    name : str
        The name of the data (e.g. 'plume-SO2-gasflux')
    station : str
        The station code (e.g. 'WI000')
    method : str
        The method of the data (e.g. 'contouring')
    sensor : str
        The sensor of the data (e.g. 'MC01')
    aspect : str
        The aspect of the data (e.g. 'nil')
    startdate : date
        The start date of the data
    enddate : date
        The end date of the data

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the requested data
    """
    tilde_url = "https://tilde.geonet.org.nz/v3/data"
    # split the request into historic and latest data
    get_historic = True
    get_latest = False
    startdate = startdate.astimezone(timezone.utc)
    enddate = enddate.astimezone(timezone.utc)
    _tstart = str(startdate.date())
    _tend = str(enddate.date())
    _today = datetime.now(timezone.utc).date()
    if enddate.date() > (_today - timedelta(days=29)):
        _tend = str((enddate.date() - timedelta(days=29)))
        get_latest = True
    if startdate.date() > (_today - timedelta(days=29)):
        get_historic = False

    assert get_historic or get_latest, "Check start and end dates."

    if get_latest:
        latest = f"{tilde_url}/{domain}/{station}/{name}/{sensor}/{method}/{aspect}/latest/30d"
        rval = requests.get(latest, headers={"Accept": "text/csv"})
        if rval.status_code != 200:
            msg = f"Download of {name} for {station} failed with status code {rval.status_code}"
            msg += f" and url {latest}"
            raise ValueError(msg)
        df_latest = pd.read_csv(
            io.StringIO(rval.text),
            index_col="timestamp",
            parse_dates=["timestamp"],
            usecols=["timestamp", "value", "error"],
            date_format="ISO8601",
        )
    if get_historic:
        historic = f"{tilde_url}/{domain}/{station}/{name}/{sensor}/{method}/{aspect}/"
        historic += f"{_tstart}/{_tend}"
        rval = requests.get(historic, headers={"Accept": "text/csv"})
        if rval.status_code != 200:
            msg = f"Download of {name} for {station} failed with status code {rval.status_code}"
            msg += f" and url {historic}"
            raise ValueError(msg)
        data = io.StringIO(rval.text)
        df_historic = pd.read_csv(
            data,
            index_col="timestamp",
            parse_dates=["timestamp"],
            usecols=["timestamp", "value", "error"],
            date_format="ISO8601",
        )
        if get_latest and len(df_latest) > 0:
            df = df_historic.combine_first(df_latest)
        else:
            df = df_historic
    else:
        df = df_latest
    df.rename(columns={"value": "obs", "error": "err"}, inplace=True)
    df.index.name = "dt"
    return df.loc[startdate:enddate]


class GeoMagWaveforms(WaveformBaseclass):
    def __init__(self, base_url, method: str, aspect: str, name: str):
        self.base_url = base_url
        self.method = method
        self.aspect = aspect
        self.name = name
        self.domain = "geomag"

    def get_waveforms(
        self, net: str, site: str, loc: str, comp: str, startdate: date, enddate: date
    ) -> pd.DataFrame:
        df_ = tilde_request(
            domain=self.domain,
            name=self.name,
            station=site,
            method=self.method,
            sensor=loc,
            aspect=self.aspect,
            startdate=startdate.datetime,
            enddate=enddate.datetime,
        )

        df = df_.reindex(
            pd.date_range(start=df_.index[0], end=df_.index[-1], freq="60s")
        )
        df.fillna(method="ffill", inplace=True)
        stats = {
            "network": net,
            "station": site,
            "location": loc,
            "channel": comp,
            "npts": len(df),
            "sampling_rate": 1 / 60,
        }
        stats["starttime"] = UTCDateTime(df.index[0])
        tr = Trace(data=df["obs"].values, header=stats)
        return tr


class DataSource:
    def __init__(self, clients, chunk_size=None, cache_dir=None):
        self.chunk_size = chunk_size

        if len(clients) < 1:
            msg = "At least one data client is required."
            raise ValueError(msg)

        # Test that all waveform clients are derived from WaveformBaseclass
        for client in clients:
            if not issubclass(client.__class__, WaveformBaseclass):
                msg = "All clients must be derived from WaveformBaseclass."
                raise ValueError(msg)

        self.clients = clients
        self.cache_dir = os.path.join(os.environ["HOME"], "zizou_cache")
        if cache_dir is not None:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_client = SDS_Client(self.cache_dir)

    def to_sds(self, tr: Trace):
        """
        Save trace to SDS directory.
        """
        start = tr.stats.starttime
        end = tr.stats.endtime
        sds_fmtstr = os.path.join(
            "{year}",
            "{network}",
            "{station}",
            "{channel}.{sds_type}",
            "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}",
        )

        current_date = start.date
        while current_date <= end.date:
            _tr = tr.slice(UTCDateTime(current_date), UTCDateTime(current_date) + 86400)
            # print(_tr.stats.starttime, _tr.stats.endtime)
            fullpath = sds_fmtstr.format(
                year=_tr.stats.starttime.year,
                doy=_tr.stats.starttime.julday,
                sds_type="D",
                **_tr.stats,
            )
            fullpath = os.path.join(self.cache_dir, fullpath)
            dirname, _ = os.path.split(fullpath)
            os.makedirs(dirname, exist_ok=True)
            logger.debug(f"writing {_tr} to {fullpath}")
            _tr.write(fullpath, format="MSEED")
            current_date += timedelta(days=1)

    def get_waveforms(self, net, site, loc, comp, start, end, cache=False):
        t_start = start
        t_end = end
        if self.chunk_size is not None:
            t_end = t_start + self.chunk_size
            t_end = min(t_end, end)
        while t_end <= end:
            tr = Trace()
            try:
                # First check the cache
                st = self.cache_client.get_waveforms(
                    net, site, loc, comp, t_start, t_end, dtype="float64"
                )
                tr = st.merge(fill_value=np.nan)[0]
                t_diff = int(t_end - t_start)
                if abs(t_diff - (tr.stats.npts - 1) * tr.stats.delta) > 1:
                    raise IndexError
                tr.stats["cached"] = True
            except (IndexError, AttributeError):
                msg = "Data for {} between {} and {} not found in cache."
                logger.debug(
                    msg.format(".".join((net, site, loc, comp)), t_start, t_end)
                )
                for client in self.clients:
                    try:
                        tr = client.get_waveforms(net, site, loc, comp, t_start, t_end)
                    except Exception as e:
                        logger.info(e)
                        continue
                    else:
                        break
                if not tr:
                    msg = "No data found for {}"
                    logger.error(msg.format(".".join((net, site, loc, comp))))
                else:
                    if cache:
                        self.to_sds(tr)
            yield tr

            if self.chunk_size is not None:
                t_start = t_end
                cond1 = (end - t_end) < self.chunk_size
                cond2 = end > t_end
                if cond1 and cond2:
                    last_chunk = end - t_end
                    t_end = t_start + last_chunk
                else:
                    t_end = t_start + self.chunk_size
            else:
                # go beyond the end to stop the loop
                t_end = end + 1.0


class VolcanoMetadata:
    """
    Read volcano/station/channel metadata
    """

    def __init__(self, configfile):
        try:
            with open(configfile, "r") as fh:
                cfg = yaml.safe_load(fh)
        except OSError:
            cfg = yaml.safe_load(configfile)
        self.data = cfg["metadata"]

    def get_available_volcanoes(self):
        return [vol["name"] for vol in self.data["volcano"]]

    def get_available_streams(self, volcano):
        """
        Get available sensor streams for a given volcano.

        Parameters
        ----------
        volcano : str
            Volcano name

        Returns
        -------
        list
            List of streams in the format 'NET.STA.LOC.CHAN'
        """
        _streams = []
        for _v in self.data["volcano"]:
            if _v["name"].lower() == volcano.lower():
                for _n in _v["network"]:
                    net_code = _n["net_code"]
                    for _s in _n["stations"]:
                        sta_code = _s["sta_code"]
                        location = _s["location"]
                        for _c in _s["channel"]:
                            chan_code = _c["code"]
                            _stream = ".".join(
                                (net_code, sta_code, location, chan_code)
                            )
                            _streams.append(_stream)
        return _streams

    def get_site_information(self, sitename):
        """
        Get site information for a given site. Note
        that if the same site name exists at multiple
        volcanoes, the first match will be returned.

        Parameters
        ----------
        sitename : str
            Three-letter site code.

        Returns
        -------
        dict
            Dictionary with site information as follows:
            {'volcano': str, 'site': str, 'channels': list,
             'latitude': float, 'longitude': float, 'starttime': datetime}
        """
        for _v in self.data["volcano"]:
            for _n in _v["network"]:
                for _s in _n["stations"]:
                    if _s["sta_code"] == sitename:
                        try:
                            starttime = datetime.strptime(
                                _s["starttime"], "%Y-%m-%dT%H:%M:%SZ"
                            )
                        except KeyError:
                            starttime = None

                        try:
                            channels = [c["code"] for c in _s["channel"]]
                        except KeyError:
                            channels = []
                        return dict(
                            volcano=_v["name"],
                            site=_s["sta_code"],
                            channels=channels,
                            latitude=float(_s["latitude"]),
                            longitude=float(_s["longitude"]),
                            starttime=starttime,
                        )
        return None

    def get_eruption_dates(self, volcano):
        """
        Get eruption dates for a given volcano.

        Parameters
        ----------
        volcano : str
            Volcano name.

        Returns
        -------
        list
            List of datetime.date objects representing eruption dates.
        """
        eruption_dates = []
        metadata = [v for v in self.data["volcano"] if v["name"] == volcano]
        er = metadata[0]["eruptions"]
        if er:
            for e in er:
                d, m, y = e.split("-")
                eruption_dates.append(date(int(y), int(m), int(d)))
        return eruption_dates

    def get_unrest_periods(self, volcano):
        """
        Get unrest periods for a given volcano.

        Parameters
        ----------
        volcano : str
            Volcano name.

        Returns
        -------
        list
            List of datetime.date objects representing start and end dates.
        """
        unrest_dates = []
        metadata = [v for v in self.data["volcano"] if v["name"] == volcano]
        ur = metadata[0]["unrest periods"]
        if ur:
            for u in ur:
                ds, ms, ys = u["starttime"].split("-")
                de, me, ye = u["endtime"].split("-")
                unrest_dates.append(
                    [date(int(ys), int(ms), int(ds)), date(int(ye), int(me), int(de))]
                )
        return unrest_dates


if __name__ == "__main__":
    import doctest

    doctest.testmod()
