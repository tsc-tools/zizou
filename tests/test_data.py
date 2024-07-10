import datetime
import inspect
from io import StringIO
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, Mock

import pytest
import boto3
from moto import mock_s3
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read_inventory, Trace
import xarray as xr
from zizou.util import xarray2hdf5, generate_test_data, test_signal


from zizou.data import (DataSource,
                       VolcanoMetadata,
                       FeatureRequest,
                       PostProcess,
                       PostProcessException)
from zizou import get_data


class DataTestCase(unittest.TestCase):

    def get_max(self, trace):
        value = np.nanmax(trace.data)
        _min = np.nanmin(trace.data)
        if abs(_min) > abs(value):
            value = _min
        return value

    def setUp(self):
        filename = inspect.getfile(inspect.currentframe())
        filedir = os.path.dirname(os.path.abspath(filename))
        self.sds_dir = os.path.join(filedir, "data", "sds_test")

    def test_exceptions(self):
        """
        Test that the right exceptions are raised.
        """
        with self.assertRaises(ValueError):
            ds = DataSource(sources=())

        with self.assertRaises(ValueError):
            ds = DataSource(sources=('blub', 'blab'))

        with self.assertRaises(ValueError):
            ds = DataSource(fdsn_urls=('/foo/bar'))

    @unittest.skip("Needs rewrite as it's testing a real FDSN server")
    def test_get_waveforms(self):
        """
        Test that the waveforms returned from SDS and FDSNws are
        consistent.
        """
        ds1 = DataSource(sources=('fdsn',))
        ds2 = DataSource(sources=('sds',), sds_dir=self.sds_dir,
                         staxml_dir=self.sds_dir)
        # dates are inclusive, and complete days
        starttime = UTCDateTime(year=2013, julday=240)
        endtime = starttime + 86400.
        for tr1 in ds1.get_waveforms('NZ', 'WIZ', '10', 'HHZ', starttime,
                                     endtime):
            for tr2 in ds2.get_waveforms('NZ', 'WIZ', '10', 'HHZ', starttime,
                                         endtime):
                self.assertEqual(tr1.max(), self.get_max(tr2))
                self.assertEqual(tr1.stats.npts, tr2.stats.npts)

    def test_with_gaps(self):
        """
        Test behaviour of traces with gaps.
        """
        ds1 = DataSource(sources=('sds',), sds_dir=self.sds_dir,
                         staxml_dir=self.sds_dir)
        starttime = UTCDateTime(2010, 11, 26, 0, 0, 0)
        endtime = starttime + 86400.
        stream = 'NZ.WIZ.10.HHZ'
        net, site, loc, comp = stream.split('.')
        gen = ds1.get_waveforms(net, site, loc, comp, starttime, endtime)
        t1 = next(gen)
        self.assertTrue(np.any(np.isnan(t1.data)))
        idx = np.where(np.isnan(t1.data))
        self.assertEqual(idx[0].size, 431238)

        starttime2 = UTCDateTime(2010, 11, 26, 0, 0, 0)
        endtime2 = starttime2 + 86400.
        gen2 = ds1.get_waveforms(net, site, loc, comp, starttime, endtime)
        t2 = next(gen2)

        # Check that we can take a trace apart and put it
        # together again
        t3 = t2.copy()
        mdata = np.ma.masked_invalid(t3.data)
        t3.data = mdata
        st = t3.split()
        st.merge(fill_value=np.nan)
        st.trim(starttime2, endtime2, pad=True, fill_value=np.nan,
                nearest_sample=False)
        idx = np.where(st[0].data != t2.data)
        # data 'differs' only at NaNs
        self.assertTrue(np.alltrue(np.isnan(st[0].data[idx])))

    @unittest.skip("Needs rewrite as it's testing a real FDSN server")
    def test_start_endtime(self):
        """
        Test that start and end times are what we expect.
        """
        ds = DataSource(sources=('sds',), sds_dir=self.sds_dir,
                        staxml_dir=self.sds_dir)
        starttime = UTCDateTime(year=2013, julday=240)
        endtime = starttime + 86400.
        stream = 'NZ.WIZ.10.HHZ'
        net, site, loc, comp = stream.split('.')
        gen = ds.get_waveforms(net, site, loc, comp, starttime, endtime)
        tr = next(gen)
        self.assertTrue(tr.stats.starttime >= starttime)
        self.assertTrue(tr.stats.endtime <= endtime)
        ds1 = DataSource(sources=('fdsn',))
        gen1 = ds1.get_waveforms(net, site, loc, comp, starttime, endtime)
        tr1 = next(gen1)
        self.assertTrue(tr1.stats.starttime >= starttime)
        self.assertTrue(tr1.stats.endtime <= endtime)
 
    @pytest.mark.slow
    def test_missing_station_metadata_sds(self):
        """
        At NTVZ, station metadata stops at
        2019-04-17T01:10:00.000000Z and restarts at
        2019-04-19T01:30:00.000000Z, but waveform data
        still exist throughout.

        Full get_waveform test across three days, which checks that traces
        are available for the time periods where there are both station
        and waveform data, np.nan otherwise.
        """
        ds = DataSource(sources=('sds',), sds_dir=self.sds_dir,
                        staxml_dir=self.sds_dir, chunk_size=86400.)
        starttime = UTCDateTime(2019, 4, 17)
        endtime = UTCDateTime(2019, 4, 20)
        net, site, loc, comp = ('NZ', 'NTVZ', '10', 'HHZ')
        gen = ds.get_waveforms(net, site, loc, comp, starttime, endtime)
        # number of non-NaN entries in each trace
        # 2019-04-17
        tr1 = next(gen)
        self.assertEqual(np.count_nonzero(~np.isnan(tr1.data)), 371902)
        # 2019-04-18
        tr2 = next(gen)
        self.assertEqual(np.count_nonzero(~np.isnan(tr2.data)), 0)
        # 2019-04-19
        tr3 = next(gen)
        self.assertEqual(np.count_nonzero(~np.isnan(tr3.data)), 8100000)
        
    @pytest.mark.slow
    @mock_s3
    def test_aws_backend(self):
        """
        Test retrieving waveform data from AWS S3.
        """
        ### Setup virtual S3
        bucket_name = 'virtual-geonet-open-data'
        client = boto3.client('s3', region_name='us-east-1',
                              aws_access_key_id='fake_access_key',
                              aws_secret_access_key='fake_secret_key')
        client.create_bucket(Bucket=bucket_name)
        starttime1 = UTCDateTime(2019, 4, 17, 11, 0, 17)
        endtime1 = UTCDateTime(2019, 4, 17, 23, 59, 59, 999999)
        starttime2 = UTCDateTime(2019, 4, 18, 0, 0, 0)
        endtime2 = UTCDateTime(2019, 4, 18, 3, 4, 5)
        key1 = "waveforms/miniseed/2019/2019.107/MAVZ.NZ/2019.107.MAVZ.10-HHZ.NZ.D"
        key2 = "waveforms/miniseed/2019/2019.108/MAVZ.NZ/2019.108.MAVZ.10-HHZ.NZ.D"
        test_data1 = test_signal(starttime=starttime1,
                                nsec=endtime1-starttime1,
                                station='MAVZ',
                                location='10') 
        test_data2 = test_signal(starttime=starttime2,
                                nsec=endtime2-starttime2,
                                station='MAVZ',
                                location='10') 
        filename = tempfile.mktemp()
        test_data1.write(filename, format='MSEED')
        client.upload_file(Filename=filename, Bucket=bucket_name, Key=key1)
        filename = tempfile.mktemp()
        test_data2.write(filename, format='MSEED')
        client.upload_file(Filename=filename, Bucket=bucket_name, Key=key2)
        
        ds = DataSource(staxml_dir=self.sds_dir, sources=('s3',),
                        s3bucket=bucket_name)
        net, site, loc, comp = ('NZ', 'MAVZ', '10', 'HHZ')
        gen = ds.get_waveforms(net, site, loc, comp, starttime1,
                               endtime2)
        tr1 = next(gen)
        test_data1.trim(starttime1, starttime1 + datetime.timedelta(days=1))
        test_inv = read_inventory(os.path.join(self.sds_dir, 'MAVZ.xml'))
        test_data1.attach_response(test_inv)
        test_data1.remove_sensitivity()
        self.assertEqual(test_data1.data.max(), np.nanmax(tr1.data))
        with self.assertRaises(StopIteration):
            tr2 = next(gen)
        self.assertEqual(tr1.stats.starttime, starttime1)
        self.assertEqual(tr1.stats.endtime, endtime2)
        gen1 = ds.get_waveforms(net, 'WIZ', loc, comp, starttime1,
                                endtime2)
        tr3 = next(gen1)
        # Check that the process returned a dummy trace
        self.assertEqual(len(tr3), 0)

    def test_mock_client(self):
        """
        Test that the mock client is working.
        """
        sds_dir = tempfile.mkdtemp()
        ds = DataSource(sds_dir=sds_dir, sources=['mock'])
        net, site, loc, comp = ('NZ', 'MAVZ', '10', 'HHZ')
        starttime = UTCDateTime(2024, 4, 11, 3, 0, 0)
        endtime = UTCDateTime(2024, 4, 11, 4, 0, 0)
        gen = ds.get_waveforms(net, site, loc, comp, starttime, endtime)
        tr = next(gen)
        self.assertEqual(tr.stats.starttime, starttime)
        self.assertEqual(tr.stats.endtime, endtime)
       

class PostProcessingTestCase(unittest.TestCase):
    """
    Tests various permutations of station metadata availability e.g. at
    MAVZ:
    [Channel 'HHZ', Location '10',
     Time range: 2012-05-22T02:00:00.000000Z - 2013-02-26T00:00:00.000000Z,
     Channel 'HHZ', Location '10',
     Time range: 2013-03-22T00:00:00.000000Z - 2013-04-09T22:15:00.000000Z,
     Channel 'HHZ', Location '10',
     Time range: 2013-04-10T00:15:00.000000Z - 2015-01-09T00:00:00.000000Z,
     Channel 'HHZ', Location '10',
     Time range: 2015-01-09T00:00:01.000000Z - 2017-04-20T22:00:00.000000Z,
     Channel 'HHZ', Location '10',
     Time range: 2017-04-20T22:00:02.000000Z - 9999-01-01T00:00:00.000000Z,
    ]
    """
    def setUp(self):
        filename = inspect.getfile(inspect.currentframe())
        filedir = os.path.dirname(os.path.abspath(filename))
        self.sds_dir = os.path.join(filedir, "data", "sds_test")
        self.ds = DataSource(sources=('sds',), sds_dir=self.sds_dir,
                             staxml_dir=self.sds_dir)

    def test_function_list(self):
        """
        Tests whether there is a matching number of tests and
        post-processing functions
        """
        pp = PostProcess()
        check_names = [check_n for (check_n, check) in pp.checklist]
        func_names = [func_n for (func_n, func) in pp.pp_functions]
        # Check correct number of checks and output function
        self.assertEqual(len(check_names), len(func_names))
        # Check checks and output functions match
        [self.assertEqual(c.replace('check', 'output'), f)
         for (c, f) in zip(check_names, func_names)]

    def test_case_single_trace_in_stream(self):
        net, site, loc, comp = ('NZ', 'MAVZ', '10', 'HHZ')
        startdate = UTCDateTime(2013, 2, 25)
        enddate = UTCDateTime(2013, 2, 26)
        st = self.ds.clients[0].client.get_waveforms(net, site, loc, comp, startdate,
                                                     enddate, dtype='float64')
        inv = read_inventory(os.path.join(self.sds_dir, 'MAVZ.xml'),
                             format='STATIONXML')
        inv_dt = inv.select(location=loc, channel=comp,
                            starttime=startdate, endtime=enddate)
        pp = PostProcess(st=st, inv=inv, inv_dt=inv_dt, loc=loc,
                         comp=comp, fill_value=np.nan, startdate=startdate,
                         enddate=enddate)
        pp.run_checks()
        # Check correct test case:
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.checklist, pp.res) if b],
            ["check_case_no_station_issues"]
        )
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.pp_functions, pp.res) if b],
            ["output_case_no_station_issues"]
        )
        # Check output
        test_st = st.copy()
        test_st.attach_response(inv_dt)
        test_st.remove_sensitivity()
        test_st.trim(
            starttime=startdate, endtime=enddate,
            nearest_sample=False, pad=True, fill_value=np.nan
        )
        tr = pp.run_post_processing()
        np.testing.assert_array_almost_equal(tr.data, test_st[0].data)

    def test_multiple_traces_in_stream(self):
        net, site, loc, comp = ('NZ', 'MAVZ', '10', 'HHZ')
        startdate = UTCDateTime(2012, 5, 23)
        enddate = UTCDateTime(2012, 5, 24)
        st = self.ds.clients[0].client.get_waveforms(net, site, loc, comp, startdate,
                                                     enddate, dtype='float64')
        inv = read_inventory(os.path.join(self.sds_dir, 'MAVZ.xml'),
                             format='STATIONXML')
        inv_dt = inv.select(location=loc, channel=comp,
                            starttime=startdate, endtime=enddate)
        pp = PostProcess(st=st, inv=inv, inv_dt=inv_dt, loc=loc,
                         comp=comp, fill_value=np.nan, startdate=startdate,
                         enddate=enddate)
        pp.run_checks()

        self.assertEqual(
            [a[0] for (a, b) in zip(pp.checklist, pp.res) if b],
            ["check_case_no_station_issues"]
        )
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.pp_functions, pp.res) if b],
            ["output_case_no_station_issues"]
        )
        tr = pp.run_post_processing()
        self.assertEqual(int(tr.stats.npts), int(8640000))

    def test_case_stream_starttime_equals_channel_endtime(self):
        net, site, loc, comp = ('NZ', 'MAVZ', '10', 'HHZ')
        startdate = UTCDateTime(2013, 2, 26)
        enddate = UTCDateTime(2013, 2, 27)
        st = self.ds.clients[0].client.get_waveforms(net, site, loc, comp, startdate,
                                                     enddate, dtype='float64')
        inv = read_inventory(os.path.join(self.sds_dir, 'MAVZ.xml'),
                             format='STATIONXML')
        inv_dt = inv.select(location=loc, channel=comp,
                            starttime=startdate, endtime=enddate)
        pp = PostProcess(st=st, inv=inv, inv_dt=inv_dt, loc=loc,
                         comp=comp, fill_value=np.nan, startdate=startdate,
                         enddate=enddate)
        pp.run_checks()

        # Check correct test case:
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.checklist, pp.res) if b],
            ["check_case_no_data"]
        )
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.pp_functions, pp.res) if b],
            ["output_case_no_data"]
        )
        with self.assertRaises(PostProcessException):
            pp.run_post_processing()

    def test_case_stream_endtime_equals_channel_starttime(self):
        net, site, loc, comp = ('NZ', 'MAVZ', '10', 'HHZ')
        startdate = UTCDateTime(2013, 3, 21)
        enddate = UTCDateTime(2013, 3, 22)
        st = self.ds.clients[0].client.get_waveforms(net, site, loc, comp, startdate,
                                                     enddate, dtype='float64')
        inv = read_inventory(os.path.join(self.sds_dir, 'MAVZ.xml'),
                             format='STATIONXML')
        inv_dt = inv.select(location=loc, channel=comp,
                            starttime=startdate, endtime=enddate)
        pp = PostProcess(st=st, inv=inv, inv_dt=inv_dt, loc=loc,
                         comp=comp, fill_value=np.nan, startdate=startdate,
                         enddate=enddate)
        pp.run_checks()

        # Check correct test case:
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.checklist, pp.res) if b],
            ["check_case_no_data"]
        )
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.pp_functions, pp.res) if b],
            ["output_case_no_data"]
        )
        with self.assertRaises(PostProcessException):
            pp.run_post_processing()

    def test_case_multiple_channel_periods(self):
        net, site, loc, comp = ('NZ', 'MAVZ', '10', 'HHZ')
        startdate = UTCDateTime(2015, 1, 9)
        enddate = UTCDateTime(2015, 1, 10)
        st = self.ds.clients[0].client.get_waveforms(net, site, loc, comp, startdate,
                                                     enddate, dtype='float64')
        inv = read_inventory(os.path.join(self.sds_dir, 'MAVZ.xml'),
                             format='STATIONXML')
        inv_dt = inv.select(location=loc, channel=comp,
                            starttime=startdate, endtime=enddate)
        pp = PostProcess(st=st, inv=inv, inv_dt=inv_dt, loc=loc,
                         comp=comp, fill_value=np.nan, startdate=startdate,
                         enddate=enddate)
        pp.run_checks()

        # Check correct test case:
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.checklist, pp.res) if b],
            ["check_case_multiple_channel_periods"]
        )
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.pp_functions, pp.res) if b],
            ["output_case_multiple_channel_periods"]
        )

    def test_case_incomplete_station_metadata(self):
        startdate = UTCDateTime(2013, 4, 9)
        enddate = UTCDateTime(2013, 4, 10)
        net, site, loc, comp = ('NZ', 'MAVZ', '10', 'HHZ')
        inv = read_inventory(os.path.join(self.sds_dir, 'MAVZ.xml'),
                             format='STATIONXML')
        inv_dt = inv.select(location=loc, channel=comp,
                            starttime=startdate, endtime=enddate)
        st = self.ds.clients[0].client.get_waveforms(net, site, loc, comp, startdate,
                                                     enddate, dtype='float64')
        pp = PostProcess(st=st, inv=inv, inv_dt=inv_dt, loc=loc,
                         comp=comp, fill_value=np.nan, startdate=startdate,
                         enddate=enddate)
        pp.run_checks()

        # Check correct test case:
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.checklist, pp.res) if b],
            ["check_case_incomplete_station_metadata"]
        )
        self.assertEqual(
            [a[0] for (a, b) in zip(pp.pp_functions, pp.res) if b],
            ["output_case_incomplete_station_metadata"]
        )
        test_st = st.copy()
        test_st.trim(startdate, inv_dt[0][0][0].end_date,
                     nearest_sample=False)
        test_st.attach_response(inv_dt)
        test_st.remove_sensitivity()
        test_st.trim(startdate, enddate, nearest_sample=False,
                     pad=True, fill_value=np.nan
                     )
        tr = pp.run_post_processing()

        np.testing.assert_array_almost_equal(tr.data, test_st[0].data)


class MetadataTestCase(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        self.volcano = 'Ruapehu'
        self.eruptions = [
            datetime.date(2006, 10, 4),
            datetime.date(2007, 9, 25)
        ]
        self.st_names = [
            "NZ.MAVZ.10.HHZ",
            "NZ.WHVZ.10.HHZ",
            "NZ.TRVZ.10.HHZ",
            "NZ.FWVZ.10.HHZ",
            "NZ.COVZ.10.HHZ",
        ]

    def test_get_streams(self):
        """
        Test that the streams returned from VolcanoMetadata class are
        consistent.
        """
        file = get_data('package_data/config.yml')
        vm = VolcanoMetadata(file)
        streams = vm.get_available_streams(self.volcano)
        for stream in streams:
            self.assertTrue(stream in self.st_names)

    def test_get_eruptions(self):
        """
        Test that the eruptions dates returned from VolcanoMetadata class
        are consistent.
        """
        file = get_data('package_data/config.yml')
        vm = VolcanoMetadata(file)
        er = vm.get_eruption_dates(self.volcano)
        for e in er:
            self.assertTrue(e in self.eruptions)

    def test_find_station_coordinates(self):
        # Sample data for testing
        test_data = """ 
        metadata:
          volcano: 
          - name: Test Volcano
            network:
            - net_code: TZ
              stations:
              - sta_code: TST1
                latitude: -37.000
                longitude: 177.000
              - sta_code: TST2
                latitude: -38.000
                longitude: 178.000
        """
        vm = VolcanoMetadata(test_data)

        # Test case for an existing station
        assert vm.get_site_information("TST1")['latitude'] == -37.0
        assert vm.get_site_information("TST1")['longitude'] == 177.0

        # Test case for a non-existing station
        assert vm.get_site_information("NONEXISTENT") is None

class FeatureRequestTestCase(unittest.TestCase):

    def setUp(self):
        # create test directory
        self.tempdir = tempfile.mkdtemp()
        self.rootdir = os.path.join(self.tempdir, 'Mt_Doom', 'MDR', 'HHZ')
        try:
            os.makedirs(self.rootdir)
        except FileExistsError:
            pass

    def tearDown(self):
        if os.path.isdir(self.tempdir):
            shutil.rmtree(self.tempdir)
            
    def test_false_feature(self):
        fq = FeatureRequest(rootdir=self.tempdir)
        with self.assertRaises(ValueError):
            fakefeature = fq('fakefeature')

    def test_call_multiple_days(self):
        startdate = UTCDateTime(2016, 1, 1)._get_datetime()
        enddate = UTCDateTime(2016, 1, 2, 12)._get_datetime()
        xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
        xarray2hdf5(xdf, self.rootdir)
        fq = FeatureRequest(rootdir=self.tempdir)
        fq.starttime = startdate
        fq.endtime = enddate
        fq.volcano = 'Mt_Doom'
        fq.site = 'MDR'
        fq.channel = 'HHZ'
        rsam = fq('rsam')
        # Check data
        xd_index = dict(datetime=slice(startdate,
                                       (enddate-pd.to_timedelta(fq.interval))))

        np.testing.assert_array_almost_equal(
            rsam.loc[startdate:enddate], xdf.rsam.loc[xd_index], 5)
        # Check datetime range is correct
        first_time = pd.to_datetime(rsam.datetime.values[0])
        last_time = pd.to_datetime(rsam.datetime.values[-1])
        self.assertEqual(pd.to_datetime(startdate), first_time)
        self.assertEqual(pd.to_datetime(enddate) - pd.to_timedelta(fq.interval),
                         last_time)

    def test_call_single_day(self):
        fq = FeatureRequest(rootdir=self.rootdir)
        startdate = datetime.datetime(2016, 1, 2, 1)
        enddate = datetime.datetime(2016, 1, 2, 12)
        xdf = generate_test_data(dim=1, tstart=startdate)
        xarray2hdf5(xdf, self.rootdir)
        fq = FeatureRequest(rootdir=self.tempdir)
        fq.starttime = startdate
        fq.endtime = enddate
        fq.volcano = 'Mt_Doom'
        fq.site = 'MDR'
        fq.channel = 'HHZ'
        rsam = fq('rsam')
        # Check datetime range is correct
        first_time = pd.to_datetime(rsam.datetime.values[0])
        last_time = pd.to_datetime(rsam.datetime.values[-1])
        self.assertEqual(pd.to_datetime(startdate), first_time)
        self.assertEqual(pd.to_datetime(enddate) - pd.to_timedelta(fq.interval),
                         last_time)

    def test_rolling_window(self):
        startdate = UTCDateTime(2016, 1, 1)._get_datetime()
        enddate = UTCDateTime(2016, 1, 2, 12)._get_datetime()
        xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
        xarray2hdf5(xdf, self.rootdir)
        fq = FeatureRequest(rootdir=self.tempdir)

        stack_len_seconds = 3600
        stack_len_string = '1H'

        num_windows = int(stack_len_seconds / pd.Timedelta(fq.interval).seconds)
        fq.starttime = startdate
        fq.endtime = enddate
        fq.volcano = 'Mt_Doom'
        fq.site = 'MDR'
        fq.channel = 'HHZ'
        rsam = fq('rsam')
        rsam_rolling = fq('rsam', stack_length=stack_len_string)

        # Check correct datetime array
        np.testing.assert_array_equal(rsam.datetime.values,
                                      rsam_rolling.datetime.values)
        # Check correct values
        rolling_mean = [
            np.nanmean(rsam.data[(ind-num_windows+1):ind+1])
            for ind in np.arange(num_windows, len(rsam_rolling.data))
        ]
        np.testing.assert_array_almost_equal(
            np.array(rolling_mean), rsam_rolling.values[num_windows:], 6
        )

    def test_cache(self):
        """
        Test that data loaded from files and from cache are identical.
        """
        startdate = UTCDateTime(2016, 1, 1)._get_datetime()
        enddate = UTCDateTime(2016, 1, 2, 12)._get_datetime()
        xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
        xarray2hdf5(xdf, self.rootdir)
        fq = FeatureRequest(rootdir=self.tempdir)
        fq.starttime = startdate 
        fq.endtime = enddate 
        fq.volcano = 'Mt_Doom'
        fq.site = 'MDR'
        fq.channel = 'HHZ'
        data_from_files = fq('RSAM')
        data_from_cache = fq('RSAM')
        xr.testing.assert_equal(data_from_files, data_from_cache)


if __name__ == '__main__':
    unittest.main()
