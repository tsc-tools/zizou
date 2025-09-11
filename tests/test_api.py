from io import StringIO
import pandas as pd


def test_waveforms(setup_api):
    client = setup_api
    params = dict(network="NZ", station="WIZ", location="00", channel="HHZ",
                  starttime="2020-01-01T00:00:00", duration=600) 
    with client.stream("GET", "/waveform", params=params) as r:
        r.read()
        txt = r.text
    assert r.status_code == 200
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert df.shape == (250, 1)


def test_spectrogram(setup_api):
    client = setup_api
    params = dict(network="NZ", station="WIZ", location="00", channel="HHZ",
                  starttime="2020-01-01T00:00:00", duration=600) 
    with client.stream("GET", "/spectrogram", params=params) as r:
        r.read()
        txt = r.text
    assert r.status_code == 200
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert df.shape == (15351, 2)