from datetime import datetime

import matplotlib.pyplot as plt
from cftime import date2num, num2date
import datashader as ds
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_ssam_mpl(xdf, log=False, axes=None, dbscale=False, cmap=cm.viridis,
                  clip=[0.0, 1.0], figsize=(12, 6)):
    """
    Plot spectrogram data using matplotlib.

    :param xdf: An xarray dataset as returned by :class:`zizou.ssam.SSAM`
    :type xdf: :class:`xarray.Dataset`
    :param log: Logarithmic frequency axis if True, linear frequency axis
                otherwise.
    :type log: bool
    :param axes: Plot into given axes.
    :type axes: :class:`matplotlib.axes.Axes`
    :param dbscale: If True 10 * log10 of color values is taken, if False the
                    sqrt is taken.
    :type dbscale: bool
    :param cmap: Specify a custom colormap instance.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param clip: adjust colormap to clip at lower and/or upper end. The given
                 percentages of the amplitude range (linear or logarithmic depending
                 on option `dbscale`) are clipped.
    :type clip: [float, float]
    :param figsize: Horizontal and vertical dimension 
                    of :class:`matplotlib.figure.Figure`
    :type figsize: [float, float]
    
    >>> import numpy as np
    >>> from zizou.ssam import SSAM, test_signal
    >>> from zizou.visualise import plot_ssam_mpl
    >>> ft = SSAM(interval=60, per_lap=.9, smooth=10, \
                  frequencies=np.arange(0., 25.1, .1), \
                  timestamp='start')
    >>> _ = ft.compute(test_signal())
    >>> fig = plot_ssam_mpl(ft.feature)

    """
    spec = xdf.data
    freq = xdf.frequency.data
    time = xdf.datetime.data
    time = pd.to_datetime(time).to_pydatetime()
    
    # db scale and remove zero/offset for amplitude
    if dbscale:
        spec = 10 * np.log10(spec[1:, :])
    else:
        spec = np.sqrt(spec[1:, :])
    freq = freq[1:]

    vmin, vmax = clip
    _range = float(spec.max() - spec.min())
    vmin = spec.min() + vmin * _range
    vmax = spec.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)
    # ignore warnings due to NaNs
    with np.errstate(invalid='ignore'):
        spec = np.where(spec < vmin, vmin, spec)
        spec = np.where(spec > vmax, vmax, spec)

    if not axes:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax = axes

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = dict(cmap=cmap)

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        # Log scaling for frequency values (y-axis)
        ax.set_yscale('log')
        # Plot times
        ps = ax.pcolormesh(time, freq, spec, norm=norm, **kwargs)
    else:
        # this method is much much faster!
        spec = np.flipud(spec)
        # center bin
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        ps = ax.imshow(spec, interpolation="nearest", extent=extent, **kwargs)

    # set correct way of axis, whitespace before and after with window
    # length
    ax.axis('tight')
    ax.grid(False)
    fig.colorbar(ps)

    if axes:
        return ax

    ax.set_ylabel('Frequency [Hz]')
    x_ticks = np.datetime64(xdf.attrs['starttime']) + ax.get_xticks().astype('timedelta64[s]')
    x_tick_labels = x_ticks.astype('datetime64[h]')
    ax.set_xticklabels(x_tick_labels, rotation=45)

    return fig


def plot_ssam_plotly(xdf, log=False, dbscale=False, cmap='Ice_r',
                     clip=[0.0, 1.0], new_fig=True, canvas_dim=(400,400)):
    """
    Plot spectrogram data using matplotlib.

    :param data: An xarray dataset as returned by :class:`zizou.ssam.SSAM`
    :type data: :class:`xarray.Dataset`
    :param log: Logarithmic frequency axis if True, linear frequency axis
                otherwise.
    :type log: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
                    sqrt is taken.
    :type dbscale: bool
    :param cmap: Specify a custom colormap instance from
                 `plotly.colors.sequential`.
    :type cmap: list
    :param clip: adjust colormap to clip at lower and/or upper end. The given
                 percentages of the amplitude range (linear or logarithmic
                 depending on option `dbscale`) are clipped.
    :type clip: [float, float]
    :param new_fig: Whether to return a figure or just a graph object.
    :type new_flag: boolean
    :param max_tpoints: Maximum number of time intervals. If the datasets
                        exceeds that number it will be resampled
                        to max_tpoints.
    :type max_tpoints: int

    >>> import numpy as np
    >>> from zizou.ssam import SSAM, test_signal
    >>> from zizou.visualise import plot_ssam_plotly
    >>> ft = SSAM(interval=60, per_lap=.9, smooth=10, \
                  frequencies=np.arange(0., 25.1, .1), \
                  timestamp='start')
    >>> _ = ft.compute(test_signal())
    >>> fig = plot_ssam_plotly(ft.feature)
    """
    vmin, vmax = clip
    _range = float(xdf.max() - xdf.min())
    vmin = xdf.min() + vmin * _range
    vmax = xdf.min() + vmax * _range
    
    with np.errstate(invalid='ignore'):
        xdf = xdf.where(xdf > vmin, vmin)
        xdf = xdf.where(xdf < vmax, vmax)

    if dbscale:
        xdf = 10*np.log10(xdf)

    freq_dim = xdf.dims[0]
    if xdf[freq_dim].data[0] == 0:
        xdf[{freq_dim:0}] = np.ones(xdf.shape[1])*np.nan

    freq = xdf[freq_dim]
    dates = xdf['datetime']
    spec = xdf.data
    if canvas_dim is not None:
        dates = date2num(xdf.datetime.data.astype('datetime64[us]').astype(datetime),
                         units='hours since 1970-01-01 00:00:00.0',
                         calendar='gregorian')
        xdf = xdf.assign_coords({'datetime': dates})
        cvs = ds.Canvas(plot_width=canvas_dim[0],
                        plot_height=canvas_dim[1])
        agg = cvs.raster(source=xdf)
        freq, d, spec = agg.coords[freq_dim], agg.coords['datetime'], agg.data
        dates = num2date(d, units='hours since 1970-01-01 00:00:00.0', calendar='gregorian')

    if not new_fig: 
        return go.Heatmap(
            z=spec,
            x=dates,
            y=freq,
            colorscale=cmap)

    fig = go.Figure(data=go.Heatmap(
            z=spec,
            x=dates,
            y=freq,
            colorscale=cmap))

    fig.update_layout(
        xaxis_nticks=20,
        yaxis_title='Frequency [Hz]'
    )

    if log:
        fig.update_yaxes(type='log')

    return fig


def get_trace(data, colour, kws_ssam):
    """
    Return a plotly graphics object from a feature.

    :param data: Feature data to plot.
    :type data: :class:`xarray.DataArray`
    :param colour: Colour for line plots.
    :type colour: str 
    :returns: A 1-D line plot or a 2D heatmap.
    :rtype: :class:`plotly.graph_objs.Scatter` or
            :class:`plotly.graph_objs.Figure`
    """
    if data.name == 'ssam':
        return plot_ssam_plotly(data, new_fig=False,
                                dbscale=True, **kws_ssam)
    elif data.name == 'filterbank':
         return plot_ssam_plotly(data, new_fig=False,
                                 dbscale=True, canvas_dim=None,
                                 **kws_ssam)
    elif data.name == 'sonogram':
         return plot_ssam_plotly(data, new_fig=False,
                                 dbscale=False, canvas_dim=None,
                                 **kws_ssam)
    else:
        # data = data.fillna(data.mean())
        dates = data.datetime.data
        dates = pd.to_datetime(dates).to_pydatetime()
        featureData = data.to_dataframe()[data.name].interpolate('index').values
        return go.Scatter(x=dates,
                          y=featureData, opacity=0.8,
                          line_color=colour,
                          name='<b>{}</b>'.format(str(data.name)))


def multi_feature_plot(feature1, feature2, equal_yrange=False,
                       rangeslider=False, log=False, kws_ssam=None):
    """
    Return a combined time-series plot of two features.

    :param feature1: First feature to plot with y-scale on
                     the left.
    :type feature1: str
    :param feature2: Second feature to plot with y-scale on
                     the right.
    :type feature2: str
    :param equal_yrange: If True align yscales.
    :type equal_yrange: bool
    :param rangeslider: If True show secondary panel for zooming.
    :type rangeslider: bool
    :param kws_ssam: Keywords to be passed to
                     :class:`zizou.visualise.plot_ssam_plotly`
    :type kws_ssam: dict
    :returns: plotly figure that can be displayed in a Jupyter
              notebook or in a dash app.
    :rtype: :class:`plotly.graph_objs.Figure`
    """
    if kws_ssam is None:
        kws_ssam = dict()
    
    if feature2.name in ['ssam', 'sonogram', 'filterbank']:
        # Always plot 2D features first
        feature_tmp = feature1
        feature1 = feature2
        feature2 = feature_tmp

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    trace = get_trace(feature1, 'deepskyblue', kws_ssam)
    fig.add_trace(trace, secondary_y=False)

    fig.update_yaxes(tickfont=dict(color='deepskyblue'),
                     title_font=dict(color='deepskyblue'))
    if feature2.name not in ['ssam', 'sonogram', 'filterbank']:
        if feature1.name in ['ssam', 'sonogram', 'filterbank']:
            trace = get_trace(feature2, 'darkred', kws_ssam)
            fig.add_trace(trace,
                          secondary_y=True)
            fig.update_yaxes(secondary_y=True,
                             tickfont=dict(color='darkred'),
                             title_font=dict(color='darkred'))
            fig.update_yaxes(tickfont=dict(color='#4d4db3'),
                             title_font=dict(color='#4d4db3'),
                             secondary_y=False)
        else:
            trace  = get_trace(feature2, 'dimgray', kws_ssam)
            fig.add_trace(trace,
                          secondary_y=True)
            fig.update_yaxes(secondary_y=True,
                             tickfont=dict(color='dimgray'),
                             title_font=dict(color='dimgray'))

    fig.update_layout(
        title={
            'text': "Features time series",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        height=500,
        xaxis_rangeslider_visible=rangeslider)

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>{}</b>'.format(str(feature1.name)), secondary_y=False)
    fig.update_yaxes(title_text='<b>{}</b>'.format(str(feature2.name)), secondary_y=True)
    
    if equal_yrange:
        if feature1.name in ['sonogram', 'filterbank', 'ssam']:
            # Smallest positive number
            y = fig['data'][0].y
            y_min = y[y>0][0] 
            y_max = fig['data'][0].y.max()
            y_range = np.array([y_min, y_max])
            if log:
                y_range = np.log10(y_range)
            fig.update_yaxes(range=y_range)
        else:
            y_min = min(float(feature1.min()),
                        float(feature2.min()))
            y_max = max(float(feature1.max()),
                        float(feature2.max()))
            if y_max > y_min:
                fig.update_yaxes(range=[y_min, y_max])

    if log:
        fig.update_yaxes(type='log')

    return fig


if __name__ == "__main__":
    import doctest
    doctest.testmod()
