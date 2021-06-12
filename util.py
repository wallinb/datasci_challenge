"""
Data loading/plotting utility functions
"""
from functools import partial

import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
from holoviews import opts, streams
from holoviews.operation.datashader import rasterize
from PIL import Image


def load_spectrogram(filepath="./spectrogram.tif"):
    """Load spectrogram as xarray.DataArray"""

    values = np.array(Image.open(filepath))
    rows, cols = values.shape
    times = np.linspace(0, 20, rows)
    frequencies = np.linspace(1000, 1800, cols)
    spectrogram = xr.DataArray(
        values, coords=[times, frequencies], dims=["time", "frequency"]
    )

    return spectrogram


def load_channel_metadata(filepath="./channel-metadata.csv"):
    """Load channel metadata as pandas.DataFrame"""

    channel_metadata = pd.read_csv(filepath)
    channel_metadata["interval"] = channel_metadata.apply(
        lambda row: pd.Interval(
            left=row.start_frequency, right=row.end_frequency, closed="left"
        ),
        axis=1,
    )
    channel_metadata["occupied"] = channel_metadata.occupied == "yes"

    return channel_metadata


def get_channel_ids(frequency, channel_metadata):
    """Return channel ids for a frequency as a set"""

    channel_metadata = channel_metadata.set_index("interval")

    try:
        subset = channel_metadata.loc[frequency].ID
    except KeyError:
        return {}

    try:
        return set(subset)
    except TypeError:
        return {subset}


def is_channel_occupied(channel_id, channel_metadata):
    """Return whether channel id is occupied"""

    channel_metadata = channel_metadata.set_index("ID")

    return channel_metadata.loc[channel_id].occupied


def slice_frequency_range(spectrogram, start_frequency, end_frequency):
    """Slice frequency range from spectrogram and return as xarray.DataArray"""

    sliced = spectrogram.loc[{"frequency": slice(start_frequency, end_frequency)}]

    return sliced


def annotate_channels(channel_ids, channel_metadata):
    """Plot rectangle spanning frequency range for given channel IDs"""

    channels = channel_metadata.set_index("ID").loc[channel_ids]
    annotation = hv.Overlay()
    for _, channel in channels.iterrows():
        annotation *= hv.VSpan(
            channel.start_frequency,
            channel.end_frequency,
        )

    return annotation


def plot_spectrogram_with_annotations(spectrogram, frequency, channel_metadata):
    """Plot spectrogram with annotation for frequency"""

    plot = hv.Image(spectrogram, kdims=["frequency", "time"])
    channel_ids = get_channel_ids(frequency, channel_metadata)
    plot *= annotate_channels(channel_ids, channel_metadata)

    return plot


def show_table_for_frequency(frequency, channel_metadata):
    """Show table with attributes for frequency"""

    channel_ids = get_channel_ids(frequency, channel_metadata)
    df = channel_metadata[channel_metadata.ID.isin(channel_ids)].drop(
        columns=["interval"]
    )
    table = hv.Table(df)

    return table


def plot_annotated_spectrograms(spectrogram, channel_metadata):
    """Plot channel spectrograms with interactive channel selection
    and table of channel attributes"""

    pointer = streams.PointerX().rename(x="frequency")

    dynamic_spectrogram = partial(
        plot_spectrogram_with_annotations,
        spectrogram=spectrogram,
        channel_metadata=channel_metadata,
    )
    plot = hv.DynamicMap(dynamic_spectrogram, streams=[pointer])
    plot = plot.redim.values(ID=channel_metadata.index)
    plot = rasterize(plot)

    dynamic_table = partial(
        show_table_for_frequency,
        channel_metadata=channel_metadata,
    )
    table = hv.DynamicMap(dynamic_table, streams=[pointer])

    layout = plot + table

    layout.opts(
        opts.Image(width=800, colorbar=True),
        opts.VSpan(fill_alpha=0, color="red"),
        opts.Table(width=800),
    ).cols(1)

    return layout
