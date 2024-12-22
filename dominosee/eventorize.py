from itertools import groupby
import numpy as np
import xarray as xr

"""
Utility functions
"""
def cut_single_threshold(ts, th, extreme="above", burst=False):
    assert extreme in ["above", "below"], "extreme should be 'above' or 'below'"
    assert isinstance(th, (int, float)), "threshold should be a single value"

    if extreme == "above":
        te = ts >= th
    elif extreme == "below":
        te = ts <= th
    
    if burst:
        te = select_burst(te)
    return te


"""
Calculate properties
"""
def durations(te):
    du_num = np.concatenate([[len(list(j)) for i, j in groupby(x) if i] for x in te]).astype('uint16')
    due = np.zeros_like(te, dtype='uint16')
    due[te] = np.repeat(du_num, du_num)
    return due


"""
Event selection
"""
def select_burst(te):
    tb = te.copy()  # time of bursts
    tb0 = np.roll(tb, 1)
    tb0[:, 0] = False
    tb[tb & tb0] = False
    return tb


def select_first_period(durations, period=3):
    ev_durations = [np.ones(du)*du if du <= period else np.concatenate((np.ones(3)*du, np.zeros(du-3))) for du in durations]
    return np.concatenate(ev_durations)


def durations_start(te, threshold=3):
    du_num = np.concatenate([[len(list(j)) for i, j in groupby(x) if i] for x in te]).astype('uint16')
    due = np.zeros_like(te, dtype='uint16')
    due[te] = select_first_period(du_num, threshold)
    return due


"""
Event layer processor
"""
def get_events(da: xr.DataArray, threshold: float, extreme: str, 
               outname: str=None, burst: bool=False) -> xr.DataArray:
    # stack dataset if location has more than one dimension other than "time"
    if "lat" in da.dims and "lon" in da.dims:
        da = da.stack(location=("lat", "lon"))
    assert da.dims[0] == "time", "Time should be the first dimension."
    assert len(da.dims) == 2, "Space dimension should be only one dimension. \
        Please flatten using utils.flatten_location"
    outname = "event" if outname is None else outname
    # drop na
    da = da.dropna(da.dims[1], how="all")
    da = cut_single_threshold(da, threshold, extreme, burst)
    da = da.rename(outname)
    da.attrs["threshold"] = threshold
    da.attrs["extreme"] = extreme
    return da

def merge_layers(da_list: list) -> xr.Dataset:
    ds = xr.merge(da_list)
    return ds


"""
The following are the obsolette versions lack of flexibility
"""
def drought_time(ts, th, burst=False):  # events
    te = ts <= th  # time of events
    if burst:
        tb = te.copy()  # time of bursts
        tb0 = np.roll(tb, 1)
        tb0[:, 0] = False
        tb[tb & tb0] = False
        return te, tb
    else:
        return te


def flood_time(ts, th, burst=False):
    te = ts >= th
    if burst:
        tb = te.copy()  # time of bursts
        tb0 = np.roll(tb, 1)
        tb0[:, 0] = False
        tb[tb & tb0] = False
        return te, tb
    else:
        return te