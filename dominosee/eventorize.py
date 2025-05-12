from itertools import groupby
import numpy as np
import xarray as xr
# TODO: 事件选择当中很多任务已经在xclim中实现

"""
Utility functions
"""
def cut_single_threshold(ts, th, extreme="above", select=None):
    assert extreme in ["above", "below"], "extreme should be 'above' or 'below'"
    assert isinstance(th, (int, float)), "threshold should be a single value"

    if extreme == "above":
        te = ts >= th
    elif extreme == "below":
        te = ts <= th
    
    if select == "burst":
        te = select_burst(te)
    elif select == "wane":
        te = select_wane(te)
    return te


"""
Event selection
"""
def select_burst(te):
    tb = te.copy()  # time of bursts
    tb0 = np.roll(tb, 1)
    tb0[0] = False
    tb[tb & tb0] = False
    return tb


def select_wane(te):
    tw = te.copy()  # time of wanes
    tw0 = np.roll(tw, -1)
    tw0[-1] = False
    tw[te & tw0] = False
    return tw

def get_event(da: xr.DataArray, threshold: float, extreme: str, 
              event_name: str=None, select: str=None) -> xr.DataArray:
    event_name = "event" if event_name is None else event_name
    da = xr.apply_ufunc(
        cut_single_threshold,
        da,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        kwargs={"th": threshold, "extreme": extreme, "select": select}
    ).rename(event_name)

    # Transform dims back to original
    
    da.attrs = {
        "threshold": threshold,
        "extreme": extreme,
        "long_name": f"{event_name} events",
        "description": f"Events with {threshold} {extreme} threshold",
        "event_name": event_name,
        "select": select
    }
    return da

def _first_consecutive(event_bool, period=3):
    du_num = np.concatenate([[len(list(j)) for i, j in groupby(event_bool) if i]]) # get event durations
    due = np.zeros_like(event_bool)
    due[event_bool] = np.concatenate([np.ones(du)*du if du <= period 
                                                     else np.concatenate((np.ones(period)*du, np.zeros(du-period))) 
                                                     for du in du_num]) # select first period
    return due


# def durations_start(te, threshold=3):
#     du_num = np.concatenate([[len(list(j)) for i, j in groupby(x) if i] for x in te]).astype('uint16')
#     due = np.zeros_like(te, dtype='uint16')
#     due[te] = select_first_period(du_num, threshold)
#     return due


def select_first_consecutive(da, period=3):
    """
    Apply select_first_consecutive to an xarray DataArray.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing event durations
    period : int, optional
        Number of time steps to select from the beginning of each event, by default 3
        
    Returns
    -------
    xarray.DataArray
        DataArray with the selected first period values
    """
    # TODO: 处理输入数据不连续的情况，先插值成连续的再计算再还原
    return xr.apply_ufunc(
        _first_consecutive,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask='parallelized',
        kwargs={"period": period}
    )


"""
Event layer processor
"""
def merge_layers(da_list: list) -> xr.Dataset:
    ds = xr.merge(da_list, combine_attrs="drop_conflicts")
    return ds

def events_to_layer(da_list: xr.DataArray | list) -> xr.Dataset:
    if isinstance(da_list, xr.DataArray):
        ds = da_list.to_dataset()
    elif isinstance(da_list, (list, tuple)):
        ds = merge_layers(da_list)
    else:
        raise ValueError("da_list should be one or a list of xarray.DataArray of events")
    return ds


"""
Calculate properties
"""
def durations(te):
    du_num = np.concatenate([[len(list(j)) for i, j in groupby(x) if i] for x in te]).astype('uint16')
    due = np.zeros_like(te, dtype='uint16')
    due[te] = np.repeat(du_num, du_num)
    return due


# """
# The following are the obsolette versions lack of flexibility
# """
# def drought_time(ts, th, burst=False):  # events
#     te = ts <= th  # time of events
#     if burst:
#         tb = te.copy()  # time of bursts
#         tb0 = np.roll(tb, 1)
#         tb0[:, 0] = False
#         tb[tb & tb0] = False
#         return te, tb
#     else:
#         return te


# def flood_time(ts, th, burst=False):
#     te = ts >= th
#     if burst:
#         tb = te.copy()  # time of bursts
#         tb0 = np.roll(tb, 1)
#         tb0[:, 0] = False
#         tb[tb & tb0] = False
#         return te, tb
#     else:
#         return te