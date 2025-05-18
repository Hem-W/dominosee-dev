import cftime
import numpy as np
import pandas as pd
from typing import Union, Tuple
from numba import njit, prange
import xarray as xr
from dominosee.utils.dims import rename_dimensions

# @njit(parallel=True)
def _event_sync(ew, ed, ewdiff, eddiff, noepw, noepd, tm, output_dtype=np.uint8):
    """
    :param ew: 一维展开后的各点bursts序列 = nodes×noe
    :param ed: 一维展开后的各点bursts序列 = nodes×noe
    :param ewdiff: 一维展开后的各点bursts序列的差分 = nodes×noe
    :param eddiff: 一维展开后的各点bursts序列的差分 = nodes×noe
    :param noepw: 各点bursts序列的列（人为指定的可能的最大值）
    :param noepd: 各点bursts序列的列（人为指定的可能的最大值）
    :param tm: ES阈值 = 30
    :return: Q
    """

    nodesA = ew.shape[0]
    nodesB = ed.shape[0]
    es = np.zeros((nodesA, nodesB), dtype=output_dtype)

    # Process each row in parallel
    for i in prange(nodesA):
        if noepw[i] > 2:
            # Pre-compute all ex values for this row
            ex = ew[i, 1:noepw[i]]
            ex_diff = ewdiff[i, 0:noepw[i]]
            ex_gapb = ex_diff[:-1]
            ex_gapf = ex_diff[1:]
            ex_tau = np.minimum(ex_gapb, ex_gapf)
            
            # Process each column sequentially for this row
            for k in range(nodesB):
                if noepd[k] > 2:
                    # Calculate synchronization for this (i,k) pair
                    count = 0
                    ey = ed[k, 1:noepd[k]]
                    ey_diff = eddiff[k, 0:noepd[k]]
                    ey_gapb = ey_diff[:-1]
                    ey_gapf = ey_diff[1:]
                    ey_tau = np.minimum(ey_gapb, ey_gapf)
                    
                    # Manual comparison to avoid broadcasting issues
                    for ix in range(len(ex)):
                        for iy in range(len(ey)):
                            # Calculate distance and minimum tau
                            dist = abs(ex[ix] - ey[iy])
                            # Choose the smaller of ex_tau[ix] and ey_tau[iy]
                            if ix < len(ex_tau) and iy < len(ey_tau):
                                tau = min(ex_tau[ix], ey_tau[iy]) / 2.0
                                # Check synchronization condition
                                if dist < tau and dist < tm:
                                    count += 1
                    
                    es[i, k] = count
        else:
            # # No events for this row
            # for k in range(nodesB):
            es[i, :] = 0
    return es


# def EvSync_2D_GB(ew, ed, ewdiff, eddiff, noepw, noepd, nodes, core, noc, tm, datanm, direc, th):
#     path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1_temp/2es'
#     print("batch %d running ..." % core)
#     Q = EvSync_2D_NB(ew, ed, ewdiff, eddiff, noepw, noepd, tm)
#     np.savez_compressed('{}/esevents_{}_glb_event{}_{}_c{}'.format(path, datanm, direc, th, core),
#                         Q=Q, noew=noepw, noed=noepd)  # 多存变量只会大一点点
#     return 0


# @njit
# def Ev_Position_NB(eb, mnoe, basedate=None):
#     if basedate is None:
#         basedate = np.zeros(eb.shape[1], dtype='uint16')
#     dat = np.zeros((eb.shape[0], mnoe), dtype='uint16')
#     for r in range(eb.shape[0]):
#         epi = np.where(eb[r, :])[0] + basedate[np.where(eb[r, :])[0]]
#         dat[r, :epi.size] = epi
#     return dat


# def ev_position_diff(eb, nob, dT):
#     ep = Ev_Position_NB(eb, nob.max() + 1).astype('int16')  # event positions
#     epdiff = np.diff(np.hstack((ep, np.zeros((ep.shape[0], 1), dtype=ep.dtype))), axis=1)  # 注意这里的负数超限
#     epdiff[epdiff < 0] = dT
#     epdiff = np.hstack((np.ones((epdiff.shape[0], 1), dtype=epdiff.dtype) * dT, epdiff))
#     return epdiff

def _extract_event_positions(binary_series, time_indices, max_count):
    """
    Extract event positions with fixed output size, converting to time indices
    
    Parameters
    ----------
    binary_series : ndarray
        1D binary time series (0s and 1s)
    time_indices : ndarray
        Array of time indices corresponding to binary_series
    max_count : int
        Maximum number of events to extract
    
    Returns
    -------
    ndarray
        Event positions (time indices) with fixed output size
    """
    # Initialize positions array with sentinel value
    positions = np.full(max_count, -1, dtype=np.int32)
    
    # Find event positions and convert to time indices
    event_pos = np.flatnonzero(binary_series)
    time_pos = time_indices[event_pos[:max_count]]
    
    # Fill positions array with time indices
    positions[:len(time_pos)] = time_pos
    
    return positions


def _DataArrayTime_to_timeindex(dt_index: xr.DataArray, reference_date: pd.Timestamp or cftime.datetime, freq: str):
    if freq == 'D':
        time_indices = (dt_index - reference_date).dt.days.values
    elif freq == 'W':
        time_indices = ((dt_index - reference_date).dt.days // 7).values
    elif freq == 'M':
        # Convert timestamps to periods and calculate month difference
        time_indices = np.array([
            (pd.Period(dt.values, freq='M').ordinal - pd.Period(reference_date.values, freq='M').ordinal)
            for dt in dt_index
        ])
    else:
        # Default to days if frequency is not recognized
        time_indices = (dt_index - reference_date).dt.days.values
    return time_indices


def get_event_positions(da, reference_date=None, freq=None):
    """
    Extract event positions from binary time series and convert to time indices
    
    Parameters
    ----------
    da : xr.DataArray
        Binary time series data
    reference_date : pd.Timestamp, optional
        Reference date for time indexing, by default None (uses first time value)
    freq : str, optional
        Frequency for time indexing, by default None (inferred from da.time)
    
    Returns
    -------
    xr.DataArray
        Event positions (time indices) for each location
    """
    
    # Get max possible events across all locations
    event_counts = da.sum(dim='time')
    max_events = int(event_counts.max().values)

    # Infer frequency from time dimension
    if freq is None:
        freq = xr.infer_freq(da.time)
        if freq in ["MS", "ME"]: 
            freq = "M"
    
    # Get time indices from da.time
    dt_index = da.time
    if reference_date is None:
        reference_date = da.time[0]
    
    time_indices = _DataArrayTime_to_timeindex(dt_index, reference_date, freq)
    
    # Create output DataArray with event dimension
    result = xr.apply_ufunc(
        _extract_event_positions,
        da,
        input_core_dims=[['time']],
        output_core_dims=[['event']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.int32],
        kwargs={'max_count': max_events, 'time_indices': time_indices}  # Pass max_events to all function calls
    )
    
    # Add event dimension coordinates
    result = result.assign_coords(event=np.arange(max_events))
    
    # Create dataset with results
    ds = xr.Dataset({
        'event_positions': result,
        'event_count': event_counts
    })
    
    return ds


def get_event_time_differences(da_positions: xr.DataArray, event_counts: xr.DataArray = None) -> xr.DataArray:
    """
    Calculate time differences between consecutive events for each location.
    
    Parameters
    ----------
    da_positions : xr.DataArray
        Event positions (time indices) for each location, as returned by get_event_positions
    event_counts : xr.DataArray, optional
        Number of events per location, by default None (will be calculated from da_positions)
        
    Returns
    -------
    xr.DataArray
        Time differences between consecutive events for each location.
        The first event at each location will have NaN as its time difference.
    """
    # If event_counts is not provided, calculate it from the positions
    # We need to identify valid events (those that are not -1 or sentinel values)
    if event_counts is None:
        # Assuming -1 or negative values are used as sentinel values for non-events
        valid_events = da_positions >= 0
        event_counts = valid_events.sum(dim='event')
    
    # Create a copy of the positions array shifted by one event
    # This will give us the previous event position for each event
    next_positions = da_positions.shift(event=-1)  # Shift in negative direction to get next event
    
    # Calculate the time differences (latter day minus previous day)
    time_diffs = next_positions - da_positions
    
    # Create a mask for valid time differences
    # A time difference is valid if both the current and next positions are valid events
    # and the current event index is less than the event count - 1 (to exclude the last event)
    event_indices = xr.DataArray(np.arange(da_positions.sizes['event']), dims=['event'])
    valid_diffs = (da_positions >= 0) & (next_positions >= 0) & (event_indices < (event_counts - 1))
    
    # Apply the mask to set invalid differences to NaN
    time_diffs = time_diffs.where(valid_diffs)
    
    # Set attributes
    time_diffs.attrs = {
        'long_name': 'Event Time Differences',
        'units': 'time steps',
        'description': 'Time differences between consecutive events for each location (latter - previous)'
    }
    
    return time_diffs


def get_event_sync_from_positions(positionsA: xr.DataArray, positionsB: xr.DataArray, 
                                  diffsA: xr.DataArray = None, diffsB: xr.DataArray = None, 
                                  event_countsA: xr.DataArray = None, event_countsB: xr.DataArray = None,
                                  tm: int = np.inf, parallel: bool = True) -> xr.DataArray:
    """
    Calculate Event Synchronization between two sets of event positions.
    
    Parameters
    ----------
    positionsA : xr.DataArray
        Event positions for location set A, as returned by get_event_positions
    positionsB : xr.DataArray
        Event positions for location set B, as returned by get_event_positions
    diffsA : xr.DataArray, optional
        Event time differences for location set A, by default None (calculated from positionsA)
    diffsB : xr.DataArray, optional
        Event time differences for location set B, by default None (calculated from positionsB)
    event_countsA : xr.DataArray, optional
        Number of events per location in set A, by default None (calculated from positionsA)
    event_countsB : xr.DataArray, optional
        Number of events per location in set B, by default None (calculated from positionsB)
    tm : int, optional
        Event synchronization threshold, by default np.inf
    parallel : bool, optional
        Whether to use parallel processing, by default True
        
    Returns
    -------
    xr.DataArray
        Event synchronization matrix between locations A and B
    """
    # Calculate time differences if not provided
    if diffsA is None:
        diffsA = get_event_time_differences(positionsA)
    if diffsB is None:
        diffsB = get_event_time_differences(positionsB)

    # First get the event counts if not provided
    if event_countsA is None:
        event_countsA = (positionsA >= 0).sum(dim='event')
    if event_countsB is None:
        event_countsB = (positionsB >= 0).sum(dim='event')
    
    # Use the rename_dimensions function for positionsA and related arrays
    positionsA, spatial_dimA = rename_dimensions(positionsA, suffix='A', keep_dims=['event'])
    diffsA, _ = rename_dimensions(diffsA, suffix='A', keep_dims=['event'])
    event_countsA, _ = rename_dimensions(event_countsA, suffix='A')
    
    # Use the rename_dimensions function for positionsB and related arrays
    positionsB, spatial_dimB = rename_dimensions(positionsB, suffix='B', keep_dims=['event'])
    diffsB, _ = rename_dimensions(diffsB, suffix='B', keep_dims=['event'])
    event_countsB, _ = rename_dimensions(event_countsB, suffix='B')


    # Determine the appropriate output dtype based on the maximum possible value
    # (which is the length of the time dimension)
    time_length = max(event_countsA.max().values, event_countsB.max().values)
    
    # Choose the smallest unsigned integer dtype that can hold the maximum value
    if time_length <= 255:  # uint8 max
        output_dtype = np.uint8
    elif time_length <= 65535:  # uint16 max
        output_dtype = np.uint16
    elif time_length <= 4294967295:  # uint32 max
        output_dtype = np.uint32
    else:  # For extremely large dimensions
        output_dtype = np.uint64
    
    # Use xarray's apply_ufunc to compute event synchronization
    es = xr.apply_ufunc(
        njit(parallel=parallel)(_event_sync),
        positionsA, positionsB, diffsA, diffsB, event_countsA, event_countsB,
        input_core_dims=[[spatial_dimA[-1], 'event'], [spatial_dimB[-1], 'event'], 
                         [spatial_dimA[-1], 'event'], [spatial_dimB[-1], 'event'], 
                         [spatial_dimA[-1]], [spatial_dimB[-1]]],
        output_core_dims=[[spatial_dimA[-1], spatial_dimB[-1]]],
        vectorize=True,
        dask='parallelized',
        kwargs={'tm': tm, 'output_dtype': output_dtype}
    )
    
    # Add metadata to the result
    es.attrs.update({
        'long_name': 'Event Synchronization',
        'units': 'count',
        'description': 'Number of synchronized events between locations A and B',
        'tau_max': tm,
        'output_dtype': str(output_dtype)
    })
    
    return es


# @njit()  # parallel=True
# def EvSync_Permu(season1, season2, tm):
#     """
#     Calculates triangle of Event Synchronization Matrix Q as list
#     dat: 点A和点B事件出现的位置
#     tlen： 时间长度（dat的长度）
#     nodes：节点数量 = 2
#     tm： ES的最大时间间隔参数
#     """
#     # tlen = length of time series
#     # workspace (tmp event series)
#     # dat0 = np.zeros(tlen, dtype="bool")
#     # dat1 = np.zeros(tlen, dtype="bool")
#     cor = np.zeros(2000)  # Null Model: 2000个样本
#     for k in prange(2000):
#         # print(k)
#         dat0 = np.random.permutation(season1)  # dat[0,] 点A事件发生的位置
#         dat1 = np.random.permutation(season2)  # dat[1,] 点B事件发生的位置

#         # %% self-production
#         ex = np.where(dat0)[0]  # 这里没有考虑场次事件的问题，因为一旦连续，事件数量会塌缩
#         ey = np.where(dat1)[0]
#         ex_diff = np.diff(ex)
#         ex_diff = ex_diff.reshape(ex_diff.size, 1)
#         ey_diff = np.diff(ey)
#         ex_gapb = ex_diff[:-1, ]
#         ex_gapf = ex_diff[1:, ]
#         ey_gapb = ey_diff[:-1]
#         ey_gapf = ey_diff[1:]
#         exeydist = np.abs(ex[1:-1].reshape(ex[1:-1].size, 1) - ey[1:-1])
#         tau = np.minimum(np.minimum(ex_gapb, ex_gapf), np.minimum(ey_gapb, ey_gapf)) / 2
#         ESij = (exeydist < tau) & (exeydist < tm)
#         cor[k] = np.sum(ESij)
#     return cor

# datanm = "spimv2"
# lon = np.load('0data/{}_lon.npy'.format(datanm))
# lat = np.load('0data/{}_lat.npy'.format(datanm))
# ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))
# tlen = ddate.size  # 数据总时长 = y*(365/366)
# mnoe = 100  # 人工指定的生成事件数量的最大值，可能是取了数据长度的5% maximum number of events
# sigs = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
# nb.config.NUMBA_DEFAULT_NUM_THREADS = 24

# # ddate = ddate[(ddate.year >= ddate[0].year) & (ddate.year <= ddate[0].year + wd)]
# # index_seasons = np.zeros(4, dtype='object')
# # season_split = {0: [12, 1, 2], 1: [3, 4, 5],
# #                 2: [6, 7, 8], 3: [9, 10, 11]}
# # for j in range(0, 4):
# #     index_seasons[j] = np.where(np.in1d(ddate.month, season_split.get(j)))[0]
# # P1 = np.zeros((mnoe + 1, mnoe + 1), dtype='int')
# P = np.zeros((mnoe + 1, mnoe + 1, sigs.size), dtype='int')

# for tm in [2]:
#     a = 0
#     BaseTime = time.perf_counter()
#     AllTime = []
#     for i in range(3, mnoe + 1):
#         for j in range(3, i + 1):  # j = 任意点对中点B上可能的事件数量
#             l = ddate.size
#             season1 = np.zeros(l, dtype="bool")
#             season2 = np.zeros(l, dtype="bool")
#             # index_seas = np.array(index_seasons[2], dtype='uint32')
#             season1[:i] = 1
#             season2[:j] = 1
#             cor = EvSync_Permu(season1, season2, tlen, tm)
#             # np.zeros(2000)  # Null Model: 2000个样本

#             P[i, j, :] = st.scoreatpercentile(cor, 100 - sigs * 100)
#             # th05 = st.scoreatpercentile(cor, 95)
#             # th02 = st.scoreatpercentile(cor, 98)
#             # th01 = st.scoreatpercentile(cor, 99)
#             # th005 = st.scoreatpercentile(cor, 99.5)
#             # th001 = st.scoreatpercentile(cor, 99.9)
#             #
#             # P1[i, j] = th05
#             # P2[i, j] = th02
#             # P3[i, j] = th01
#             # P4[i, j] = th005
#             # P5[i, j] = th001

#             a += 1

#         AllTime.append(time.perf_counter() - BaseTime)
#         print(i, j, P[i, j, 0], AllTime[-1])
#     for nsig, sig in enumerate(sigs):
#         P0 = diagonal_mirror(P[:, :, nsig])
#         np.save('2eca/null/esnull_{}_win{}_sig{}_evmax{}.npy'.format(datanm, tm, sig, mnoe), P0)


"""
# Null Model for Event Synchronization
"""

# @njit(parallel=True) # inline parallel by njit
def _event_sync_null(time_indice, noeA, noeB, tm, samples=2000):
    """
    Calculates event synchronization using permutation tests based on actual time indices
    
    Parameters:
    -----------
    time_indice : array
        Time indices of events in location A
    noeA : int
        Number of events in location A
    noeB : int
        Number of events in location B
    tm : int
        Maximum time interval parameter for ES
    samples : int, optional
        Number of samples to generate, by default 2000
    
    Returns:
    --------
    cor : array
        Array of synchronization values from permutations
    """
    cor = np.zeros(samples, dtype='int')  # Null Model: 2000 samples
    # require noeA and noeB to be at least 3
    if noeA < 3 or noeB < 3:
        return cor
    if len(time_indice) < 3:
        return cor
    
    for k in prange(samples):
        # Random permutation of time indices
        dat0_indices = np.random.choice(time_indice, size=noeA, replace=False)
        dat1_indices = np.random.choice(time_indice, size=noeB, replace=False)
        
        ex = dat0_indices[1:-1]
        ey = dat1_indices[1:-1]
        ex_diff = np.diff(dat0_indices)
        ey_diff = np.diff(dat1_indices)
        
        # ex_gapb = ex_diff[:-1, ]
        # ex_gapf = ex_diff[1:, ]
        # ey_gapb = ey_diff[:-1]
        # ey_gapf = ey_diff[1:]
        # exeydist = np.abs(ex.reshape(ex.size, 1) - ey)
        # tau = np.minimum(np.minimum(ex_gapb, ex_gapf), np.minimum(ey_gapb, ey_gapf)) / 2
        # ESij = (exeydist < tau) & (exeydist < tm)
        # cor[k] = np.sum(ESij)

        ex_tau = np.minimum(ex_diff[:-1], ex_diff[1:])
        ey_tau = np.minimum(ey_diff[:-1], ey_diff[1:])
        count = 0
        for ix in range(len(ex)):
            for iy in range(len(ey)):
                dist = abs(ex[ix] - ey[iy])
                if ix < len(ex_tau) and iy < len(ey_tau):
                    tau = min(ex_tau[ix], ey_tau[iy]) / 2.0
                    if dist < tau and dist < tm:
                        count += 1
        cor[k] = count
    
    return cor


def _diagonal_mirror(arr):
    farr = arr.copy()
    pos = np.where(arr)  # only applicable for int > 0
    farr[(pos[1], pos[0])] = arr[pos]
    return farr


def create_null_model_from_indices(da_timeIndex: xr.DataArray, tm: int, max_events: Union[int, Tuple[int, int], np.ndarray], 
                                   significances: list=[0.05], samples: int=2000, min_es: int=None, parallel: bool=True) -> xr.DataArray:
    """
    Creates a null model for event synchronization based on actual time indices
    
    Parameters:
    -----------
    da_timeIndex : xr.DataArray
        Array of time indices where events occur in datasets of two locations
    tm : int
        Maximum time interval parameter for ES
    max_events : int, tuple of two ints, or numpy.ndarray
        Maximum number of events to consider. If a single int or 1D array with one element, calculates over [max_events, max_events].
        If a tuple or 1D array with two elements, calculates over [max_events[0], max_events[1]].
    significances : list, optional
        List of significance levels to calculate thresholds for
    samples : int, optional
        Number of samples to generate, by default 2000
    min_es : int, optional
        Minimum number of event synchronizations, by default None
    parallel : bool, optional
        Whether to use parallelization, by default True
    
    Returns:
    --------
    da_critical_values : xr.DataArray
        List of arrays containing critical values for event synchronizations
    """
    # import time
    import scipy.stats as st
    
    # Define significance levels
    sigs = np.array(significances)
    
    # Determine the range of events to calculate
    if isinstance(max_events, int):
        max_events_A = max_events_B = max_events
    elif isinstance(max_events, np.ndarray):
        if max_events.size == 1:
            max_events_A = max_events_B = int(max_events.item())
        elif max_events.size == 2:
            max_events_A, max_events_B = max_events
        else:
            raise ValueError("If max_events is a numpy array, it must have one or two elements")
    elif isinstance(max_events, tuple) and len(max_events) == 2:
        max_events_A, max_events_B = max_events
    else:
        raise ValueError("max_events must be either an int or a tuple of two ints")
    
    # Initialize results array
    critical_values = np.zeros((max_events_A + 1, max_events_B + 1, len(sigs)), dtype='int')
    freq = xr.infer_freq(da_timeIndex.time)
    if freq in ["MS", "ME"]: 
        freq = "M"
    time_indice = _DataArrayTime_to_timeindex(da_timeIndex, da_timeIndex.time.values[0], freq)

    # base_time = time.perf_counter()
    # all_time = []

    event_sync_null = njit(parallel=parallel)(_event_sync_null)

    for i in range(3, max_events_A + 1):
        for j in range(3, min(i + 1, max_events_B + 1)):
            
            # Calculate null distribution
            cor = event_sync_null(time_indice, i, j, tm, samples)
            
            # Calculate significance thresholds
            critical_values[i, j, :] = st.scoreatpercentile(cor, 100 - sigs * 100)
            
            # print(f"Processed: events A={i}, events B={j}, threshold={critical_values[i, j, 0]}")
        
        # all_time.append(time.perf_counter() - base_time)
        # print(f"Completed events A={i}, time elapsed={all_time[-1]:.2f}s")
    
    # Mirror the matrix for symmetry
    for nsig, sig in enumerate(sigs):
        critical_values[:, :, nsig] = _diagonal_mirror(critical_values[:, :, nsig])
    
    # Set minimum of event synchronizations
    if min_es is not None:
        critical_values[critical_values < min_es] = min_es
    
    da_critical_values = xr.DataArray(critical_values, 
                                      dims=["noeA", "noeB", "significance"], 
                                      coords={"noeA": np.arange(max_events_A + 1), "noeB": np.arange(max_events_B + 1), "significance": sigs})
    # add time information
    # da_critical_values = da_critical_values.assign_coords(time=da_timeIndex.time) # TODO: record time info
    da_critical_values = da_critical_values.assign_attrs({"description": "Event synchronization null model for pairs of number of events",
                                                          "tau_max": tm,
                                                          "max_events": max_events,
                                                          "min_es": min_es,
                                                          })
    
    return da_critical_values


def convert_null_model_for_locations(da_critical_values: xr.DataArray, da_evN_locA: xr.DataArray, da_evN_locB: xr.DataArray, 
                                     sig: float=None) -> xr.DataArray:
    # rename dimension of da_evN_locA and da_evN_locB
    da_evN_locA, _ = rename_dimensions(da_evN_locA, suffix='A')
    da_evN_locB, _ = rename_dimensions(da_evN_locB, suffix='B')
    
    if sig is None and da_critical_values.dims["significance"] == 1:
        sig = da_critical_values.coords["significance"][0]
    elif sig is None:
        raise ValueError("sig must be specified")
    
    da_null = da_critical_values.sel(noeA=da_evN_locA, noeB=da_evN_locB, significance=sig)
    return da_null