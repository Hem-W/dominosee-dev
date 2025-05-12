import numpy as np
import pandas as pd
from numba import njit, prange
import xarray as xr

@njit#(parallel=True)
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
    
    # Calculate time indices based on frequency
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
                            tm: int = np.inf) -> xr.DataArray:
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
    tm : int, optional
        Event synchronization threshold, by default np.inf
        
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

    def rename_dimensions(xr_obj, suffix='', keep_dims=None):
        """
        Rename dimensions in an xarray object, adding a suffix to each dimension name.
        Handles both standard dimensions and stacked dimensions with MultiIndex.
        
        Parameters
        ----------
        xr_obj : xr.DataArray or xr.Dataset
            The xarray object whose dimensions to rename
        suffix : str, optional
            Suffix to add to dimension names, by default ''
        keep_dims : list, optional
            List of dimension names to keep unchanged, by default None
            
        Returns
        -------
        tuple
            (renamed_obj, spatial_dims)
            renamed_obj: The xarray object with renamed dimensions
            spatial_dims: List of the renamed spatial dimensions
        """
        if keep_dims is None:
            keep_dims = []
        
        # Get all dimensions to rename (excluding those in keep_dims)
        dims_to_rename = [dim for dim in list(xr_obj.dims) if dim not in keep_dims]
        
        # Dictionary to store dimension renames
        rename_dict = {}
        
        # Process each dimension
        for dim in dims_to_rename:
            # Check if it's a stacked dimension (has a MultiIndex)
            if isinstance(xr_obj.indexes.get(dim), pd.MultiIndex):
                # Get the original dimensions before stacking
                original_dims = xr_obj[dim].attrs.get('stacked_dim_names', [])
                if not original_dims:  # Fallback if attrs not available
                    original_dims = list(xr_obj[dim].indexes[dim].names)
                
                # Add suffix to stacked dimension name
                rename_dict[dim] = f"{dim}{suffix}"
                
                # Add renamed original dimensions to the dict for reference
                for orig_dim in original_dims:
                    rename_dict[orig_dim] = f"{orig_dim}{suffix}"
            else:
                # Regular dimension, just add suffix
                rename_dict[dim] = f"{dim}{suffix}"
        
        # Apply renaming
        renamed_obj = xr_obj.rename(rename_dict)
        
        # Get the renamed spatial dimensions
        spatial_dims = list(np.setdiff1d(list(renamed_obj.dims), keep_dims))
        
        return renamed_obj, spatial_dims

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


    # Use xarray's apply_ufunc to compute event synchronization
    es = xr.apply_ufunc(
        _event_sync,
        positionsA, positionsB, diffsA, diffsB, event_countsA, event_countsB,
        input_core_dims=[[spatial_dimA[-1], 'event'], [spatial_dimB[-1], 'event'], 
                         [spatial_dimA[-1], 'event'], [spatial_dimB[-1], 'event'], 
                         [spatial_dimA[-1]], [spatial_dimB[-1]]],
        output_core_dims=[[spatial_dimA[-1], spatial_dimB[-1]]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.uint8],  # Match the return type from EvSync_2D_NB
        kwargs={'tm': tm, 'output_dtype': np.uint8}
    )
    
    # Add metadata to the result
    es.attrs.update({
        'long_name': 'Event Synchronization',
        'units': 'count',
        'description': 'Number of synchronized events between locations A and B',
        'threshold': tm
    })
    
    return es