import xarray as xr

"""
Location flatter
"""
def stack_lonlat(da: xr.DataArray, stack_dims: list=None) -> xr.DataArray:
    """Stack the space dimensions into one dimension, which is needed for network construction

    Args:
        da (xr.DataArray): DataArray with multiple space dimensions
        stack_dims (list, optional): list of dimension names to be stacked. Defaults to None.

    Raises:
        ValueError: `stack_dims` should be list of dimension names if 
        lat/lon or latitude/longitude are not in dims

    Returns:
        xr.DataArray: DataArray with the stacked space dimension
    """   
    # TODO: location may not satisfy the CF-1.6; use "cell" instead?
    #                                            use "node" instead?
    # http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#appendix-cell-methods
    if stack_dims is None:
        if "lat" in da.dims and "lon" in da.dims:
            da_stack = da.stack(location=("lat", "lon"))
        elif "latitude" in da.dims and "longitude" in da.dims:
            da_stack = da.stack(location=("latitude", "longitude"))
        else:
            raise ValueError("stack_dims should not be None if lat/lon or latitude/longitude are not in dims")
    else:
        da_stack = da.stack(location=stack_dims)
    return da_stack



"""
Region selector and remover
"""

def remove_nodes(ds: xr.Dataset, drop_nodes: xr.DataArray) -> xr.Dataset:
    """"""
    ds_wo_drop = ds
    return ds_wo_drop

