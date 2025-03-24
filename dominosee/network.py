import xarray as xr


"""
get link from threshold
"""
def get_link_from_threshold(da_sig: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Get the link from threshold DataArray
    """
    da_link = da_sig >= threshold
    return da_link


"""
get link from significance
"""
def get_link_from_significance(da_sig: xr.DataArray, p_threshold: float) -> xr.DataArray:
    """
    Get the link from significance DataArray
    """
    # get link
    da_link = da_sig <= p_threshold
    return da_link

def get_link_from_confidence(da_conf: xr.DataArray, confidence_level: float) -> xr.DataArray:
    """
    Get the link from confidence DataArray
    """
    # get link
    da_link = da_conf >= confidence_level
    return da_link


"""
get link from quantile
"""
def get_link_from_quantile(da_quant: xr.DataArray, q: float) -> xr.DataArray:
    """
    Get the link from quantile DataArray
    """
    # get global quantile
    quant = da_quant.quantile(q)
    # get link
    da_link = da_quant >= quant
    return da_link