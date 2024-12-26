import json
import numpy as np
from scipy.stats import binom
from numba import njit, prange
import xarray as xr

@njit(parallel=False)
def eca(b1, b2, b1w, b2wr, dtype='uint16'):
    # TODO: 拆分成precursor & trigger 两个函数；因为有时只需要一种
    KRprec = np.zeros((b1.shape[0], b2.shape[0]), dtype=dtype)  # precursor rates
    KRtrig = np.zeros((b1.shape[0], b2.shape[0]), dtype=dtype)  # triggering rates
    for j in range(b1.shape[0]):
        for k in range(b2.shape[0]): # TODO: 规范化代码时考虑只在这一层使用多核，对比一下速度；避免MPI，从而避免稀疏要求
            KRprec[j, k] = np.sum(b2[k, :] & b1w[j, :])   # precursor: b1   => (b2)  
            KRtrig[j, k] = np.sum(b1[j, :] & b2wr[k, :])  # trigger: (b1) => b2 
    return KRprec, KRtrig


@njit(parallel=True)
def eca_parallel(b1, b2, b1w, b2wr, dtype='uint16'):
    KRprec = np.zeros((b1.shape[0], b2.shape[0]), dtype=dtype)  # precursor rates
    KRtrig = np.zeros((b1.shape[0], b2.shape[0]), dtype=dtype)  # triggering rates
    for j in prange(b1.shape[0]):
        for k in range(b2.shape[0]): # TODO: 规范化代码时考虑只在这一层使用多核，对比一下速度；避免MPI，从而避免稀疏要求
            KRprec[j, k] = np.sum(b2[k, :] & b1w[j, :])   # b1   => (b2)  
            KRtrig[j, k] = np.sum(b1[j, :] & b2wr[k, :])  # (b1) => b2    
    return KRprec, KRtrig


def eca_dataset(b1: xr.DataArray, b2: xr.DataArray, b1w: xr.DataArray, b2wr: xr.DataArray, dtype=None, parallel=True):
    # TODO: 修正loc A/B的命名
    # infer dtype based on the length of time dimension
    if dtype is None:
        dtype = np.uint8 if b1.shape[0] < 256 else np.uint16 if b1.shape[0] < 65536 else np.uint32  # Cannot use string here in numba
    # get the location name from dims
    xdim = np.setdiff1d(b1.dims, "time")[0]
    layernames = [f"{b1.name}_{xdim}A", f"{b2.name}_{xdim}B"]

    # make sure all DataArray is in ("location", "time") coordinate order if not
    if b1.dims[0] != 'location':
        b1 = b1.transpose('location', 'time')
    if b2.dims[0] != 'location':
        b2 = b2.transpose('location', 'time')
    if b1w.dims[0] != 'location':
        b1w = b1w.transpose('location', 'time')
    if b2wr.dims[0] != 'location':
        b2wr = b2wr.transpose('location', 'time')
    # calculate the ECA
    if parallel:
        ECRprec, ECRtrig = eca_parallel(b1.values, b2.values, b1w.values, b2wr.values, dtype=dtype)
    else:
        ECRprec, ECRtrig = eca(b1.values, b2.values, b1w.values, b2wr.values, dtype=dtype)
    # create DataArray
    coords_locA = b1.indexes['location'].rename(["lat_locA", "lon_locA"])  # 这里一定不能用b1.coords['location'] rename，因为不会作用于MultiIndex
    coords_locB = b2.indexes['location'].rename(["lat_locB", "lon_locB"])
    ECRprec = xr.DataArray(ECRprec, coords=[coords_locA, coords_locB], dims=layernames, name="prec_evt",
                           attrs={'long_name': 'Precursor Events', 'units': 'count', 'dtype': dtype.__name__, 
                                  'description': 'Number of precursor events (from location A to location B) in location B',
                                  'eca_params': b1w.attrs["eca_params"]})
    ECRtrig = xr.DataArray(ECRtrig, coords=[coords_locA, coords_locB], dims=layernames, name="trig_evt",
                           attrs={'long_name': 'Trigger Events', 'units': 'count', 'dtype': dtype.__name__, 
                                  'description': 'Number of trigger events (from location A to location B) in location A',
                                  'eca_params': b2wr.attrs["eca_params"]})

    return ECRprec, ECRtrig


# def eca_dask(b1: xr.DataArray, b2: xr.DataArray, b1w: xr.DataArray, b2wr: xr.DataArray, dtype=None, chunksize=162):
#     layernames = [b1.name + "_locationA", b2.name + "_locationB"]
#     # make sure all DataArray is in ("time", "location") coordinate order if not
#     if b1.dims[0] != 'time':
#         b1 = b1.transpose('time', 'location')
#     if b2.dims[0] != 'time':
#         b2 = b2.transpose('time', 'location')
#     if b1w.dims[0] != 'time':
#         b1w = b1w.transpose('time', 'location')
#     if b2wr.dims[0] != 'time':
#         b2wr = b2wr.transpose('time', 'location')

#     b1_chunk = b1.rename({"location": layernames[0], "lon": "lon1", "lat": "lat1"}).chunk({layernames[0]: 162})  # chunk size 对 graph size 几乎没有影响
#     b2wr_chunk = b2wr.rename({"location": layernames[1], "lon": "lon2", "lat": "lat2"}).chunk({layernames[1]: 162})
#     ECRtrig = (b1_chunk & b2wr_chunk).sum(dim="time").compute().rename("trig_evt")
#     ECRtrig.attrs = {'long_name': 'Trigger Events', 'units': 'count', 'dtype': 'uint32', 'description': 'Number of trigger events (from location A to location B) in location A'}
#     return ECRprec, ECRtrig



def eca_window(b, delt=2, sym=True, tau=0):
    """
    b: 一维向量，表示时间序列
    delt: ECA窗口的大小
    sym: ECA窗口是否对称
    tau: ECA窗口的延迟

    return: 
    bw: 用于计算precursor的窗口
    bwr: 用于计算trigger的窗口

    note:
    这里窗口的计算服务于后续与之对应的点的ECA计算, 因此此处不是反映precursor/trigger的对应点, 而是另外一侧
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    window = np.ones((1 + 1 * sym) * delt + 1)
    # 用于precursor计算的窗口
    if delt == 0:
        bw = b  #.copy()
    else:
        # bw = np.apply_along_axis(lambda x: np.convolve(x, window)[sym*delt:-delt] >= 0.5, 1, b)
        bw = (np.convolve(b, window)[sym*delt:-delt] >= 0.5)
    if tau > 0:
        bw = np.roll(bw, tau)
        bw[:, :tau] = False

    # 用于trigger计算的窗口
    if sym:
        bwr = bw.copy()
    else:
        bwr = np.convolve(b, window)[delt:] >= 0.5
    if tau > 0:
        bwr = np.roll(bwr, -tau)
        bwr[:, -tau:] = False

    return bw, bwr


def get_eca_window(da: xr.DataArray, delt: int=2, sym: bool=True, tau: int=0) -> xr.DataArray:
    eca_params = {'delt': delt, 'sym': sym, 'tau': tau}
    da_prec_window, da_trig_window = xr.apply_ufunc(eca_window, da, input_core_dims=[["time"]], output_core_dims=[["time"], ["time"]], 
                                        vectorize=True, dask="parallelized", kwargs=eca_params)
    da_prec_window.attrs = {'long_name': 'Precursor Window', 'units': 'boolean', 'description': 'Window for precursor event identification',
                            "eca_params": json.dumps(eca_params)}
    da_trig_window.attrs = {'long_name': 'Trigger Window', 'units': 'boolean', 'description': 'Window for trigger event identification', 
                            "eca_params": json.dumps(eca_params)}
    return da_prec_window, da_trig_window


"""
Significance calculation
"""
def get_prec_significance(kp, na, nb, TOL, T, tau):
    # TODO: 应当强制检查na, nb的数量，当等于0时，返回1？
    return binom.cdf(kp, n=nb, p=1-(1-TOL/(T-tau))**na.reshape(-1, 1)).astype(np.float32)

def get_trig_significance(kt, na, nb, TOL, T, tau):
    return binom.cdf(kt, n=na.reshape(-1, 1), p=1-(1-TOL/(T-tau))**nb).astype(np.float32)