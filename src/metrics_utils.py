import xarray as xr
import numpy as np

def compute_error_metrics(
    obs: xr.DataArray,
    model: xr.DataArray,
    metrics: list[str] = None
) -> xr.Dataset:
    """
    Compute error metrics between observation (obs) and model data.
    Supported metrics: ['MAE','RMSE','Bias']
    """
    diff = model - obs
    abs_diff = np.abs(diff)
    if metrics is None:
        metrics = ['MAE','RMSE','Bias']
    data_vars = {}
    if 'MAE' in metrics:
        data_vars['MAE'] = abs_diff.mean(dim=['latitude','longitude'], skipna=True)
    if 'RMSE' in metrics:
        data_vars['RMSE'] = np.sqrt((diff**2).mean(dim=['latitude','longitude'], skipna=True))
    if 'Bias' in metrics:
        data_vars['Bias'] = diff.mean(dim=['latitude','longitude'], skipna=True)
    return xr.Dataset(data_vars)
