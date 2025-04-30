import gcsfs
import jax
import numpy as np
import pickle
import xarray
from pathlib import Path

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

from prefect import task, flow
from prefect.cache_policies import NO_CACHE

from loguru import logger
import sys

import typer


# Configure app cli with Typer 
app = typer.Typer(help="NeuralGCM Atmospheric River Predictions")

# logger setup
logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}")


@task
def load_model(
    model_name: str = 'v1/deterministic_2_8_deg.pkl',
    token: str = 'anon'
) -> neuralgcm.PressureLevelModel:
    """
    Load NeuralGCM model from Google Cloud Storage.
    """
    fs = gcsfs.GCSFileSystem(token=token)
    with fs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

@task
def load_era5(
    era5_path: str = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    token: str = 'anon'
) -> xarray.Dataset:
    """
    Load ERA5 dataset in Zarr format.
    """
    return xarray.open_zarr(
        era5_path,
        chunks=None,
        storage_options=dict(token=token)
    )

@task(cache_policy=NO_CACHE)
def preprocess_era5(
    full_era5: xarray.Dataset,
    model,
    start_time: str = '2024-02-04',
    end_time: str = '2024-02-08',
    data_inner_steps: int = 24
) -> xarray.Dataset:
    """
    Slice and regrid ERA5 data for model input.
    """
    sliced = (
        full_era5[model.input_variables + model.forcing_variables]
        .pipe(
            xarray_utils.selective_temporal_shift,
            variables=model.forcing_variables,
            time_shift=f'{data_inner_steps} hours'
        )
        .sel(
            time=slice(start_time, end_time, data_inner_steps)
        )
        .compute()
    )
    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=full_era5.sizes['latitude'],
        longitude_nodes=full_era5.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid,
        model.data_coords.horizontal,
        skipna=True
    )
    eval_era5 = xarray_utils.regrid(sliced, regridder)
    return xarray_utils.fill_nan_with_nearest(eval_era5)

@task(cache_policy=NO_CACHE)
def forecast(
    model,
    eval_era5: xarray.Dataset,
    init_time: str = '2024-02-04T00:00:00',
    hour_inner_steps: int = 24,
    days: int = 4
) -> xarray.Dataset:
    """
    Run forecast for specified days and return combined ERA5 vs NeuralGCM dataset.
    """
    init_np = np.datetime64(init_time)
    inner_steps = hour_inner_steps
    outer_steps = days * 24 // inner_steps
    timedelta = np.timedelta64(1, 'h') * inner_steps
    times = np.arange(outer_steps) * inner_steps

    inputs = model.inputs_from_xarray(eval_era5.sel(time=init_np))
    input_forcings = model.forcings_from_xarray(eval_era5.sel(time=init_np))
    rng_key = jax.random.key(42)
    initial_state = model.encode(inputs, input_forcings, rng_key)
    all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

    final_state, predictions = model.unroll(
        initial_state,
        all_forcings,
        steps=outer_steps,
        timedelta=timedelta,
        start_with_input=True
    )
    predictions_ds = model.data_to_xarray(predictions, times=times)

    target_traj = model.inputs_from_xarray(
        eval_era5
        .thin(time=(inner_steps // hour_inner_steps))
        .isel(time=slice(outer_steps))
    )
    target_ds = model.data_to_xarray(target_traj, times=times)

    combined = xarray.concat([target_ds, predictions_ds], dim='model')
    combined.coords['model'] = ['ERA5', 'NeuralGCM']
    return combined

@task(cache_policy=NO_CACHE)
def save_zarr(
    ds: xarray.Dataset,
    path: Path = Path('../data/neuralgcm_atmospheric_river_predictions.zarr')
) -> None:
    """
    Save dataset to a Zarr store.
    """
    # 確保 parent 資料夾存在
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode='w', compute=True)


@flow
@app.command()
def main(
    model_name: str = typer.Option(
        'v1/deterministic_2_8_deg.pkl',
        help='GCS model file path'
    ),
    era5_path: str = typer.Option(
        'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
        help='ERA5 Zarr dataset path'
    ),
    start_time: str = typer.Option(
        '2024-02-04',
        help='Data slice start time'
    ),
    end_time: str = typer.Option(
        '2024-02-08',
        help='Data slice end time'
    ),
    inner_steps: int = typer.Option(
        24,
        help='Inner time step (hours)'
    ),
    days: int = typer.Option(
        4,
        help='Number of forecast days'
    ),
    save: bool = typer.Option(
        True,
        help='Save results to Zarr?'
    ),
):
    model = load_model(model_name)
    full_era5 = load_era5(era5_path)
    eval_era5 = preprocess_era5(
        full_era5,
        model,
        start_time=start_time,
        end_time=end_time,
        data_inner_steps=inner_steps,
    )
    combined_ds = forecast(
        model,
        eval_era5,
        init_time=f'{start_time}T00:00:00',
        hour_inner_steps=inner_steps,
        days=days,
    )
  
    if save:
        save_zarr(combined_ds)
        logger.success('Forecast complete and saved to Zarr.')
    else:
        logger.success('Forecast complete.')

if __name__ == '__main__':
  app()