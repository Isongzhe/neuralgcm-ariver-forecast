# Visualize ERA5 vs NeuralGCM trajectories
import xarray as xr
from pathlib import Path
from plot_utils import plot_overview, plot_error_boxplot, plot_error_metrics

from prefect import task, flow


from loguru import logger
import sys

import typer

from dataclasses import dataclass
@dataclass
class Region:
    min: float
    max: float

# Configure app cli with Typer 
app = typer.Typer(help="Plotting Atmospheric River Predictions")

# logger setup
logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}")


@task
def read_zarr(zarr_path: str = '../data/neuralgcm_atmospheric_river_predictions.zarr') -> xr.Dataset:
    """
    Read a Zarr file and return an xarray dataset.
    """
    try:
        ds = xr.open_zarr(zarr_path, chunks='auto', consolidated=False)
    except FileNotFoundError:
        logger.error(f"Zarr file {zarr_path} not found.")
        raise
    logger.info(f"Zarr file {zarr_path} read successfully.")
    logger.debug(f"Dataset dimensions: {ds.sizes}")
    logger.debug(f"Dataset variables: {ds.data_vars}")
    logger.debug(f"Dataset coordinates: {ds.coords}")
    logger.debug(f"Dataset attributes: {ds.attrs}")
    return ds

@task
def select_subset_region(
    ds: xr.Dataset, 
    lon_range: Region, 
    lat_range: Region,
) -> xr.Dataset:
    """
    Slice the dataset to a specific latitude and longitude range.
    """
    if lat_range.min >= lat_range.max or lon_range.min >= lon_range.max:
        logger.error("Invalid region: min must be less than max.")
        raise ValueError("Region min must be less than max.")
    ds_subset = ds.where(
        (ds.latitude >= lat_range.min) & (ds.latitude <= lat_range.max) &
        (ds.longitude >= lon_range.min) & (ds.longitude <= lon_range.max),
        drop=True
    )
    if ds_subset.sizes["latitude"] == 0 or ds_subset.sizes["longitude"] == 0:
        logger.warning("Selected region is empty!")
    return ds_subset

@flow
def plot(
    ds: xr.Dataset, 
    us_region: bool,
    dir_path: Path = Path('../results')
) -> None:
    """
    Plot the data for each variable in the dataset.
    """
    # Define the variables and levels to plot
    variables = ['temperature', 'geopotential', 'specific_cloud_liquid_water_content']
    levels = [850, 500]
    
    # Define the event time
    from datetime import datetime
    event_time_iso = '2024-02-04T00:00:00'
    event_dt = datetime.fromisoformat(event_time_iso)
    display_time = event_dt.strftime('%Y-%m-%d %H:%M UTC') 
    
    if us_region:
        logger.info("Plotting for US region.")
        region_name = "US-Region"
    else:
        logger.info("Plotting for Global region.")
        region_name = "Global-Region"
        
    
    # Synchronous plotting: overview, error distribution, error metrics
    logger.info('Starting synchronous plotting tasks...')
    for var in variables:
        for level in levels:
            plot_overview(ds=ds, var=var, level=level, region_name=region_name,
                          event_time=display_time, dir_path=dir_path)
            plot_error_boxplot(ds=ds, var=var, level=level, region_name=region_name,
                              event_time=display_time, dir_path=dir_path)
            plot_error_metrics(ds=ds, var=var, level=level, region_name=region_name,
                               event_time=display_time, dir_path=dir_path)
    logger.info('All synchronous plotting tasks finished.')

@flow
@app.command()
def main(
    us: bool = False,
):
    """
    Main function to read the Zarr file and plot the data.
    """
    ds = read_zarr()
    if us:
        logger.info("Selecting US region.")
        ds = select_subset_region(ds, 
            lon_range=Region(220, 300), 
            lat_range=Region(10, 60)
        )
    else:
        logger.info("Selecting Global region.")
        pass

    logger.info("Plotting data.")
    plot(ds, us)

if __name__ == "__main__":
    app()