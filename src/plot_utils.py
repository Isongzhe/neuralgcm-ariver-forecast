import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cmc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from prefect import task
from metrics_utils import compute_error_metrics


@task
def plot_overview(
    ds: xr.Dataset,
    var: str,
    level: int,
    region_name: str,
    event_time: str,
    dir_path: Path,
) -> None:
    """
    Plot an overview facet grid for a given variable and level.
    """
    output_subdir = dir_path / 'overview'
    output_subdir.mkdir(parents=True, exist_ok=True)
    da = ds[var].sel(level=level)
    g = da.plot(
        x='longitude', y='latitude', row='time', col='model', robust=True, aspect=2, size=2,
        cmap=cmc.roma_r,
        cbar_kwargs={'label': f'{var} at {level} hPa'},
        subplot_kws={'projection': ccrs.PlateCarree()}
    )
    num_rows, num_cols = g.axs.shape
    for r in range(num_rows):
        for c in range(num_cols):
            ax = g.axs[r, c]
            ax.coastlines(resolution='10m', color='black', linewidth=1)
            ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
            ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.3)

            # Add gridlines, initially without drawing labels
            gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

            # Set label style and formatters (can be done for all)
            gl.xlabel_style = {'size': 10} # Smaller size might be needed
            gl.ylabel_style = {'size': 10}
            gl.xformatter = ccrs.cartopy.mpl.ticker.LongitudeFormatter()
            gl.yformatter = ccrs.cartopy.mpl.ticker.LatitudeFormatter()

            # Selectively enable labels for outer plots
            # Show longitude labels (bottom) only for the last row
            if r == num_rows - 1:
                gl.bottom_labels = True

            # Show latitude labels (left) only for the first column
            if c == 0:
                gl.left_labels = True

    if region_name == "Global-Region":     
        plt.subplots_adjust(
            wspace=0.05,  # 控制子圖之間的水平間距
            hspace=0.25,  # 控制子圖之間的垂直間距
            right=0.75,   # 控制整個圖形區域的右邊界（留出空間給colorbar）
            left=0.05,    # 控制左邊界位置
            top=0.95,      # 控制上邊界位置
            bottom=0.1    # 控制下邊界位置
        )

    plt.suptitle(f'{var} at {level} hPa ({region_name})\nevent time: {event_time}', fontsize=16, y=1.05)
    outfile = output_subdir / f'{region_name}_{var}_{level}_overview.png'
    g.fig.savefig(outfile, bbox_inches='tight')
    plt.close(g.fig)
    
@task
def plot_error_boxplot(
    ds: xr.Dataset,
    var: str,
    level: int,
    region_name: str,
    event_time: str,
    dir_path: Path,
) -> None:
    """
    Plot seaborn boxplot of absolute error distribution by time, adjusting y-limits by percentiles.
    """
    output_subdir = dir_path / 'boxplot_seaborn'
    output_subdir.mkdir(parents=True, exist_ok=True)
    da = ds[var].sel(level=level)
    obs = da.sel(model='ERA5')
    model = da.sel(model='NeuralGCM')
    error = np.abs(model - obs)
    df = error.to_dataframe(name='error').reset_index()
    df['time_str'] = df['time'].astype(str)
    times = sorted(df['time_str'].unique(), key=lambda x: int(x))
    # dynamic figure size
    fig_width = max(10, len(times) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    palette = sns.color_palette('Set3', n_colors=len(times))
    # draw boxplot using hue=time_str to satisfy seaborn v0.14 requirements, disable legend
    sns.boxplot(
        data=df,
        x='time_str',
        y='error',
        hue='time_str',
        dodge=False,
        palette=palette,
        width=0.6,
        fliersize=0,
        ax=ax,
        legend=False
    )
    ax.tick_params(axis='x', rotation=45)
    # set y-limits based on robust percentiles to avoid extreme outliers
    qmin, qmax = df['error'].quantile([0.01, 0.99])
    pad = (qmax - qmin) * 0.05
    ax.set_ylim(max(0, qmin - pad), qmax + pad)
    ax.set_xlabel('Forecast Lead (hours)')
    ax.set_ylabel('Absolute Error')
    plt.title(f'{var} {level}hPa Error Distribution ({region_name})')
    outfile = output_subdir / f'{region_name}_{var}_{level}_error_boxplot.png'
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    plt.close(fig)

@task
def plot_error_metrics(
    ds: xr.Dataset,
    var: str,
    level: int,
    region_name: str,
    event_time: str,
    dir_path: Path,
    metrics: list[str] = ['MAE', 'RMSE', 'Bias'],
) -> None:
    """
    Plot lineplot of error metrics vs time.
    """
    output_subdir = dir_path / 'metrics'
    output_subdir.mkdir(parents=True, exist_ok=True)
    da = ds[var].sel(level=level)
    obs = da.sel(model='ERA5')
    model = da.sel(model='NeuralGCM')
    metrics_ds = compute_error_metrics(obs, model, metrics)
    dfm = metrics_ds.to_dataframe().reset_index()
    dfm['time_str'] = dfm['time'].astype(str)
    # dynamic figure size
    times = sorted(dfm['time_str'].unique(), key=lambda x: int(x))
    fig_width = max(10, len(times) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    dfm_melt = dfm.melt(id_vars=['time_str'], value_vars=metrics, var_name='Metric', value_name='Value')
    sns.lineplot(
        data=dfm_melt,
        x='time_str',
        y='Value',
        hue='Metric',
        marker='o',
        ax=ax
    )
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('Forecast Lead (hours)')
    ax.set_ylabel('Metric Value')
    # set y-limits by data min/max with 5% padding
    vals = dfm_melt['Value']
    pad = (vals.max() - vals.min()) * 0.05
    ax.set_ylim(vals.min() - pad, vals.max() + pad)
    plt.title(f'{var} {level}hPa Error Metrics ({region_name})')
    outfile = output_subdir / f'{region_name}_{var}_{level}_error_metrics.png'
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    plt.close(fig)