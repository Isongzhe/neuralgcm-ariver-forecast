# NeuralGCM Atmospheric River Project

## Overview
This project demonstrates the use of the NeuralGCM model to forecast atmospheric variables and visualize atmospheric river events.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Forecasts](#running-forecasts)
  - [Plotting Results](#plotting-results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure
```
├── data/                # Input data for experiments
├── environment.yml      # Conda environment
├── notebooks/           # Jupyter notebooks for analysis and visualization
├── README.md            # This file
├── results/             # Output results (plots, metrics, etc.)
├── src/                 # Source code (plotting, metrics, model scripts)
├── tests/               # Unit tests for source code
├── __pycache__/         # Python cache files
```

## Installation
1. Clone this repository:
   ```bash
   git clone <repo_url>
   cd <project_root>
   ```
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate neuralgcm_env
   ```
3. Install additional Python packages if needed:
   ```bash
   pip install xarray matplotlib cartopy seaborn cmcrameri prefect typer loguru
   ```

## Usage

### Running Forecasts
Run the NeuralGCM model forecast and save results:
```bash
python src/run_neuralgcm.py [OPTIONS]
```
**Key Parameters:**
- `--model-name`: Model checkpoint file (default: `v1/deterministic_2_8_deg.pkl`)
- `--era5-path`: Path to ERA5 Zarr dataset (default: public Google Cloud path)
- `--start-time`: Forecast start date (default: `2024-02-04`)
- `--end-time`: Forecast end date (default: `2024-02-08`)
- `--inner-steps`: Time step in hours (default: `24`)
- `--days`: Number of forecast days (default: `4`)
- `--save`: Whether to save the output to Zarr (default: `True`)

**Examples:**
- Run with all defaults:
  ```bash
  python src/run_neuralgcm.py
  ```
- Custom date range:
  ```bash
  python src/run_neuralgcm.py --start-time 2024-03-01 --end-time 2024-03-05 --days 5
  ```
- Skip saving output:
  ```bash
  python src/run_neuralgcm.py --save False
  ```

### Plotting Results
Generate visualizations from forecast results:
```bash
python src/atm_river_plot.py [OPTIONS]
```
- `--us`: Plot for the US region (default: global)

**Examples:**
- Plot global results:
  ```bash
  python src/atm_river_plot.py
  ```
- Plot US region only:
  ```bash
  python src/atm_river_plot.py --us
  ```

### Notebooks
- `notebooks/neural_gcm.ipynb`: Load the NeuralGCM model and forecast.
- `notebooks/atm_river_plot.ipynb`: Data exploration and static visualizations.

## Contributing
Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features. For major changes, please discuss them in an issue first.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or collaboration, contact: SUNG-CHE, LIN (<your.email@domain.com>)

---

For most use cases, running the above commands with default parameters is sufficient. Adjust parameters as needed for custom experiments or regions.
