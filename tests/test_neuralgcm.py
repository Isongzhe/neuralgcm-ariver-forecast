import os, sys
# 把 src 加入 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import xarray as xr
import numpy as np
from typer.testing import CliRunner
from neuralgcm import app, load_model, load_era5, preprocess_era5, forecast



runner = CliRunner()

# 定義 DummyModel 取代實際模型
class DummyModel:
    input_variables = ['a']
    forcing_variables = ['a']
    data_coords = type('C', (), {'horizontal': None})()
    @staticmethod
    def from_checkpoint(x):
        return DummyModel()
    def inputs_from_xarray(self, ds):
        return np.zeros((1,))
    def forcings_from_xarray(self, ds):
        return np.zeros((1,))
    def encode(self, inputs, forcings, rng):
        return 'state'
    def unroll(self, state, forcings, steps, timedelta, start_with_input):
        # 返回 final_state, predictions(shape=(1,))
        return state, np.zeros((1,))
    def data_to_xarray(self, data, times):
        return xr.DataArray(data, coords={'time': times}, dims=['time'])

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # patch load_model, load_era5, preprocess_era5, forecast
    monkeypatch.setattr('neuralgcm.load_model', lambda name: DummyModel())
    # 建立簡易 ERA5 Dataset
    def fake_load_era5(path, token='anon'):
        times = np.array(['2024-01-01T00:00'], dtype='datetime64')
        ds = xr.Dataset({'a': (['time'], [0])}, coords={'time': times})
        return ds
    monkeypatch.setattr('neuralgcm.load_era5', fake_load_era5)
    # preprocess_era5 直接回傳同一資料
    monkeypatch.setattr('neuralgcm.preprocess_era5', lambda full, model, **kwargs: fake_load_era5(None))
    # forecast 回傳 Dataset
    monkeypatch.setattr('neuralgcm.forecast', lambda model, ds, **kwargs: xr.Dataset({'a': (['time'], [0])}, coords={'time': np.array(['2024-01-01T00:00'], dtype='datetime64')}))
    yield

def test_cli_help():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'NeuralGCM Atmospheric River Predictions' in result.stdout

@pytest.mark.parametrize('save_flag,expected', [
    ('--save', 'Forecast complete and saved to Zarr.'),
    ('--no-save', 'Forecast complete.')
])
def test_main_run(save_flag, expected):
    result = runner.invoke(app, [save_flag])
    assert result.exit_code == 0
    assert expected in result.stdout
