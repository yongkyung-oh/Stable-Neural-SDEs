import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torchcde


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_ROOT = ROOT / "benchmark_classification"
FORECASTING_ROOT = ROOT / "benchmark_forecasting"
TORCH_ISTS_NSDE = ROOT / "torch-ists" / "torch_ists" / "diff_module" / "NSDE" / "nsde_model.py"

PROPOSAL_METHOD_CONTRACT = {
    "lsde": (2, 16),
    "lnsde": (4, 17),
    "gsde": (6, 17),
}


@contextlib.contextmanager
def _sys_path(path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path.pop(0)


def _purge_benchmark_modules():
    for name in list(sys.modules):
        if name == "models_sde" or name.startswith("models_sde."):
            del sys.modules[name]
        if name == "controldiffeq" or name.startswith("controldiffeq."):
            del sys.modules[name]


def _load_benchmark_module(root):
    _purge_benchmark_modules()
    importlib.invalidate_caches()
    with _sys_path(root):
        return importlib.import_module("models_sde.neuralsde")


def _load_torch_ists_module():
    spec = importlib.util.spec_from_file_location("torch_ists_nsde_alignment", TORCH_ISTS_NSDE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_spline_data(batch_size, length, input_channels):
    times = torch.linspace(0.0, 1.0, steps=length)
    values = torch.linspace(
        0.1,
        1.0,
        steps=batch_size * length * input_channels,
        dtype=torch.float32,
    ).reshape(batch_size, length, input_channels)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(values, t=times)
    return coeffs, times


@pytest.mark.parametrize("benchmark_root", [CLASSIFICATION_ROOT, FORECASTING_ROOT], ids=["classification", "forecasting"])
@pytest.mark.parametrize("proposal_name", sorted(PROPOSAL_METHOD_CONTRACT))
def test_proposal_core_alignment_with_torch_ists(benchmark_root, proposal_name):
    benchmark_module = _load_benchmark_module(benchmark_root)
    torch_ists_module = _load_torch_ists_module()

    input_option, noise_option = PROPOSAL_METHOD_CONTRACT[proposal_name]
    hidden_channels = 4
    hidden_hidden_channels = hidden_channels
    input_channels = 3
    batch_size = 2
    length = 5

    benchmark_model = benchmark_module.Diffusion_model(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        hidden_hidden_channels=hidden_hidden_channels,
        num_hidden_layers=2,
        input_option=input_option,
        noise_option=noise_option,
    )
    torch_ists_model = torch_ists_module.Diffusion_model(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        hidden_hidden_channels=hidden_hidden_channels,
        num_hidden_layers=2,
        input_option=input_option,
        noise_option=noise_option,
    )

    coeffs, times = _make_spline_data(batch_size, length, input_channels)
    benchmark_model.set_X(coeffs, times)
    torch_ists_model.set_X(coeffs, times)

    benchmark_shapes = {name: tuple(param.shape) for name, param in benchmark_model.state_dict().items()}
    torch_ists_shapes = {name: tuple(param.shape) for name, param in torch_ists_model.state_dict().items()}
    assert benchmark_shapes == torch_ists_shapes

    torch_ists_model.load_state_dict(benchmark_model.state_dict())

    y = torch.linspace(
        0.2,
        0.9,
        steps=batch_size * hidden_channels,
        dtype=torch.float32,
    ).reshape(batch_size, hidden_channels)
    t = times[2]

    benchmark_f = benchmark_model.f(t, y)
    torch_ists_f = torch_ists_model.f(t, y)
    benchmark_g = benchmark_model.g(t, y)
    torch_ists_g = torch_ists_model.g(t, y)

    assert benchmark_f.shape == torch_ists_f.shape == (batch_size, hidden_channels)
    assert benchmark_g.shape == torch_ists_g.shape == (batch_size, hidden_channels)
    assert torch.isfinite(benchmark_f).all()
    assert torch.isfinite(torch_ists_f).all()
    assert torch.isfinite(benchmark_g).all()
    assert torch.isfinite(torch_ists_g).all()
    assert torch.allclose(benchmark_f, torch_ists_f, atol=1e-6, rtol=1e-6)
    assert torch.allclose(benchmark_g, torch_ists_g, atol=1e-6, rtol=1e-6)
