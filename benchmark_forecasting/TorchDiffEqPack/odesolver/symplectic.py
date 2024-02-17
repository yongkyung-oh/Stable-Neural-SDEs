from .adaptive_grid_solver import AdaptiveGridSolver
from .fixed_grid_solver import FixedGridSolver

__all__ = ['Sym12Async', 'FixedStep_Sym12Async']
Lambda = 1.0

def sym12async_forward(func, t, dt, y, adaptive = False):
    _len = len(y)
    y0, v0 = y[0:_len // 2], y[_len // 2:]

    y1 = tuple(_y0 + 0.5 * dt * _v0 for _y0, _v0 in zip(y0, v0))
    vt1 = func(t + 0.5 * dt, y1)

    v1 = tuple(2 * Lambda * (_vt1 - _v0) + _v0 for _vt1, _v0 in zip(vt1, v0))
    y2 = tuple(_y1 + 0.5 * dt * _v1 for _y1, _v1 in zip(y1, v1))

    out = tuple(list(y2) + list(v1))

    if adaptive:
        error = tuple(_v1 * dt / 2.0 - _v0 * dt / 2.0 for _v1, _v0 in zip(vt1, v0))
        return out, error, [vt1, y1]
    else:
        return out, None, [v1, y1]

def sym12async_inverse(func, t1, dt, y):
    t1 = t1.to(y[0].device)
    dt = dt.to(y[0].device)
    t = t1 - dt  # initial time

    _len = len(y)
    y2, v1 = y[0:_len // 2], y[_len // 2:]
    y1 = tuple(_y2 - 0.5 * dt * _v1 for _y2, _v1 in zip(y2, v1))

    vt1 = func(t + 0.5 * dt, y1)
    v0 = tuple((2 * Lambda * _vt1 - _v1) / (2.0 * Lambda - 1.0) for _vt1, _v1 in zip(vt1, v1))

    y0 = tuple(_y1 - 0.5 * dt * _v0 for _y1, _v0 in zip(y1, v0))

    out = tuple(list(y0) + list(v0))

    return out, [y1, vt1]

class Sym12Async(AdaptiveGridSolver):
    order = 1
    def step(self, func, t, dt, y, return_variables=False):
        out, error, variables = sym12async_forward(func, t, dt, y, adaptive=True)
        if return_variables:
            return out, error, variables
        else:
            return out, error

    def inverse_async(self, func, t1, dt, y):
        return sym12async_inverse(func, t1, dt, y)

class FixedStep_Sym12Async(FixedGridSolver):
    order = 1
    def step(self, func, t, dt, y, return_variables=False):
        out, error, variables = sym12async_forward(func, t, dt, y, adaptive=False)
        if return_variables:
            return out, error, variables
        else:
            return out, error

    def inverse_async(self, func, t1, dt, y):
        return sym12async_inverse(func, t1, dt, y)
