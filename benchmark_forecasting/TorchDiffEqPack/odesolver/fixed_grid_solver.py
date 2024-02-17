from .base import ODESolver
import torch
__all__ = ['Euler','RK2','RK4']

class FixedGridSolver(ODESolver):

    def __init__(self, func, t0, y0, t1=1.0, h=0.1, rtol=1e-3, atol=1e-6, neval_max=500000,
                 print_neval=False, print_direction=False, step_dif_ratio=1e-3, safety=0.9,
                 regenerate_graph=False, dense_output=True, interpolation_method = 'cubic', end_point_mode = False, print_time = False):
        super(FixedGridSolver, self).__init__(func=func, t0=t0, y0=y0, t1=t1, h=h, rtol=rtol,
                                              atol=atol, neval_max=neval_max,
                 print_neval=print_neval, print_direction=print_direction, step_dif_ratio=step_dif_ratio, safety=safety,
                 regenerate_graph=regenerate_graph, dense_output=dense_output,
                                              interpolation_method = interpolation_method,
                                              end_point_mode = end_point_mode, print_time = print_time)

        if h is None:
            print('Stepsize h is required for fixed grid solvers')

        if not isinstance(h, torch.Tensor):
            h = torch.tensor(h).to(y0[0].device).float()
        self.h = h

        self.Nt = round(abs(self.t1.item() - self.t0.item())/self.h.item())

    def step(self, *args, **kwargs):
        pass

    def integrate(self, y0, t0, predefine_steps=None, return_steps=False, t_eval=None):
        # determine integration steps
        if predefine_steps is None:  # use steps defined by h
            steps = [self.t0 + (n + 1) * torch.abs(self.h) * self.time_direction for n in range(self.Nt)]
            steps = torch.stack(steps).view(-1).float()
        else:
            steps = predefine_steps

        out = self.integrate_predefined_grids(y0, t0, predefine_steps=steps, t_eval=t_eval)

        if return_steps:
            return out, steps
        else:
            return out

class Euler(FixedGridSolver):
    order = 1
    def step(self, func, t, dt, y, return_variables=False):
        k1 = func(t,y)
        out = tuple( _y + dt * _k1 for _y, _k1 in zip(y, k1) )
        if return_variables:
            return out, None, k1
        else:
            return out, None,

class RK2(FixedGridSolver):
    order = 2
    def step(self, func, t, dt, y, return_variables=False):
        k1 = func(t, y)
        k2 = func(t + dt / 2.0, tuple( _y + 1.0 / 2.0 * dt *_k1 for _y, _k1 in zip(y, k1)) )
        out = tuple( _y + dt * _k2 for _y, _k2 in zip(y, k2) )
        if return_variables:
            return out, None, [k1, k2]
        else:
            return out, None

class RK4(FixedGridSolver):
    order = 4
    def step(self, func, t, dt, y, return_variables=False):
        k1 = func(t, y)
        k2 = func(t + dt / 2.0, tuple( _y + 1.0 / 2.0 * dt *_k1 for _y, _k1 in zip(y, k1)  )   )
        k3 = func(t + dt / 2.0, tuple( _y + 1.0 / 2.0 * dt *_k2 for _y, _k2 in zip(y, k2)  ) )
        k4 = func(t + dt,  tuple( _y + dt *_k3 for _y, _k3 in zip(y, k3)  )   )
        out = tuple( _y + 1.0 / 6.0 * dt * _k1 + 1.0 / 3.0 * dt * _k2 + 1.0 / 3.0 * dt * _k3 + 1.0 / 6.0 * dt * _k4
                     for _y, _k1, _k2, _k3, _k4 in zip(y, k1, k2, k3, k4))
        if return_variables:
            return out, None, [k1, k2, k3, k4]
        else:
            return out, None
