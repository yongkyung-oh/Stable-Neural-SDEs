from ..odesolver.fixed_grid_solver import *
from ..odesolver.adaptive_grid_solver import *
from ..utils import extract_keys
from ..odesolver.symplectic import *
from ..odesolver.stiff_ode_solver import *
from ..odesolver.base import check_arguments
import torch
__all__ = ['odesolve_endtime']

def odesolve_endtime(func, z0, options, return_solver=False, **kwargs):
    hyperparams = extract_keys(options)
    if 'end_point_mode' not in hyperparams.keys():
        hyperparams['end_point_mode'] = True

    if options['method'].lower() == 'euler':
        solver = Euler(func, y0=z0, **hyperparams,**kwargs)
    elif options['method'].lower() == 'rk2':
        solver = RK2(func, y0=z0, **hyperparams, **kwargs)
    elif options['method'].lower() == 'rk4':
        solver = RK4(func,y0=z0,**hyperparams, **kwargs)
    elif options['method'].lower() == 'rk12':
        solver = RK12(func,y0=z0,**hyperparams, **kwargs)
    elif options['method'].lower() == 'rk23':
        solver = RK23(func,y0=z0, **hyperparams, **kwargs)
    elif options['method'].lower() == 'dopri5':
        solver = Dopri5(func,y0=z0, **hyperparams, **kwargs)
    elif options['method'].lower() == 'sym12async':
        _tensor_input, _func, _y0 = check_arguments(func, z0, options['t0'])
        v0 = _func(options['t0'], _y0)# [torch.zeros_like(_y0_) for _y0_ in _y0] #_func(options['t0'], _y0)
        initial_condition = tuple(list(_y0) + list(v0))
        solver = Sym12Async(func=_func, y0=_y0, **hyperparams, **kwargs)
    elif options['method'].lower() == 'fixedstep_sym12async':
        _tensor_input, _func, _y0 = check_arguments(func, z0, options['t0'])
        v0 = _func(options['t0'], _y0)#[torch.zeros_like(_y0_) for _y0_ in _y0]
        initial_condition = tuple(list(_y0) + list(v0))
        solver = FixedStep_Sym12Async(func = _func, y0 = _y0, **hyperparams, **kwargs)
    elif options['method'].lower() == 'ode23s':
        solver = ODE23s(func=func, y0=z0,   **hyperparams, **kwargs)
    else:
        print('Name of solver not found.')

    if return_solver: # return solver
        return solver
    else: # return integrated value
        # normal methods
        if options['method'].lower() not in ['leapfrog', 'yoshida', 'sym12', 'sym23',
                                             'sym34', 'sym12async', 'adalf', 'fixedstep_adalf',
                                             'fixedstep_sym12async']:
            z1 = solver.integrate(z0, t0 = options['t0'], t_eval = options['t1'])
            return z1
        # symplectic methods
        else: # need to use tuple(y,v) as initial condition instead of y
            z1 = solver.integrate(y0=initial_condition, t0=options['t0'], t_eval=[options['t1']])
            out = z1[0:len(_y0)]
            if len(out) == 1:
                out = out[0]

            return out
