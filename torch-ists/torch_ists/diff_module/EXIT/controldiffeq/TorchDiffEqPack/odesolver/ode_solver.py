
from .fixed_grid_solver import *
from .adaptive_grid_solver import *
from ..utils import extract_keys
from .stiff_ode_solver import *
from .symplectic import *
import torch
from .base import check_arguments

__all__ = ['odesolve']

def odesolve(func, y0, options, return_solver=False, **kwargs):
    r"""
    Implementation of ICML 2020 paper "Adaptive checkpoint adjoint method for accurate gradient esitmation in Neural ODEs"

    How to use:
    
    * from TorchDiffEqPack import odesolve
    * options = {} 
    * options.update({'method':method}) : string, 'method' must be in ['euler','rk2','rk12','rk23','dopri5', 'ode23s'], 'ode23s' for stiff ODEs 
    * options.update({'h': h}) : float, initial stepsize for integration. Must be specified for fixed stepsize solvers; for adaptive solvers, can be set as None, then the solver witll automatically determine the initial stepsize 
    * options.update({'t0': t0}) : float, initial time for integration 
    * options.update({'t1': t1}) : float, end time for integration 
    * options.update({'rtol': rtol}) : float or list of floats (must be same length as y0), relative tolerance for integration, typically set as 1e-5 or 1e-6 for dopri5 
    * options.update({'atol': atol}) : float or list of floats (must be same length as y0), absolute tolerance for integration, typically set as 1e-6 or 1e-7 for dopri5 
    * options.update({'print_neval': print_neval}) : bool, when print number of function evaluations, recommended to set as False 
    * options.update({'neval_max': neval_max}) : int, maximum number of evaluations when encountering stiff problems, typically set as 5e5 
    * options.update({'t_eval': [t0, t0 + (t1-t0)/10, ...  ,t1]}) : Evaluation time points, a list of float; if is None, only output the value at time t1 

    * out = odesolve(func, y0, options = options) : func is the ODE; y0 is the initial condition, could be either a tensor or a tuple of tensors
    """
    hyperparams = extract_keys(options)

    if options['method'].lower() == 'euler':
        solver = Euler(func=func, y0=y0,  **hyperparams, **kwargs)
    elif options['method'].lower() == 'rk2':
        solver = RK2(func=func, y0=y0,  **hyperparams, **kwargs)
    elif options['method'].lower() == 'rk4':
        solver = RK4(func=func, y0=y0,   **hyperparams, **kwargs)
    elif options['method'].lower() == 'rk12':
        solver = RK12(func=func, y0=y0,   **hyperparams, **kwargs)
    elif options['method'].lower() == 'rk23':
        solver = RK23(func=func, y0=y0, **hyperparams, **kwargs)
    elif options['method'].lower() == 'dopri5':
        solver = Dopri5(func=func, y0=y0,   **hyperparams, **kwargs)
    elif options['method'].lower() == 'ode23s':
        solver = ODE23s(func=func, y0=y0,   **hyperparams, **kwargs)
    elif options['method'].lower() == 'sym12async':
        _tensor_input, _func, _y0 = check_arguments(func, y0, options['t0'])
        initial_condition = tuple(list(_y0) + list(_func(options['t0'], _y0)))
        solver = Sym12Async(func = _func, y0 = _y0, **hyperparams, **kwargs)
    elif options['method'].lower() == 'fixedstep_sym12async':
        _tensor_input, _func, _y0 = check_arguments(func, y0, options['t0'])
        initial_condition = tuple(list(_y0) + list(_func(options['t0'], _y0)))
        solver = FixedStep_Sym12Async(func = _func, y0 = _y0, **hyperparams, **kwargs)
    else:
        print('Name of solver not found.')

    if return_solver:  # return solver
        return solver
    else:  # return integrated value
        if options['method'].lower() not in  ['sym12async','fixedstep_sym12async']:
            if 't_eval' in options.keys():
                #assert isinstance(options['t_eval'], list), "t_eval must be list type or None"
                z1 = solver.integrate(y0=y0,t0=options['t0'], t_eval = options['t_eval'])
            else:
                z1 = solver.integrate(y0=y0,t0=options['t0'], t_eval = [options['t1']])
            return z1

        elif options['method'].lower() in  ['sym12async','fixedstep_sym12async']: # need to use tuple(y,v) as initial condition instead of y
            if 't_eval' in options.keys():
                #assert isinstance(options['t_eval'], list), "t_eval must be list type or None"
                z1 = solver.integrate(y0=initial_condition,t0=options['t0'], t_eval = options['t_eval'])
            else:
                z1 = solver.integrate(y0=initial_condition,t0=options['t0'], t_eval = [options['t1']])

            out = z1[0:len(_y0)]
            if len(out) == 1:
                out = out[0]
            return out
