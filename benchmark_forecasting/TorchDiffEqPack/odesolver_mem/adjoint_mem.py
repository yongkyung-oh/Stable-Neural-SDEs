import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from ..misc import delete_local_computation_graph, flatten
from ..odesolver.base import check_arguments
from ..odesolver.symplectic import *
from ..utils import extract_keys

reload_state = False
__all__ = ['odesolve_adjoint_sym12']

def flatten_params(params):
    flat_params = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])

def flatten_params_grad(params, params_ref):
    _params = [p for p in params]
    _params_ref = [p for p in params_ref]
    flat_params = [p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(_params, _params_ref)]

    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])

class Checkpointing_Adjoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args): # z0 = [y0, v0] for symplectic solvers
        z0, func, t0, t1, flatt_param, options =  args[:-5], args[-5] , args[-4] , args[-3] ,args[-2] ,args[-1]

        assert isinstance(z0, tuple)
        if len(z0) == 1: # even if input is a single tensor, z0 is now a tuple due to the code args[:-5]
            z0 = z0[0]

        ctx.func = func
        state0 = func.state_dict()
        ctx.state0 = state0
        ctx.options = options

        with torch.no_grad():
            hyperparams = extract_keys(options)
            if 'end_point_mode' not in hyperparams.keys():
                hyperparams['end_point_mode'] = True

            if options['method'].lower() == 'sym12async':
                solver = Sym12Async(func=func, y0=z0, **hyperparams)
            elif options['method'].lower() == 'fixedstep_sym12async':
                solver = FixedStep_Sym12Async(func=func, y0=z0, **hyperparams)
            else:
                print('Optimizers for adjoint_mem method can only be in ["sym12async","adalf","fixedstep_sym12async"'
                      ',"fixedstep_adalf"]')
            ans, steps = solver.integrate(z0, t0 = options['t0'], return_steps=True)

        ctx.steps = steps
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.z0 = z0
        ctx.ans = tuple(Variable(_ans.data) for _ans in ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        options, func, steps, state0, ans = ctx.options, ctx.func, ctx.steps, ctx.state0, ctx.ans
        f_params = tuple( flatten(func.parameters()) )

        flat_params = flatten_params(func.parameters())
        if reload_state:
            func.load_state_dict(state0)

        hyperparams = extract_keys(options)
        if 'end_point_mode' not in hyperparams.keys():
            hyperparams['end_point_mode'] = True
        z = ans

        if options['method'].lower() == 'sym12async':
            solver = Sym12Async(func=func, y0=z, **hyperparams)
        elif options['method'].lower() == 'fixedstep_sym12async':
            solver = FixedStep_Sym12Async(func=func, y0=z, **hyperparams)
        else:
            print('Optimizers for adjoint_mem method can only be in ["sym12async",",adalf"]')

        y_current = z

        solver.to(y_current[0].device)
        # compute gradient w.r.t t1
        func_i = solver.func(solver.t1,y_current[:len(y_current)//2])
        dLdt1 = sum(
            torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
            for func_i_, grad_output_i_ in zip(func_i, grad_output)
        )
        
        ###################################
        # note that steps does not include the start point, need to include it
        if torch.is_tensor(steps):
            steps = [steps[j] for j in range(steps.shape[0])]
        
        assert isinstance(steps, list), 'Steps must be of tuple type.'
        steps = [options['t0']] + steps
        # now two list corresponds, steps = [t0, teval1, teval2, ... tevaln, t1]
        #                           inputs = [z0, z1, z2, ... , z_out]
        ###################################
        if torch.is_tensor(steps[0]):
            steps2 = [_step.clone() for _step in steps]
        else:
            steps2 = copy.deepcopy(steps)
        steps2.pop(0)
        steps.pop(-1)

        # steps = [t0, eval1, eval2, ... evaln, t1], after pop is [t0, eval1, ... evaln]
        # steps2 = [t0, eval1, eval2, ... evaln, t1], after pop is [eval1, ... evaln, t1]

        # after reverse, they are
        # steps = [evaln, evaln-1, ... eval2, eval1, t0]
        # steps2 = [t1, evaln, ... eval2, eval1s]

        steps.reverse()
        steps2.reverse()

        assert  len(steps) == len(steps2), print('len steps {}, len steps2 {}'.format( len(steps), len(steps2)))

        grad_t0 = torch.zeros_like(solver.t0).to(solver.t0)
        grad_y = tuple( torch.zeros_like(_z0).to(_z0) for _z0 in z)
        grad_flat_param = torch.zeros_like(flat_params).to(flat_params)

        for point, point2 in zip(steps, steps2): # step in reverse-time
            if torch.is_tensor(point) and torch.is_tensor(point2):
                point2 = point2.clone().detach().to(y_current[0].device)
                point = point.clone().detach().to(y_current[0].device)
            # reconstruct input
            with torch.no_grad():
                
                input, variable_recon = solver.inverse_async( solver.func, point2, point2 - point, y_current)

            input = tuple( [Variable(_input.data, requires_grad = True) for _input in input])

            # local forward
            with torch.enable_grad():
                point.requires_grad = True
                y, error, variables = solver.step(solver.func, point, point2 - point, input, return_variables=True)

                
                _grad_t, *_grad_intput_and_param = torch.autograd.grad(
                    y, (point,) + input + f_params,
                    grad_output, allow_unused=True)

            # delete variables
            delete_local_computation_graph(  flatten( [y, error, y_current] + list(variables) + list(variable_recon) ) )

            # accumulate gradients
            _grad_t = torch.zeros_like(point).to(solver.t0.device) if _grad_t is None else _grad_t
            _grad_y = _grad_intput_and_param[:len(y)]
            _grad_params = _grad_intput_and_param[len(y):]

            grad_t0.data.add_(_grad_t.data)

            for _i,(tmp1, tmp2) in enumerate( zip(grad_y, _grad_y) ):
                if torch.is_tensor(tmp2):
                    tmp1.data.add_(tmp2.data)
                else:
                    _grad_y[_i] = torch.zeros_like(tmp1.data * 0.0).to(y_current[0].device)

            grad_output = _grad_y  # gradient of input is the gradient of output for previour step

            grad_flat_param.data.add_(flatten_params_grad(_grad_params, f_params).data)

            # set up loopable variable
            y_current = input


        t0, t1 = ctx.t0, ctx.t1
        if not torch.is_tensor(t0):
            grad_t0 = None
        else:
            grad_t0.to(t0.device)
        if not torch.is_tensor(t1):
            dLdt1 = None
        else:
            dLdt1.to(t1.device)
        out = tuple([*grad_output] + [None, grad_t0, dLdt1, grad_flat_param,None])

        del  state0, ctx.state0, ans, ctx.ans, ctx.z0

        return out


def odesolve_adjoint_sym12(func, y0, options = None):
    """
    
    How to use:
    
    from TorchDiffEqPack import odesolve_adjoint_sym12 \n
    options = {} \n
    options.update({'method':method}) # string, method must be in ['sym12async', 'fixedstep_sym12async']\n
    options.update({'h': h}) # float, initial stepsize for integration. Must be specified for "fixedstep_sym12async"; for "sym12async", can be set as None, then the solver witll automatically determine the initial stepsize\n
    options.update({'t0': t0}) # float, initial time for integration\n
    options.update({'t1': t1}) # float, end time for integration\n
    options.update({'rtol': rtol}) # float or list of floats (must be same length as y0), relative tolerance for integration, typically set as 1e-2 or 1e-3 for MALI\n
    options.update({'atol': atol}) # float or list of floats (must be same length as y0), absolute tolerance for integration, typically set as 1e-3 for MALI\n
    options.update({'print_neval': print_neval}) # bool, when print number of function evaluations, recommended to set as False\n
    options.update({'neval_max': neval_max}) # int, maximum number of evaluations when encountering stiff problems, typically set as 5e5\n
    options.update({'t_eval': [t0, t0 + (t1-t0)/10, ...  ,t1]}) # list of float; if is None, then the output is the value at time t1\n

    out = odesolve_adjoint_sym12(func, y0, options = options) # func is the ODE; y0 is the initial condition, could be either a tensor or a tuple of tensors\n
    """
    assert options['method'].lower() in ['sym12async', 'fixedstep_sym12async'], \
        'odesolve_adjoint_sym12 must be used together with Sym12Async'

    flat_params = flatten_params(func.parameters())

    _tensor_input, _func, _y0 = check_arguments(func, y0, options['t0'])

    if options['method'].lower() in ['sym12async', 'fixedstep_sym12async']:
        initial_condition = tuple(list(_y0) + list(_func(options['t0'], _y0)))
    elif options['method'].lower() in ['adalf', 'fixedstep_adalf']:
        v0 = _func(options['t0'], _y0)
        initial_condition = tuple( list(_y0) + list(v0) + list(v0) )

    t0 = options['t0']
    if not isinstance(t0, torch.Tensor):
        t0 = torch.tensor(float(t0)).float().to(y0[0].device)
    if len(t0.shape) > 0:
        t0 = t0[0]
    t0 = t0.float().to(y0[0].device)
    options['t0'] = t0

    zs = Checkpointing_Adjoint.apply(*initial_condition, _func, options['t0'], options['t1'], flat_params, options)

    out = zs[0:len(_y0)]
    if _tensor_input:
        out = out[0]
    return out
