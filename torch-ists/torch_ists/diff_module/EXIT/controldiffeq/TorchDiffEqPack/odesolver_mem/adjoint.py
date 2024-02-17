import torch
import torch.nn as nn
from .odesolver_endtime import odesolve_endtime
from torch.autograd import Variable
import copy
from ..misc import delete_local_computation_graph, flatten
from ..odesolver.base import check_arguments
reload_state = False
__all__ = ['odesolve_adjoint']

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
    def forward(ctx, *args):
        z0, func, t0, t1, flatt_param, options =  args[:-5], args[-5] , args[-4] , args[-3] ,args[-2] ,args[-1]

        assert isinstance(z0, tuple)
        if len(z0) == 1: # even if input is a single tensor, z0 is now a tuple due to the code args[:-5]
            z0 = z0[0]

        ctx.func = func
        state0 = func.state_dict()
        ctx.state0 = state0
        ctx.z0 = z0
        ctx.options = options

        with torch.no_grad():
            solver = odesolve_endtime(func, z0, options, return_solver=True, regenerate_graph = False)
            ans, steps = solver.integrate(z0, t0 = options['t0'], return_steps=True)

        ctx.steps = steps
        ctx.t0 = t0
        ctx.t1 = t1

        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        z0, options, func, steps, state0 = ctx.z0, ctx.options, ctx.func, ctx.steps, ctx.state0
        f_params = tuple( flatten(func.parameters()) )

        flat_params = flatten_params(func.parameters())

        if reload_state:
            func.load_state_dict(state0)

        if torch.is_tensor(z0):
            z = (Variable(z0.data, requires_grad = True), )
        else:
            z = tuple([ Variable(_z.data, requires_grad = True) for _z in z0])

        solver = odesolve_endtime(func, z0, options, return_solver=True)

        # record inputs to each step
        inputs = [z]

        t_current = solver.t0
        y_current = z
        for point in steps:
            solver.neval += 1
            with torch.no_grad():
                y_current, error, variables = solver.step(solver.func, t_current, point - t_current, y_current, return_variables=True)
                t_current = point
                inputs.append(tuple( [Variable(_y.data, requires_grad=True) for _y in y_current]) )
                delete_local_computation_graph( flatten(list(error) + list(variables)) )

        # compute gradient w.r.t t1
        func_i = solver.func(solver.t1, y_current)
        dLdt1 = sum(
            torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
            for func_i_, grad_output_i_ in zip(func_i, grad_output)
        )

        ###################################
        # note that steps does not include the start point, need to include it
        steps = [options['t0']] + steps
        # now two list corresponds, steps = [t0, teval1, teval2, ... tevaln, t1]
        #                           inputs = [z0, z1, z2, ... , z_out]
        ###################################

        inputs.pop(-1)
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

        inputs.reverse()
        steps.reverse()
        steps2.reverse()

        assert len(inputs) == len(steps) == len(steps2), print('len inputs {}, len steps {}, len steps2 {}'.format(len(inputs), len(steps), len(steps2)))

        grad_t0 = torch.zeros_like(solver.t0)
        grad_y = tuple( torch.zeros_like(_z0) for _z0 in z)
        grad_flat_param = torch.zeros_like(flat_params)


        for input, point, point2 in zip(inputs, steps, steps2): # step in reverse-time
            if torch.is_tensor(point) and torch.is_tensor(point2):
                point2 = point2.clone().detach().to(y_current[0].device)
                point = point.clone().detach().to(y_current[0].device)
            with torch.enable_grad():
                point.requires_grad = True
                y, error, variables = solver.step(solver.func, point, point2 - point, input, return_variables=True)

                _grad_t, *_grad_intput_and_param = torch.autograd.grad(
                    y, (point,) + input + f_params,
                    grad_output, allow_unused=True)

                delete_local_computation_graph(  flatten( [y, error] + list(variables)) )

                # accumulate gradients
                _grad_t = torch.zeros_like(point) if _grad_t is None else _grad_t
                _grad_y = _grad_intput_and_param[:len(y)]
                grad_output = _grad_y # gradient of input is the gradient of output for previour step
                _grad_params = _grad_intput_and_param[len(y):]

                grad_t0 += _grad_t

                for tmp1, tmp2 in zip(grad_y, _grad_y):
                    tmp1 += tmp2

                grad_flat_param += flatten_params_grad(_grad_params, f_params)

        t0, t1 = ctx.t0, ctx.t1
        if not torch.is_tensor(t0):
            grad_t0 = None
        if not torch.is_tensor(t1):
            dLdt1 = None
        out = tuple([*grad_output] + [None, grad_t0, dLdt1, grad_flat_param ,None])

        del z0, ctx.z0, state0, ctx.state0

        return out


def odesolve_adjoint(func, y0, options = None):
    """
    Implementation of ICML 2020 paper "Adaptive checkpoint adjoint method for accurate gradient esitmation in Neural ODEs"

    How to use:
    
    from TorchDiffEqPack import odesolve_adjoint \n
    options = {}\n
    options.update({'method':method}) # string, method must be in ['euler','rk2','rk12','rk23','dopri5']\n
    options.update({'h': h}) # float, initial stepsize for integration. Must be specified for fixed stepsize solvers; for adaptive solvers, can be set as None, then the solver witll automatically determine the initial stepsize\n
    options.update({'t0': t0}) # float, initial time for integration\n
    options.update({'t1': t1}) # float, end time for integration\n
    options.update({'rtol': rtol}) # float or list of floats (must be same length as y0), relative tolerance for integration, typically set as 1e-5 or 1e-6 for dopri5\n
    options.update({'atol': atol}) # float or list of floats (must be same length as y0), absolute tolerance for integration, typically set as 1e-6 or 1e-7 for dopri5\n
    options.update({'print_neval': print_neval}) # bool, when print number of function evaluations, recommended to set as False\n
    options.update({'neval_max': neval_max}) # int, maximum number of evaluations when encountering stiff problems, typically set as 5e5\n
    options.update({'t_eval': [t0, t0 + (t1-t0)/10, ...  ,t1]}) # Must be None, only output the value at time t1\n

    out = odesolve_adjoint(func, y0, options = options) # func is the ODE; y0 is the initial condition, could be either a tensor or a tuple of tensors
    """

    assert options['method'].lower() not in ['sym12async','fixedstep_sym12async'], 'odesolve_adjoint cannot be used with sym12async method, ' \
                                                            'please use odesolve_adjoint_sym12'
    flat_params = flatten_params(func.parameters())

    _y0 = (y0,) if torch.is_tensor(y0) else tuple(y0)

    t0 = options['t0']
    if not isinstance(t0, torch.Tensor):
        t0 = torch.tensor(float(t0)).float().to(y0[0].device)
    if len(t0.shape) > 0:
        t0 = t0[0]
    t0 = t0.float().to(y0[0].device)
    options['t0'] = t0

    zs = Checkpointing_Adjoint.apply(*_y0, func, options['t0'], options['t1'], flat_params, options)
    return zs
