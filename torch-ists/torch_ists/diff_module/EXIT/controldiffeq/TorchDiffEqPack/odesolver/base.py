import abc
import torch
import copy
import numpy as np
from torch import nn
from ..misc import interp_cubic_hermite_spline, flatten, delete_local_computation_graph
from ..utils import monotonic

class FuncWrapper(nn.Module):
    """
    Wrapper to convert an tensor-input tensor-output function into tuple-input tuple-output
    """
    def __init__(self, func_tensor):
        super(FuncWrapper, self).__init__()
        self.func_tensor = func_tensor
    def forward(self, t, y):
        return ( self.func_tensor(t,y[0]), )

def check_arguments(func, y0, t):
    tensor_input = False
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = (y0,)
        func = FuncWrapper(func)
    assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(type(y0_))

    for y0_ in y0:
        if not torch.is_floating_point(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.type()))

    if not isinstance(t, torch.Tensor):
        t = torch.Tensor([float(t)])[0].float().to(y0[0].device)
    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    return tensor_input, func, y0

class ODESolver(nn.Module):
    def __init__(self, func, t0, y0, t1=1.0, h=0.1, rtol=1e-3, atol=1e-6, neval_max=500000,
                 print_neval=False, print_direction=False, step_dif_ratio=1e-3, safety=0.9,
                 regenerate_graph=False, dense_output=True, interpolation_method = 'cubic',
                 print_time = False, end_point_mode = False):
        super(ODESolver, self).__init__()
        """
        ----------------
        :param func: callable
                the function to compute derivative, should be the form  dy/dt = func(t,y), y is a tensor
        :param t0: Tensor of shape 0
                initial time
        :param y0: Tuple
        :param t1:Tensor of shape 0
                ending time
        :param h: float
                initial stepsize, could be none
        :param rtol: float
                relative error tolerance
        :param atol: float
                absolute error tolerance
        :param neval_max: int
                maximum number of evaluations, typically set as an extermely large number, e.g. 500,000
        :param print_neval: bool
                print number of evaluations or not
        :param print_direction: bool
                print direction of time (if t0 < t1, print 1; if t0 > t1, print -1)
        :param step_dif_ratio: float
                A ratio to avoid dead loop.
                if abs(old_step_size - new_step_size) < step_dif_ratio AND error > tolerance,
                then accept current stepsize and continue
        :param safety: float,
                same as scipy.odeint, used to adjut stepsize
        :param regenerate_graph, bool, whether re-generate computation graph using calculated grids
        :param dense_output, bool, whether store dense outputs
        :param interpolation_method, string, "cubic" for cubic Hermite spline interpolation, 
               "linear" for linear interpolation
        ----------------
        """

        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0).to(y0[0].device).float()

        if not isinstance(t1, torch.Tensor):
            t1 = torch.tensor(t1).to(y0[0].device).float()

        self.t0 = t0.to(y0[0].device)
        self.t1 = t1.to(y0[0].device)
        self.h = h
        self.rtol = rtol
        self.atol = atol
        self.neval_max = neval_max
        self.print_neval = print_neval
        self.neval = 0
        # if two stepsizes are too similar, not update it, otherwise stuck in a loop
        self.step_dif_ratio = step_dif_ratio
        self.regenerate_graph = regenerate_graph
        self.dense_output = dense_output
        self.safety = safety
        self.print_time = print_time

        assert interpolation_method in ['linear', 'cubic', 'polynomial'], 'interpolation method must be in ["linear", "cubic"]'
        self.interpolation_method = interpolation_method

        self.tensor_input, self.func, self.y0 = check_arguments(func, y0, t0)

        if self.dense_output:
            if not hasattr(self, 'dense_states'):
                self.init_dense_states()

        # set time direction, forward-in-time is 1.0, reverse-time is -1.0
        if self.t1 > self.t0:
            self.time_direction = 1.0
            if print_direction:
                print('Forward-time integration')
        else:
            self.time_direction = -1.0
            if print_direction:
                print("Reverse-time integration")

        self.end_point_mode = end_point_mode

    def check_t(self, t_eval):
        if t_eval is  None:
            t_eval = self.t1

        if isinstance(t_eval, float):
            t_eval = torch.tensor([t_eval]).float().to(self.y0[0].device)
        elif isinstance(t_eval, list) and not torch.is_tensor(t_eval[0]):
            t_eval = torch.from_numpy(np.asarray(t_eval)).float().to(self.y0[0].device)
        elif isinstance(t_eval, list) and torch.is_tensor(t_eval[0]):
            t_eval = [_t_eval.float().to(self.y0[0].device).view(-1) for _t_eval in t_eval]
            t_eval = torch.stack(t_eval, 0)
        elif isinstance(t_eval, np.ndarray):
            t_eval = torch.from_numpy(t_eval).float().to(self.y0[0].device)

        t_eval = t_eval.view(-1)
        return t_eval

    def before_integrate(self, y0, t_eval):

        t_eval = self.check_t(t_eval)

        assert isinstance(t_eval, torch.Tensor), 't_eval should be a tensor'

        self.t_eval_ind = 0 # index for evaluation time points

        if (t_eval is not None) and (t_eval.numel() > 0):
            self.t_eval = t_eval
            assert (self.t1 - t_eval[-1]) * (t_eval[0] - self.t0) >= 0, \
                'value of t_eval must be within t0<= t_eval <= t1'
            if t_eval.numel() > 1:
                assert monotonic(t_eval.data.cpu().numpy().tolist()), 't_eval muist be monotonic'
                assert (t_eval[-1] - t_eval[0]) * self.time_direction > 0, \
                    't_eval must be arranged in the same direction as [t0, t1]'
            self.t_end = self.t_eval[self.t_eval_ind]
        else:
            self.t_end = self.t1
            self.t_eval = t_eval

    def update_t_end(self):
        # update t_end
        if self.t_eval is None or self.t_eval_ind == (self.t_eval.numel()-1):
            self.t_end = None
        else:
            self.t_eval_ind = self.t_eval_ind + 1
            self.t_end = self.t_eval[self.t_eval_ind]

    def init_dense_states(self):
        self.dense_states = {
            't_start': [],
            't_end':[],
            'y_start':[],
            'y_end':[],
            'variables':[],
            'coefficients':[],
        }

    def delete_dense_states(self):
        if len(self.dense_states) > 0:
            if len(self.dense_states['t_start']) > 0:
                self.dense_states['t_start'].clear()
            if len(self.dense_states['t_end']) > 0:
                self.dense_states['t_end'].clear()
            if len(self.dense_states['y_start']) > 0:
                delete_local_computation_graph( flatten(self.dense_states['y_start'][1:]) )
            if len(self.dense_states['y_end']) > 0:
                delete_local_computation_graph( flatten(self.dense_states['y_end'][1:]) )
            if len(self.dense_states['variables']) > 0:
                delete_local_computation_graph( flatten(self.dense_states['variables'][1:]) )
            if len(self.dense_states['coefficients']) > 0:
                delete_local_computation_graph( flatten(self.dense_states['coefficients'][1:]) )

    def interpolate(self, t_old, t_new, t_eval, y0, y1, k=None, **kwargs):
        if self.interpolation_method == 'linear':
            # linear interpolation
            outs = tuple( (t_eval - t_old).expand_as(_y0) * (_y1 - _y0) / (t_new - t_old).expand_as(_y0) + _y0 for
                         _y0, _y1 in zip(y0, y1))
        elif self.interpolation_method == 'cubic':
            # Hermite cubic spline interpolation
            y_start, y_end = y0, y1
            times = torch.stack([t_old, t_new]).view(-1).to(y_start[0].device)

            outs = []
            for _y_start, _y_end in zip(y_start, y_end):
                points = torch.stack((_y_start, _y_end), 0)
                outs.append( interp_cubic_hermite_spline(times, points,t_eval)[0] )
            outs = tuple(outs)
        elif self.interpolation_method == 'polynomial':
            assert hasattr(self, 'P') and hasattr(self, 'n_stages'), 'Polynomial interpolation requires a "P" matrix and "n_stages", ' \
                                       'currently only supported for RK23 and Dopri5; for other solvers please choose' \
                                       'interpolation method from ["linear","cubic"]'
            # compute Q, correspond to _dense_output_impl in scipy
            # Q = self.K.T.dot(self.P)
            K = []
            for j in range(len(k[0])): # for j-th variable
                tmp = []
                for i in range(len(k)): # for i-th stage
                    tmp.append(k[i][j])
                K.append( torch.stack(tmp, dim=1))
            K = tuple( K)  # Nx(n_stages+1)x...
            shapes = tuple(_y0.shape for _y0 in y0)
            K = tuple( _K.view(_shape[0], self.n_stages + 1, -1) for _K, _shape in zip(K, shapes) )  # Nx(n_stages +1)x-1
            K = tuple( _K.permute(0, -1, 1) for _K in K)  # Nx -1 x (n_stages+1)

            # self.P.shape = (n_stages+1)xn_stages
            Q = tuple( torch.matmul(_K, self.P.to(_y0.device)) for _K, _y0 in zip(K, y0))  # Nx-1xn_stages

            x = abs(t_eval - t_old) / abs(t_new - t_old)
            if np.array(t_eval).ndim == 0:
                p = tuple( np.tile(x, _Q.shape[-1]) for _Q in Q)
                p = tuple( np.cumprod(_p) for _p in p)  # p.shape = n_stages

            p = tuple( torch.from_numpy(_p).float().to(_y0.device)  for _p, _y0 in zip(p, y0))  # n_stages

            dif = tuple( float(t_new - t_old) * torch.matmul(_Q, _p) for _Q, _p in zip(Q, p))  # Nx-1
            dif = tuple( _dif.view(_y0.shape) for _dif, _y0 in zip(dif, y0))

            out = tuple( _y0 + _dif for _y0, _dif in zip(y0, dif))
            return out
        else:
            print('interpolation method must be in ["linear","cubic"], current is {}'.format( self.interpolation_method) )
        return outs

    def update_dense_state(self, t_old, t_new, y_old, y_new, save_current_step = True):
        if self.dense_output and save_current_step:
            self.dense_states['t_start'].append(t_old)
            self.dense_states['t_end'].append(t_new)
            self.dense_states['y_start'].append(y_old)
            self.dense_states['y_end'].append(y_new)

    def concate_results(self, inputs):
        """
        inputs = [ tuple1(tensor1, tensor2, .. tensorm),
                 tuple2(tensor1, tensor2, ... tensorm),
                 ...
                tupleN(tensor1, tensor2, ... tensorm)]
        if inputs has only one input,
             outs = [ tuple1(tensor1, tensor2, .. tensorm)]
             return tuple1(tensor1, tensor2, .. tensorm)
        else:
             output = tuple( N x tensor1, N x tensor2, ... N x tensorm )

        :param inputs: outs is a list of tuples. N time points hence N tuples, each has m tensors of shape xxx
        :return: a tuple, each has m tensors, of shape N x xxx
        """
        # concatenate into a tensor
        if len(inputs) == 1:
            out = inputs[0]
        elif len(inputs) > 1:
            out = []
            if isinstance( inputs[0], tuple):
                for i in range(len(inputs[0])): # for i-th tensor in a tuple
                    out.append( torch.stack( [ _tmp[i] for _tmp in inputs ], 0 ) )
                out = tuple(out)
            elif torch.is_tensor(inputs[0]):
                out = inputs
        else:
            out = None
            print('Error, Length of evaluated results is 0, please check')
        return out

    def evaluate_dense_mode(self, t_eval, scipy_mode = True, **kwargs):# evaluate at time points in t_eval, with dense mode.
        all_evaluations = []

        t_eval = self.check_t(t_eval)

        for _iter in range(t_eval.numel()):
            _t_eval = t_eval[_iter]
            # find the correct interval for t
            ind = 0
            ind_found = False
            while ind < len(self.dense_states['t_start']):
                t_start, t_end = self.dense_states['t_start'][ind], self.dense_states['t_end'][ind],
                if torch.abs(t_end - self.t0) > torch.abs(_t_eval - self.t0) and \
                    torch.abs(t_start - self.t0) <= torch.abs(_t_eval - self.t0):
                    ind_found = True
                    break
                else:
                    ind += 1

            if not ind_found:
                print('Evaluation time: {} outside integration range.'.format(_t_eval))
                if torch.abs(self.dense_states['t_start'][0] - _t_eval) > torch.abs(self.dense_states['t_start'][-1] - _t_eval):
                    ind = -1
                    print('Extrapolate using the last interval')
                else:
                    ind = 0
                    print('Extrapolate using the first interval')

            # evaluate by cubic spline interpolation
            t_start, t_end = self.dense_states['t_start'][ind], self.dense_states['t_end'][ind]
            y_start, y_end = self.dense_states['y_start'][ind], self.dense_states['y_end'][ind]

            all_evaluations.append(self.interpolate(t_start, t_end, _t_eval, y_start, y_end))

        out = self.concate_results(all_evaluations)
        if self.tensor_input:
            out = out[0]
        return out

    def integrate(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def integrate_predefined_grids(self, y0, t0, predefine_steps=None, return_steps=False, t_eval=None):

        if torch.is_tensor(y0):
            y0 = (y0,)
            self.tensor_input = True
            self.y0 = y0

        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(float(t0)).float().to(self.y0[0].device)

        if len(t0.shape) > 0:
            t0 = t0[0]
        t0 = t0.float().to(self.y0[0].device)
        self.t0 = t0

        if t_eval is not None:
            t_eval = self.check_t(t_eval)

        if isinstance(predefine_steps, list):
            predefine_steps = torch.from_numpy(np.asarray(predefine_steps)).float().to(self.y0[0].device)

        assert isinstance(predefine_steps, torch.Tensor), 'Predefined steps can be a list, but later must be converted to a Tensor'
        predefine_steps = predefine_steps.float().to(self.y0[0].device)

        all_evaluations = []
        # print(len(predefine_steps))
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)

        self.before_integrate(y0, t_eval)

        time_points = predefine_steps # time points to evaluate, not the step

        # advance a small step in time
        t_current = self.t0
        y_current = y0
        # print(steps)
        for _iter in range(time_points.numel()):
            point = time_points[_iter]
            self.neval += 1
            y_old = y_current
            # print(y_current.shape)

            # time passed into step function must be of type Tensor with shape None
            y_current, error, variables = self.step(self.func, t_current, (point - t_current), y_current,
                                                    return_variables=True)

            if not self.end_point_mode:
                self.update_dense_state(t_current, point, y_old, y_current)
                while (self.t_end is not None) and torch.abs(point - self.t0) >= torch.abs(self.t_end - self.t0) and \
                        torch.abs(t_current - self.t0) <= torch.abs(self.t_end - self.t0):  # if next step is beyond integration time
                    # interpolate and record output
                    all_evaluations.append(
                        self.interpolate(t_current, point, self.t_end, y_old, y_current)
                    )
                    #print(self.t_end)
                    self.update_t_end()

            t_current = point
            if self.print_time:
                print(t_current)

        # if have points outside the integration range
        while self.t_end is not None and not self.end_point_mode:
            print('Evaluation points outside integration range. Please re-specify t0 and t1 s.t. t0 < t_eval < t1 or t1 < t_eval < t0 STRICTLY, and use a FINER grid.')
            if not self.dense_output:
                print('DenseOutput mode is not enabled. ')
            else:
                print('Extrapolate in dense mode')
                tmp = self.evaluate_dense_mode([self.t_end])
                if self.tensor_input:
                    tmp = (tmp, )
                all_evaluations.append(tmp)
            self.update_t_end()

        if self.end_point_mode:
            all_evaluations = y_current

        out = self.concate_results(all_evaluations)
        if self.tensor_input:
            if not torch.is_tensor(out):
                out = out[0]
        return out