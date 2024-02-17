import torch
from .adaptive_grid_solver import AdaptiveGridSolver
from .autograd_functional import jacobian, hessian
from .tuple_to_tensor_wrapper import tuple_to_tensor, tensor_to_tuple, TupleFuncToTensorFunc
import numpy as np

__all__ = ['ODE23s']

SAFETY = 0.9
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.
EPS = 1e-8
class ODE23s(AdaptiveGridSolver):

    def __init__(self, func, t0, y0, t1=1.0, h=0.1, rtol=1e-3, atol=1e-6, neval_max=500000,
                 print_neval=False, print_direction=False, step_dif_ratio=1e-3, safety=SAFETY,
                 regenerate_graph=False, dense_output=True, interpolation_method = 'cubic',
                 print_time = False, end_point_mode = False):
        super(ODE23s, self).__init__(func=func, t0=t0, y0=y0, t1=t1, h=h, rtol=rtol,
                                              atol=atol, neval_max=neval_max,
                 print_neval=print_neval, print_direction=print_direction, step_dif_ratio=step_dif_ratio, safety=safety,
                 regenerate_graph=regenerate_graph, dense_output=dense_output, interpolation_method = interpolation_method,
                                     print_time = print_time, end_point_mode = end_point_mode)
        self.order = 2

        # constants
        self._d = float(1 / (2 + np.sqrt(2)))
        self._e32 =float( 6 + np.sqrt(2) )

    def step(self, func, t, dt, y, return_variables=False):
        if y[0].requires_grad == False:
            create_graph = False
        else:
            create_graph = True
        # convert to tensor for easer calculation of jacobian
        shapes, y_concats = tuple_to_tensor(y)
        func_tensor = TupleFuncToTensorFunc(func, shapes) # a function with batch input and batch output

        # calculate jacobian and time derivative for each element in the batch
        T, W = [], []
        for _iter in range(y_concats.shape[0]): # loop for each item in the batch
            # calculate jacobian
            j_all = jacobian(func_tensor, (t, y_concats[_iter, ...].unsqueeze(0)), create_graph=create_graph)

            # time derivative
            T.append( dt * self._d * j_all[0] )

            # jacobian
            W.append( torch.eye(y_concats.shape[1]) - dt * self._d * j_all[1].squeeze() )

        T, W = torch.cat(T, 0), torch.stack(W,0) # T.shape = Nx-1, W.shape = Nx-1x-1

        # modified rosenbrock formula
        W_inv = torch.inverse(W ) # W_inv.shape = Nx-1x-1

        # copy Julia version
        F0 = func_tensor(t, y_concats) # Nx-1
        k1 = torch.matmul(W_inv, (F0 + T).unsqueeze(-1) ) # N x -1 x 1
        k1 = torch.squeeze(k1, -1) # N x-1

        F1 = func_tensor(t + 0.5 * dt, y_concats + 0.5 * dt * k1) # N x -1
        k2 = torch.matmul(W_inv, (F1 - k1).unsqueeze(-1) ) # N x -1 x 1
        k2 = torch.squeeze(k2, -1) + k1 # N x -1

        y_new = y_concats + k2 * dt

        F2 = func_tensor(t + dt, y_new) # N x -1
        k3 = (F2 - self._e32*(k2 - F1) - 2*(k1 - F0) + T ) # N x -1 x 1
        k3 = torch.matmul(W_inv, k3.unsqueeze(-1))
        k3 = k3.squeeze(-1) # N x -1

        error = dt / 6 * (k1 - 2*k2 + k3)

        if return_variables:
            return tensor_to_tuple(shapes,y_new), tensor_to_tuple(shapes,error), \
                   [tensor_to_tuple(shapes,k1), tensor_to_tuple(shapes,k2), tensor_to_tuple(shapes,k3)]
        else:
            return  tensor_to_tuple(shapes,y_new), tensor_to_tuple(shapes,error)
