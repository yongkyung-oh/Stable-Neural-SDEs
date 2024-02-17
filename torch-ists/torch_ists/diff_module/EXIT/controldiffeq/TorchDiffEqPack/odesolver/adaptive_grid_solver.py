import abc
import torch
import copy
import numpy as np
from torch.autograd import Variable
from torch import nn
from .base import ODESolver
from ..misc import norm, delete_local_computation_graph, flatten, _is_iterable
__all__ = ['RK12', 'RK23', 'Dopri5']

SAFETY = 0.9
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.
EPS = 0.0

reload_state = False
class AdaptiveGridSolver(ODESolver):

    def __init__(self, func, t0, y0, t1=1.0, h=0.1, rtol=1e-3, atol=1e-6, neval_max=500000,
                 print_neval=False, print_direction=False, step_dif_ratio=1e-3, safety=SAFETY,
                 regenerate_graph=False, dense_output=True, interpolation_method = 'cubic', print_time = False,
                 end_point_mode = False):
        '''
        If end_point_mode is set as True, evaluated at t0 <= s1, s2, s3, ..., sn = t1, return value at t1 without interpolation
        '''
        if safety is None:
            safety = SAFETY

        self.end_point_mode = end_point_mode
        if end_point_mode:
            assert t1 is not None, 't1 must be specified in end-point mode in adaptive solvers'
        super(AdaptiveGridSolver, self).__init__(func=func, t0=t0, y0=y0, t1=t1, h=h, rtol=rtol,
                                              atol=atol, neval_max=neval_max,
                 print_neval=print_neval, print_direction=print_direction, step_dif_ratio=step_dif_ratio, safety=safety,
                 regenerate_graph=regenerate_graph, dense_output=dense_output, interpolation_method = interpolation_method,
                                                 print_time=print_time, end_point_mode = end_point_mode)

    def select_initial_step_scipy(self, t0, y0, f0):
        """Empirically select a good initial step.
        The algorithm is described in [1]_.
        Parameters
        ----------
        fun : callable
            Right-hand side of the system.
        t0 : float
            Initial value of the independent variable.
        y0 : Tuple
            Initial value of the dependent variable.
        f0 : Tuple
            Initial value of the derivative, i. e. ``fun(t0, y0)``.
        direction : float
            Integration direction.
        order : float
            Error estimator order. It means that the error controlled by the
            algorithm is proportional to ``step_size ** (order + 1)`.
        rtol : float
            Desired relative tolerance.
        atol : float
            Desired absolute tolerance.
        Returns
        -------
        h_abs : float
            Absolute value of the suggested initial step.
        References
        ----------
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
               Equations I: Nonstiff Problems", Sec. II.4.
        """
        self.rtol = self.rtol if _is_iterable(self.rtol) else [self.rtol] * len(y0)
        self.atol = self.atol if _is_iterable(self.atol) else [self.atol] * len(y0)

        scale = tuple( _atol + torch.abs(_y0) * _rtol + EPS for _y0, _rtol, _atol in zip(y0, self.rtol, self.atol) )
        #import pdb
        #pdb.set_trace()
        d0 = norm(tuple(_y0 / _scale for _y0, _scale in zip(y0, scale)  ) )
        d1 = norm(tuple(_f0 / _scale for _f0, _scale in zip(f0, scale)  ) )
        if d0.item() < 1e-5 or d1.item() < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = tuple( _y0 + h0 * self.time_direction * _f0 for _y0, _f0 in zip(y0, f0) )
        f1 = self.func(t0 + h0 * self.time_direction, y1)
        d2 = norm(    tuple( (_f1 - _f0) / _scale for _f1, _f0, _scale in zip(f0, f1, scale) )        ) / h0

        if d1.item() <= 1e-15 and d2.item() <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1.item(), d2.item())) ** (1 / (self.order + 1))

        return min(100 * h0, h1)

    def adapt_stepsize(self, y, y_new, error, h_abs, step_accepted, step_rejected):
        """
        Adaptively modify the step size, code is modified from scipy.integrate package
        :param y: tuple
        :param y_new: tuple
        :param error: tuple
        :param h_abs: step size, float
        :return: step_accepted: True if h_abs is acceptable. If False, set it as False, re-update h_abs
                 h_abs:  step size
        """
        self.rtol = self.rtol if _is_iterable(self.rtol) else [self.rtol] * len(y)
        self.atol = self.atol if _is_iterable(self.atol) else [self.atol] * len(y)

        scale = tuple( _atol + torch.max(torch.abs(_y), torch.abs(_y_new)) * _rtol + EPS
                       for _y, _y_new, _atol, _rtol in zip(y, y_new, self.atol, self.rtol)  )

        # print(scale)
        # error_norm = norm(error) / norm(scale)
        # print(error_norm)
        error_norm = norm(  tuple(_error / _scale  for _error, _scale in zip(error, scale)) ).item()

        '''
        if error_norm == 0.0:
            factor = MAX_FACTOR
            step_accepted = True

        elif error_norm < 1:
            factor = min(MAX_FACTOR, max(1, self.safety * error_norm ** (-1 / (self.order + 1))))
            step_accepted = True

        else:
            factor = max(MIN_FACTOR, self.safety * error_norm ** (-1 / (self.order + 1)))
            step_accepted = False
        
        h_abs *= factor
        '''


        if error_norm < 1:
            if error_norm == 0:
                factor = MAX_FACTOR
            else:
                factor = min(MAX_FACTOR, SAFETY * error_norm ** (-1 / (self.order + 1)))

            if step_rejected:
                factor = min(1, factor)

            h_abs *= factor

            step_accepted = True
        else:
            h_abs *= max(MIN_FACTOR,
                         SAFETY * error_norm ** (-1 / (self.order + 1)))
            step_rejected = True

        if torch.is_tensor(h_abs):
            h_abs = float(h_abs.item())

        return h_abs, step_accepted, step_rejected

    def integrate(self, y0, t0, predefine_steps=None, return_steps=False, t_eval=None):
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

        ################################################################################
        #                    If integrate using predefined grids                       #
        ################################################################################
        if predefine_steps is not None:
            if isinstance(predefine_steps, list):
                predefine_steps = torch.from_numpy(np.asarray(predefine_steps)).float().to(self.y0[0].device)

            assert isinstance(predefine_steps, torch.Tensor), 'Predefined steps can be a list, but later must be converted to a Tensor'
            predefine_steps = predefine_steps.float().to(self.y0[0].device)
            out = self.integrate_predefined_grids(y0, t0, predefine_steps=predefine_steps, t_eval=t_eval)
            steps = predefine_steps
        else:
            out, steps = self.integrate_search_grids(y0, t0, t_eval = t_eval)

        if return_steps:
            return out, steps
        else:
            return out

    def integrate_search_grids(self, y0, t0, return_steps=True,t_eval = None, *args,**kwargs):
        self.t0 = t0
        self.y0 = y0
        ###############################################################################
        #         before integrate, initialize, choose initial stepsize               #
        ###############################################################################

        self.before_integrate(y0, t_eval)

        is_stiff = False

        state0 = self.func.state_dict()
        y_current = y0
        t_current = self.t0

        if self.h is not None:
            h_current = self.h
        else:
            # select initial step
            y0_clone = tuple( Variable(_y0.clone().detach(), requires_grad = False) for _y0 in y0 )

            with torch.no_grad():
                _f0 = self.func(self.t0, y0_clone)
                h_current = self.select_initial_step_scipy(t_current, y0_clone, _f0)

            delete_local_computation_graph(flatten([y0_clone, _f0]))

        self.neval = 0  # number of evaluation steps

        if reload_state:
            self.func.load_state_dict(state0)

        #####################################################################################
        #             Step forward in time, if steps are not predefined                     #
        #####################################################################################

        steps = []
        all_evaluations = []  # record outputs at t_eval

        # keep advancing a small step in time
        # merge two types of conditions, first for non end_point mode, second for end_point mode
        while ( (self.t_end is not None) and self.neval < self.neval_max and not self.end_point_mode) or \
                (abs(t_current-self.t0) <= abs(self.t1-self.t0) and abs(t_current + h_current * self.time_direction -self.t0) < abs(self.t1-self.t0)
                 and self.neval < self.neval_max and self.end_point_mode):
            # if not self.keep_small_step:
            step_accepted = False
            step_rejected = False

            self.neval += 1
            h_new = h_current
            state0 = self.func.state_dict()
            n_try = 0

            #########################################################################
            #                   Determine optimal stepsize                          #
            #########################################################################
            while not step_accepted:
                n_try += 1

                if n_try >= self.neval_max:  # if is stiff, use predefined stepsize, not sure if this works well
                    is_stiff = True

                if is_stiff:
                    h_new = min(self.h, abs(self.t1 - t_current))
                    step_accepted = True
                    print('Stiff problem, please use other solvers')

                #####################################################################
                #                   Delete redundant computation graph              #
                #####################################################################

                # detach y in order to avoid extra unused computation graphs
                with torch.no_grad():

                    y_detach = tuple( Variable(_y_current.clone().detach(), requires_grad = False) for _y_current in y_current)

                    h_current = h_new  # .clone().detach()

                    _y_new, _error, _variables = self.step(self.func, t_current, h_current * self.time_direction,
                                                           y_detach, return_variables=True)

                    h_new, step_accepted, step_rejected = self.adapt_stepsize(y_detach, _y_new, _error, h_current,
                                                               step_accepted=step_accepted, step_rejected=step_rejected)

                    if not step_accepted:
                        if abs(h_new - h_current) / (h_current) < self.step_dif_ratio:
                            step_accepted = True

                    # print(h_new)

                    delete_local_computation_graph(flatten([y_detach, _y_new, _error] + list(_variables)))

                # restore state dict to before integrate
                if reload_state:
                    self.func.load_state_dict(state0)

            ##########################################################################
            #                         step forward                                   #
            ##########################################################################
            if self.print_time:
                print(t_current)
            self.h = h_current
            y_old = y_current
            y_current, error, variables = self.step(self.func, t_current, h_current * self.time_direction,
                                                    y_current, return_variables=True)

            if not self.end_point_mode: # evaluate at some points on the fly if not in end_time_mode
                # if regenerate computation graph, do not save dense states at this step.
                self.update_dense_state(t_current, t_current + h_current * self.time_direction, y_old, y_current)

                while (self.t_end is not None) and torch.abs(t_current + h_current * self.time_direction - self.t0) > torch.abs(
                        self.t_end - self.t0) and torch.abs(t_current - self.t0) <= torch.abs(self.t_end - self.t0):  # if next step is beyond integration time
                    # interpolate and record output
                    all_evaluations.append(
                        self.interpolate(t_current, t_current + h_current * self.time_direction, self.t_end, y_old,
                                         y_current, variables)
                    )
                    self.update_t_end()

            t_current = t_current + h_current * self.time_direction
            steps.append(t_current)
            # update stepsize
            h_current = h_new

            # print current time
            # print(t_current)

        if self.end_point_mode:
            # if t_current < self.t1, make the last move
            if abs(t_current - self.t0) < abs(self.t1 - self.t0):
                step_current = self.t1 - t_current
                y_current, error, variables = self.step(self.func, t_current, step_current,
                                                        y_current, return_variables=True)
                # self.delete_local_computation_graph([_error] + list(_variables))

                t_current = self.t1
                steps.append(t_current)

            all_evaluations = y_current
        else:
            all_evaluations = self.concate_results(all_evaluations)

        if self.tensor_input:
            if not torch.is_tensor(all_evaluations):
                all_evaluations = all_evaluations[0]

        ##################################################################################
        #           If regenerate computation graph using estimated stepsizes            #
        ##################################################################################
        if self.regenerate_graph:

            # reset dense states
            if self.dense_output:
                self.delete_dense_states()
                self.init_dense_states()

            all_evaluations = self.integrate_predefined_grids(y0, t0, predefine_steps=steps, t_eval=t_eval)

        return all_evaluations, steps


class RK12(AdaptiveGridSolver):
    """
    Constants follow wikipedia
    """
    order = 1

    def step(self, func, t, dt, y, return_variables=False):
        k1 = func(t, y)
        k2 = func(t + dt, tuple(_y + 1.0 * dt * _k1 for _y, _k1 in zip(y, k1)))
        out1 = tuple(_y + _k1 * 0.5 * dt + _k2 * 0.5 * dt for _y, _k1, _k2 in zip(y, k1, k2))
        error = tuple(0.5 * dt * (_k1 - _k2) for _k1, _k2 in zip(k1, k2))

        if return_variables:
            return out1, error, [k1, k2]
        else:
            return out1, error

class RK23(AdaptiveGridSolver):
    """
    Constants follow scipy implementation, https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Kutta's_third-order_method
    """
    order = 2

    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2 / 3],
                  [0, 4 / 3, -8 / 9],
                  [0, -1, 1]])
    P = torch.from_numpy(P).float()

    def step(self, func, t, dt, y, return_variables=False):

        k1 = func(t, y)
        k2 = func(t + dt / 2.0, tuple(_y + 1.0 / 2.0 * dt * _k1 for _y, _k1 in zip(y, k1))  )
        k3 = func(t + dt * 0.75, tuple( _y + 0.75 * dt * _k2 for _y, _k2 in zip(y, k2))    )
        k4 = func(t + dt, tuple( _y + 2. / 9. * dt * _k1 + 1. / 3. * dt * _k2 + 4. / 9. * dt * _k3 for
                                 _y, _k1, _k2, _k3 in zip(y, k1, k2, k3))  )
        out1 = tuple( _y + 2. / 9. * dt * _k1 + 1. / 3. * dt * _k2 + 4. / 9. * dt * _k3 for
                      _y, _k1, _k2, _k3 in zip(y, k1, k2, k3))
        error = tuple( 5/72 * dt * _k1 - 1/12 * dt * _k2 -1/9 * dt * _k3 + 1/8 * dt * _k4 for
                       _k1, _k2, _k3, _k4 in zip(k1, k2, k3, k4))

        if return_variables:
            return out1, error, [k1, k2, k3, k4]
        else:
            return out1, error

class Dopri5(AdaptiveGridSolver):
    """
    Constants follow wikipedia, https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Kutta's_third-order_method
    Dormand-Prince's method
    """
    order = 4
    n_stages = 6
    P = np.array([
        [1, -8048581381 / 2820520608, 8663915743 / 2820520608,
         -12715105075 / 11282082432],
        [0, 0, 0, 0],
        [0, 131558114200 / 32700410799, -68118460800 / 10900136933,
         87487479700 / 32700410799],
        [0, -1754552775 / 470086768, 14199869525 / 1410260304,
         -10690763975 / 1880347072],
        [0, 127303824393 / 49829197408, -318862633887 / 49829197408,
         701980252875 / 199316789632],
        [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]])

    P = torch.from_numpy(P).float()

    def step(self, func, t, dt, y, return_variables=False):
        k1 = func(t, y)
        k2 = func(t + dt / 5, tuple( _y + 1 / 5 * dt * _k1 for _y, _k1 in zip(y, k1))   )
        k3 = func(t + dt * 3 / 10,  tuple( _y + 3 / 40 * dt * _k1 + 9.0 / 40.0 * dt * _k2 for
                                           _y, _k1, _k2 in zip(y, k1, k2)) )
        k4 = func(t + dt * 4. / 5., tuple( _y + 44. / 45. * dt * _k1 - 56. / 15. * dt * _k2 + 32. / 9. * dt * _k3 for
                                           _y, _k1, _k2, _k3 in zip(y, k1, k2, k3)))
        k5 = func(t + dt * 8. / 9.,
                       tuple( _y + 19372. / 6561. * dt * _k1 - 25360. / 2187. *dt * _k2 + \
                              64448. / 6561. * dt * _k3 - 212. / 729. * dt * _k4 for
                              _y, _k1, _k2, _k3, _k4 in zip(y, k1, k2, k3, k4) ))

        k6 = func(t + dt,
                       tuple( _y + 9017. / 3168.*dt * _k1 - 355. / 33. * dt * _k2 + 46732. / 5247. * dt * _k3 + \
                              49. / 176. * dt * _k4 - 5103. / 18656. * dt * _k5 for
                        _y, _k1, _k2, _k3, _k4, _k5 in zip(y, k1, k2, k3, k4, k5)) )

        k7 = func(t + dt,
                       tuple( _y + 35. / 384. *dt * _k1 + 0*dt * _k2 + 500. / 1113.*dt * _k3 + \
                              125. / 192.* dt * _k4 - 2187. / 6784. * dt * _k5 + 11. / 84. * dt * _k6 for \
                              _y, _k1, _k2, _k3, _k4, _k5, _k6 in zip(y, k1, k2, k3, k4, k5, k6)) )

        out1 = tuple( _y + 35. / 384. * dt * _k1 + 0 * dt * _k2 + 500. / 1113. *dt * _k3 +
                      125. / 192. * dt * _k4 - 2187. / 6784. * dt * _k5 + 11. / 84. *dt * _k6 for
                      _y, _k1, _k2, _k3, _k4, _k5, _k6 in zip(y, k1, k2, k3, k4, k5, k6))

        error = tuple( (35 / 384 - 5179 / 57600) * dt * _k1 + 0 * dt * _k2 + (500 / 1113 - 7571 / 16695) * dt * _k3 + \
                       (125 / 192 - 393 / 640) * dt * _k4 + (-2187 / 6784 + 92097 / 339200) * dt * _k5 + \
                       (11 / 84 - 187 / 2100) * dt * _k6 - 1 / 40 * dt * _k7
                       for _k1, _k2, _k3, _k4, _k5, _k6, _k7 in zip(k1, k2, k3, k4, k5, k6, k7))

        if return_variables:
            return out1, error, [k1, k2, k3, k4, k5, k6, k7]
        else:
            return out1, error