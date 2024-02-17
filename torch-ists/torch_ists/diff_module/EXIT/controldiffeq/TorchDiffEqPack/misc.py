import torch
import warnings
try:
    pytorch_version_one_and_above = int(torch.__version__[0]) > 0
except TypeError:
    pytorch_version_one_and_above = True

def norm(x):
    """Compute RMS norm."""
    if torch.is_tensor(x):
        return x.norm() / (x.numel()**0.5)
    else:
        return torch.sqrt(sum(x_.norm()**2 for x_ in x) / sum(x_.numel() for x_ in x))

def flatten(iterable):
   out = []
   for i in iterable:
      if hasattr(i,'__iter__') and not isinstance(i, torch.Tensor):
         out.extend(flatten(i))
      else:
         out.append(i)
   return out


def delete_local_computation_graph( inputs):
    for i in inputs:
        #i.set_()
        del i
    #torch.cuda.empty_cache()
    return

def _possibly_nonzero(x):
    return isinstance(x, torch.Tensor) or x != 0

def _scaled_dot_product(scale, xs, ys):
    """Calculate a scaled, vector inner product between lists of Tensors."""
    # Using _possibly_nonzero lets us avoid wasted computation.
    return sum([(scale * x) * y for x, y in zip(xs, ys) if _possibly_nonzero(x) or _possibly_nonzero(y)])

def _convert_to_tensor(a, dtype=None, device=None):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if dtype is not None:
        a = a.type(dtype)
    if device is not None:
        a = a.to(device)
    return a

def _dot_product(xs, ys):
    """Calculate the vector inner product between two lists of Tensors."""
    return sum([x * y for x, y in zip(xs, ys)])

def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.
    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.
    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = tuple(
        _dot_product([-2 * dt, 2 * dt, -8, -8, 16], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    b = tuple(
        _dot_product([5 * dt, -3 * dt, 18, 14, -32], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    c = tuple(
        _dot_product([-4 * dt, dt, -11, -5, 16], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    d = tuple(dt * f0_ for f0_ in f0)
    e = y0
    return [a, b, c, d, e]


def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.
    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: scalar float64 Tensor giving the desired interpolation point.
    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """

    dtype = coefficients[0][0].dtype
    device = coefficients[0][0].device

    t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
    t1 = _convert_to_tensor(t1, dtype=dtype, device=device)
    t = _convert_to_tensor(t, dtype=dtype, device=device)

    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    x = ((t - t0) / (t1 - t0)).type(dtype).to(device)

    xs = [torch.tensor(1).type(dtype).to(device), x]
    for _ in range(2, len(coefficients)):
        xs.append(xs[-1] * x)

    return tuple(_dot_product(coefficients_, reversed(xs)) for coefficients_ in zip(*coefficients))


# ----------------------------------------------------------------------------------------------------
# cubic hermite spline
import matplotlib.pylab as P
import torch as T

def h_poly_helper(tt):
    A = T.tensor([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
    return [
      sum( A[i, j]*tt[j] for j in range(4) )
      for i in range(4) ]

def h_poly(t):
    tt = [ None for _ in range(4) ]
    tt[0] = 1
    for i in range(1, 4):
        tt[i] = tt[i-1]*t
    return h_poly_helper(tt)

def H_poly(t):
    tt = [ None for _ in range(4) ]
    tt[0] = t
    for i in range(1, 4):
        tt[i] = tt[i-1]*t*i/(i+1)
    return h_poly_helper(tt)

def interp_cubic_hermite_spline(x, y, xs):
    """
    :param x: tensor
    :param y: tensor
    :param xs: tensor
    :return:
    """
    if isinstance(xs, T.Tensor):
        xs_np = xs.data.cpu().numpy()
        xs_np = float(xs_np)
    else:
        xs_np = float(xs)
        xs = T.tensor(xs_np).to(y.device)

    x_tmp = (x[1:] - x[:-1])
    if x_tmp == 0:
        return y[0].unsqueeze(0)

    if y.dim() > 1:
        x_tmp = x_tmp.view([-1]+[1]*(y.dim()-1))
    m = (y[1:] - y[:-1])/ x_tmp
    m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])

    I = P.searchsorted(x[1:].data.cpu().numpy(), xs_np)
    if isinstance(I, P.int64):
        I = P.array([I])
    I[I== (x.shape[0]-1)] = I[I== (x.shape[0]-1)] - 2
    dx = (x[I+1]-x[I])
    hh = h_poly((xs.expand_as(x[I])-x[I])/dx)

    if y.dim() > 1:
        hh = [tmp.view([-1]+[1]*(y.dim()-1)) for tmp in hh]
        dx = dx.view([-1]+[1]*(y.dim()-1))
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

def integ(x, y, xs):
    x_tmp = (x[1:] - x[:-1])
    if y.dim() > 1:
        x_tmp = x_tmp.view([-1] + [1] * (y.dim() - 1))
    m = (y[1:] - y[:-1])/ x_tmp
    m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
    I = P.searchsorted(x[1:], xs)
    I[I == (x.shape[0] - 1)] = I[I == (x.shape[0] - 1)] - 2
    Y = T.zeros_like(y)
    Y[1:] = x_tmp*(
      (y[:-1]+y[1:])/2 + (m[:-1] - m[1:])*x_tmp/12
      )
    Y = Y.cumsum(0)
    dx = (x[I+1]-x[I])
    hh = H_poly((xs-x[I])/dx)
    if y.dim() > 1:
        hh = [tmp.view([-1]+[1]*(y.dim()-1)) for tmp in hh]
        dx = dx.view([-1]+[1]*(y.dim()-1))
    return Y[I] + dx*(
      hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
      )

def _is_iterable(inputs):
    try:
        iter(inputs)
        return True
    except TypeError:
        return False

