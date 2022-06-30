from typing import Any, Callable, Optional, Sequence, Tuple, Union
import jax.numpy as jnp
import numpy as np
import dataclasses
import numbers
from jax.tree_util import register_pytree_node_class
import jax 
from jax import lax
import tree_math
import functools
import operator
import scipy.linalg
from helper import inner_prod_with_legendre

class BCType:
  PERIODIC = 'periodic'
  DIRICHLET = 'dirichlet'
  NEUMANN = 'neumann'



@dataclasses.dataclass(init=False, frozen=True)
class BoundaryConditions:
  types: Tuple[Tuple[str, str], ...]

  def shift(self, u, offset, axis):
    raise NotImplementedError(
        'shift() not implemented in BoundaryConditions base class.')

  def values(self, axis, grid, offset, time):
    raise NotImplementedError(
        'values() not implemented in BoundaryConditions base class.')

#Array = Union[np.ndarray, jnp.DeviceArray]
Array = jnp.ndarray

@dataclasses.dataclass(init=False, frozen=True)
class ConstantBoundaryConditions(BoundaryConditions):
  types: Tuple[Tuple[str, str], ...]
  _values: Tuple[Tuple[Optional[float], Optional[float]], ...]

  def __init__(self, types, values):
    types = tuple(types)
    values = tuple(values)
    object.__setattr__(self, 'types', types)
    object.__setattr__(self, '_values', values)

  def shift(self, u, offset, axis):
    padded = self._pad(u, offset, axis)
    trimmed = self._trim(padded, -offset, axis)
    return trimmed

  def _pad(self, u, width, axis):
    if width < 0:  # pad lower boundary
      bc_type = self.types[axis][0]
      padding = (-width, 0)
    else:  # pad upper boundary
      bc_type = self.types[axis][1]
      padding = (0, width)

    full_padding = [(0, 0)] * u.grid.ndim
    full_padding[axis] = padding

    offset = list(u.offset)
    offset[axis] -= padding[0]

    if bc_type != BCType.PERIODIC and abs(width) > 1:
      raise ValueError(
          'Padding past 1 ghost cell is not defined in nonperiodic case.')

    if bc_type == BCType.PERIODIC:
      pad_kwargs = dict(mode='wrap')
    elif bc_type == BCType.DIRICHLET:
      if np.isclose(u.offset[axis] % 1, 0.5):  # cell center
        data = (2 * jnp.pad(
            u.data, full_padding, mode='constant', constant_values=self._values)
                - jnp.pad(u.data, full_padding, mode='symmetric'))
        return GridArray(data, tuple(offset), u.grid)
      elif np.isclose(u.offset[axis] % 1, 0):  # cell edge
        pad_kwargs = dict(mode='constant', constant_values=self._values)
      else:
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
    elif bc_type == BCType.NEUMANN:
      if not (np.isclose(u.offset[axis] % 1, 0) or
              np.isclose(u.offset[axis] % 1, 0.5)):
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
      else:
        data = (
            jnp.pad(u.data, full_padding, mode='edge') + u.grid.step[axis] *
            (jnp.pad(u.data, full_padding, mode='constant') - jnp.pad(
                u.data,
                full_padding,
                mode='constant',
                constant_values=self._values)))
        return GridArray(data, tuple(offset), u.grid)

    else:
      raise ValueError('invalid boundary type')

    data = jnp.pad(u.data, full_padding, **pad_kwargs)
    return GridArray(data, tuple(offset), u.grid)

  def _trim(self, u, width, axis):
    if width < 0:  # trim lower boundary
      padding = (-width, 0)
    else:  # trim upper boundary
      padding = (0, width)

    limit_index = u.data.shape[axis] - padding[1]
    data = lax.slice_in_dim(u.data, padding[0], limit_index, axis=axis)
    offset = list(u.offset)
    offset[axis] += padding[0]
    return GridArray(data, tuple(offset), u.grid)

  def values(self, axis, grid):
    if None in self._values[axis]:
      return (None, None)
    bc = tuple(
        jnp.full(grid.shape[:axis] +
                 grid.shape[axis + 1:], self._values[axis][-i]) for i in [0, 1])
    return bc

  trim = _trim
  pad = _pad


class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
  def __init__(self, types):
    ndim = len(types)
    values = ((0.0, 0.0),) * ndim
    super(HomogeneousBoundaryConditions, self).__init__(types, values)


def periodic_boundary_conditions(ndim):
  return HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),) * ndim)

class InconsistentOffsetError(Exception):
  """Raised for cases of inconsistent offset in GridArrays."""

class InconsistentOffsetError(Exception):
  """Raised for cases of inconsistent offset in GridArrays."""

def _normalize_axis(axis, ndim):
  if not -ndim <= axis < ndim:
    raise ValueError(f'invalid axis {axis} for ndim {ndim}')
  if axis < 0:
    axis += ndim
  return axis

def slice_along_axis(inputs, axis, idx, expect_same_dims):
  arrays, tree_def = jax.tree_flatten(inputs)
  ndims = set(a.ndim for a in arrays)
  if expect_same_dims and len(ndims) != 1:
    raise ValueError('arrays in `inputs` expected to have same ndims, but have '
                     f'{ndims}. To allow this, pass expect_same_dims=False')
  sliced = []
  for array in arrays:
    ndim = array.ndim
    slc = tuple(idx if j == _normalize_axis(axis, ndim) else slice(None)
                for j in range(ndim))
    sliced.append(array[slc])
  return jax.tree_unflatten(tree_def, sliced)

def split_along_axis(inputs, split_idx, axis, expect_same_dims):
  first_slice = slice_along_axis(
      inputs, axis, slice(0, split_idx), expect_same_dims)
  second_slice = slice_along_axis(
      inputs, axis, slice(split_idx, None), expect_same_dims)
  return first_slice, second_slice

def consistent_grid(*arrays):
  grids = {array.grid for array in arrays}
  if len(grids) != 1:
    raise InconsistentGridError(f'arrays do not have a unique grid: {grids}')
  grid, = grids
  return grid

def consistent_offset(*arrays):
  offsets = {array.offset for array in arrays}
  if len(offsets) != 1:
    raise InconsistentOffsetError(
        f'arrays do not have a unique offset: {offsets}')
  offset, = offsets
  return offset


@dataclasses.dataclass(init=False, frozen=True)
class Grid:
  shape: Tuple[int, ...]
  step: Tuple[float, ...]
  domain: Tuple[Tuple[float, float], ...]

  def __init__(
      self,
      shape: Sequence[int],
      step: Optional[Union[float, Sequence[float]]] = None,
      domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
  ):
    shape = tuple(operator.index(s) for s in shape)
    object.__setattr__(self, 'shape', shape)

    if step is not None and domain is not None:
      raise TypeError('cannot provide both step and domain')
    elif domain is not None:
      if isinstance(domain, (int, float)):
        domain = ((0, domain),) * len(shape)
      else:
        if len(domain) != self.ndim:
          raise ValueError('length of domain does not match ndim: '
                           f'{len(domain)} != {self.ndim}')
        for bounds in domain:
          if len(bounds) != 2:
            raise ValueError(
                f'domain is not sequence of pairs of numbers: {domain}')
      domain = tuple((float(lower), float(upper)) for lower, upper in domain)
    else:
      if step is None:
        step = 1
      if isinstance(step, numbers.Number):
        step = (step,) * self.ndim
      elif len(step) != self.ndim:
        raise ValueError('length of step does not match ndim: '
                         f'{len(step)} != {self.ndim}')
      domain = tuple(
          (0.0, float(step_ * size)) for step_, size in zip(step, shape))

    object.__setattr__(self, 'domain', domain)

    step = tuple(
        (upper - lower) / size for (lower, upper), size in zip(domain, shape))
    object.__setattr__(self, 'step', step)

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def cell_center(self):
    return self.ndim * (0.5,)

  @property
  def cell_faces(self):
    d = self.ndim
    offsets = (np.eye(d) + np.ones([d, d])) / 2.
    return tuple(tuple(float(o) for o in offset) for offset in offsets)

  def stagger(self, v):
    offsets = self.cell_faces
    return tuple(GridArray(u, o, self) for u, o in zip(v, offsets))

  def center(self, v):
    offset = self.cell_center
    return jax.tree_map(lambda u: GridArray(u, offset, self), v)

  def axes(self, offset):
    if offset is None:
      offset = self.cell_center
    if len(offset) != self.ndim:
      raise ValueError(f'unexpected offset length: {len(offset)} vs '
                       f'{self.ndim}')
    return tuple(lower + (jnp.arange(length) + offset_i) * step
                 for (lower, _), offset_i, length, step in zip(
                     self.domain, offset, self.shape, self.step))

  def fft_axes(self):
    freq_axes = tuple(
        jnp.fft.fftfreq(n, d=s) for (n, s) in zip(self.shape, self.step))
    return freq_axes

  def rfft_axes(self):
    fft_axes = tuple(
        jnp.fft.fftfreq(n, d=s)
        for (n, s) in zip(self.shape[:-1], self.step[:-1]))
    rfft_axis = (jnp.fft.rfftfreq(self.shape[-1], d=self.step[-1]),)
    return fft_axes + rfft_axis

  def mesh(self, offset):
    axes = self.axes(offset)
    return tuple(jnp.meshgrid(*axes, indexing='ij'))

  def rfft_mesh(self):
    rfft_axes = self.rfft_axes()
    return tuple(jnp.meshgrid(*rfft_axes, indexing='ij'))

  def eval_on_mesh(self, fn, offset):
    if offset is None:
      offset = self.cell_center
    return GridArray(fn(*self.mesh(offset)), offset, self)


@register_pytree_node_class
@dataclasses.dataclass
class GridArray(np.lib.mixins.NDArrayOperatorsMixin):
  data: Array
  offset: Tuple[float, ...]
  grid: Grid

  def tree_flatten(self):
    children = (self.data,)
    aux_data = (self.offset, self.grid)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, *aux_data)

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def shape(self):
    return self.data.shape

  _HANDLED_TYPES = (numbers.Number, np.ndarray, jnp.DeviceArray,
                    jax.ShapedArray, jax.core.Tracer)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    for x in inputs:
      if not isinstance(x, self._HANDLED_TYPES + (GridArray,)):
        return NotImplemented
    if method != '__call__':
      return NotImplemented
    try:
      func = getattr(jnp, ufunc.__name__)
    except AttributeError:
      return NotImplemented
    arrays = [x.data if isinstance(x, GridArray) else x for x in inputs]
    result = func(*arrays)
    offset = consistent_offset(*[x for x in inputs if isinstance(x, GridArray)])
    grid = consistent_grid(*[x for x in inputs if isinstance(x, GridArray)])
    if isinstance(result, tuple):
      return tuple(GridArray(r, offset, grid) for r in result)
    else:
      return GridArray(result, offset, grid)


@register_pytree_node_class
@dataclasses.dataclass
class GridVariable:
  array: GridArray
  bc: BoundaryConditions

  def __post_init__(self):
    if not isinstance(self.array, GridArray):  # frequently missed by pytype
      raise ValueError(
          f'Expected array type to be GridArray, got {type(self.array)}')
    if len(self.bc.types) != self.grid.ndim:
      raise ValueError(
          'Incompatible dimension between grid and bc, grid dimension = '
          f'{self.grid.ndim}, bc dimension = {len(self.bc.types)}')

  def tree_flatten(self):
    children = (self.array,)
    aux_data = (self.bc,)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, *aux_data)

  @property
  def dtype(self):
    return self.array.dtype

  @property
  def shape(self):
    return self.array.shape

  @property
  def data(self):
    return self.array.data

  @property
  def offset(self):
    return self.array.offset

  @property
  def grid(self):
    return self.array.grid

  def shift(self, offset, axis):
    return self.bc.shift(self.array, offset, axis)

  def _interior_grid(self):
    grid = self.array.grid
    domain = list(grid.domain)
    shape = list(grid.shape)
    for axis in range(self.grid.ndim):
      # nothing happens in periodic case
      if self.bc.types[axis][1] == 'periodic':
        continue
      # nothing happens if the offset is not 0.0 or 1.0
      # this will automatically set the grid to interior.
      if np.isclose(self.array.offset[axis], 1.0):
        shape[axis] -= 1
        domain[axis] = (domain[axis][0], domain[axis][1] - grid.step[axis])
      elif np.isclose(self.array.offset[axis], 0.0):
        shape[axis] -= 1
        domain[axis] = (domain[axis][0] + grid.step[axis], domain[axis][1])
    return Grid(shape, domain=tuple(domain))

  def _interior_array(self) -> Array:
    data = self.array.data
    for axis in range(self.grid.ndim):
      # nothing happens in periodic case
      if self.bc.types[axis][1] == 'periodic':
        continue
      # nothing happens if the offset is not 0.0 or 1.0
      if np.isclose(self.offset[axis], 1.0):
        data, _ = split_along_axis(data, -1, axis)
      elif np.isclose(self.offset[axis], 0.0):
        _, data = split_along_axis(data, 1, axis)

    return data

  def interior(self) -> GridArray:
    interior_array = self._interior_array()
    interior_grid = self._interior_grid()
    return GridArray(interior_array, self.array.offset, interior_grid)

  def enforce_edge_bc(self, *args):
    if self.grid.shape != self.array.data.shape:
      raise ValueError('Stored array and grid have mismatched sizes.')
    data = jnp.array(self.array.data)
    for axis in range(self.grid.ndim):
      if 'periodic' not in self.bc.types[axis]:
        values = self.bc.values(axis, self.grid, *args)
        for boundary_side in range(2):
          if np.isclose(self.array.offset[axis], boundary_side):
            # boundary data is set to match self.bc:
            all_slice = [
                slice(None, None, None),
            ] * self.grid.ndim
            all_slice[axis] = -boundary_side
            data = data.at[tuple(all_slice)].set(values[boundary_side])
    return GridVariable(
        array=GridArray(data, self.array.offset, self.grid),
        bc=self.bc)

class InconsistentBoundaryConditionsError(Exception):
  """Raised for cases of inconsistent bc between GridVariables."""

def consistent_boundary_conditions(*arrays):
  bcs = {array.bc for array in arrays}
  if len(bcs) != 1:
    raise InconsistentBoundaryConditionsError(
        f'arrays do not have a unique bc: {bcs}')
  bc, = bcs
  return bc



def control_volume_offsets(c):
  return tuple(
      tuple(o + .5 if i == j else o
            for i, o in enumerate(c.offset))
      for j in range(len(c.offset)))

def applied(func):
  def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
    for arg in args + tuple(kwargs.values()):
      if isinstance(arg, GridVariable):
        raise ValueError('applied() cannot be used with GridVariable')

    offset = consistent_offset(*[
        arg for arg in args + tuple(kwargs.values())
        if isinstance(arg, GridArray)
    ])
    grid = consistent_grid(*[
        arg for arg in args + tuple(kwargs.values())
        if isinstance(arg, GridArray)
    ])
    raw_args = [arg.data if isinstance(arg, GridArray) else arg for arg in args]
    raw_kwargs = {
        k: v.data if isinstance(v, GridArray) else v for k, v in kwargs.items()
    }
    data = func(*raw_args, **raw_kwargs)
    return GridArray(data, offset, grid)

  return wrapper


grids_where = applied(jnp.where)


def safe_div(x, y, default_numerator=1):
  return x / jnp.where(y != 0, y, default_numerator)

def _linear_along_axis(c, offset, axis):
  offset_delta = offset - c.offset[axis]
  if offset_delta == 0:
    return c
  new_offset = tuple(offset if j == axis else o
                     for j, o in enumerate(c.offset))
  if int(offset_delta) == offset_delta:
    return GridVariable(
        array=GridArray(data=c.shift(int(offset_delta), axis).data,
                              offset=new_offset,
                              grid=c.grid),
        bc=c.bc)
  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  floor_weight = ceil - offset_delta
  ceil_weight = 1. - floor_weight
  data = (floor_weight * c.shift(floor, axis).data +
          ceil_weight * c.shift(ceil, axis).data)
  return GridVariable(
      array=GridArray(data, new_offset, c.grid), bc=c.bc)

def linear(c, offset, v, dt):
  del v, dt  # unused
  if len(offset) != len(c.offset):
    raise ValueError('`c.offset` and `offset` must have the same length;'
                     f'got {c.offset} and {offset}.')
  interpolated = c
  for a, o in enumerate(offset):
    interpolated = _linear_along_axis(interpolated, offset=o, axis=a)
  return interpolated

def upwind(c, offset, v, dt):
  del dt  # unused
  if c.offset == offset: return c
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if current != target
  )
  if len(interpolation_axes) != 1:
    raise InconsistentOffsetError(
        f'for upwind interpolation `c.offset` and `offset` must differ at most '
        f'in one entry, but got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  u = v[axis]
  offset_delta = u.offset[axis] - c.offset[axis]

  # If offsets differ by an integer, we can just shift `c`.
  if int(offset_delta) == offset_delta:
    return GridVariable(
        array=GridArray(data=c.shift(int(offset_delta), axis).data,
                              offset=offset,
                              grid=consistent_grid(c, u)),
        bc=c.bc)

  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  array = applied(jnp.where)(
      u.array > 0, c.shift(floor, axis).data, c.shift(ceil, axis).data
  )
  grid = consistent_grid(c, u)
  return GridVariable(
      array=GridArray(array.data, offset, grid),
      bc=periodic_boundary_conditions(grid.ndim))


def apply_tvd_limiter(interpolation_fn, limiter):
  def tvd_interpolation(c, offset, v, dt):
    for axis, axis_offset in enumerate(offset):
      interpolation_offset = tuple([
          c_offset if i != axis else axis_offset
          for i, c_offset in enumerate(c.offset)
      ])
      if interpolation_offset != c.offset:
        if interpolation_offset[axis] - c.offset[axis] != 0.5:
          raise NotImplementedError('tvd_interpolation only supports forward '
                                    'interpolation to control volume faces.')
        c_low = upwind(c, offset, v, dt)
        c_high = interpolation_fn(c, offset, v, dt)

        c_left = c.shift(-1, axis)
        c_right = c.shift(1, axis)
        c_next_right = c.shift(2, axis)
        positive_u_r = safe_div(c.data - c_left.data, c_right.data - c.data)
        negative_u_r = safe_div(c_next_right.data - c_right.data,
                                c_right.data - c.data)
        positive_u_phi = GridArray(
            limiter(positive_u_r), c_low.offset, c.grid)
        negative_u_phi = GridArray(
            limiter(negative_u_r), c_low.offset, c.grid)
        u = v[axis]
        phi = applied(jnp.where)(
            u.array > 0, positive_u_phi, negative_u_phi)
        c_interpolated = c_low.array - (c_low.array - c_high.array) * phi
        c = GridVariable(
            GridArray(c_interpolated.data, interpolation_offset, c.grid),
            c.bc)
    return c

  return tvd_interpolation

def lax_wendroff(c, offset, v, dt):
  if c.offset == offset: 
    return c
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if current != target
  )
  if len(interpolation_axes) != 1:
    raise InconsistentOffsetError(
        f'for Lax-Wendroff interpolation `c.offset` and `offset` must differ at'
        f' most in one entry, but got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  u = v[axis]
  offset_delta = u.offset[axis] - c.offset[axis]
  floor = int(np.floor(offset_delta))  # used for positive velocity
  ceil = int(np.ceil(offset_delta))  # used for negative velocity
  grid = consistent_grid(c, u)
  courant_numbers = (dt / grid.step[axis]) * u.data
  positive_u_case = (
      c.shift(floor, axis).data + 0.5 * (1 - courant_numbers) *
      (c.shift(ceil, axis).data - c.shift(floor, axis).data))
  negative_u_case = (
      c.shift(ceil, axis).data - 0.5 * (1 + courant_numbers) *
      (c.shift(ceil, axis).data - c.shift(floor, axis).data))
  array = grids_where(u.array > 0, positive_u_case, negative_u_case)
  grid = consistent_grid(c, u)
  return GridVariable(
      array=GridArray(array.data, offset, grid),
      bc=periodic_boundary_conditions(grid.ndim))

def van_leer_limiter(r):
  return jnp.where(r > 0, safe_div(2 * r, 1 + r), 0.0)

def averaged_offset(*arrays):
  offset = np.mean([array.offset for array in arrays], axis=0)
  return tuple(offset.tolist())

def stencil_sum(*arrays):
  offset = averaged_offset(*arrays)
  result = sum(array.data for array in arrays)  # type: ignore
  grid = consistent_grid(*arrays)
  return GridArray(result, offset, grid)

def backward_difference(u, axis=None):
  if axis is None:
    axis = range(u.grid.ndim)
  if not isinstance(axis, int):
    return tuple(backward_difference(u, a) for a in axis)
  diff = stencil_sum(u.array, -u.shift(-1, axis))
  return diff / u.grid.step[axis]

def divergence(v):
  grid = consistent_grid(*v)
  if len(v) != grid.ndim:
    raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                     f'Expected length {grid.ndim}; got {len(v)}.')
  differences = [backward_difference(u, axis) for axis, u in enumerate(v)]
  return sum(differences)

def _advect_aligned(cs, v):
  if len(cs) != len(v):
    raise ValueError('`cs` and `v` must have the same length;'
                     f'got {len(cs)} vs. {len(v)}.')
  flux = tuple(c.array * u.array for c, u in zip(cs, v))
  flux = tuple(GridVariable(f, c.bc) for f, c in zip(flux, cs))
  return -divergence(flux)

def advect_general(c, v, u_interpolation_fn, c_interpolation_fn, dt):
  target_offsets = control_volume_offsets(c)
  aligned_v = tuple(u_interpolation_fn(u, target_offset, v, dt)
                    for u, target_offset in zip(v, target_offsets))
  aligned_c = tuple(c_interpolation_fn(c, target_offset, aligned_v, dt)
                    for target_offset in target_offsets)
  return _advect_aligned(aligned_c, aligned_v)

def advect_van_leer_using_limiters(c, v, dt):
  c_interpolation_fn = apply_tvd_limiter(
      lax_wendroff, limiter=van_leer_limiter)
  return advect_general(c, v, linear, c_interpolation_fn, dt)

def _wrap_term_as_vector(fun, *, name):
  return tree_math.unwrap(jax.named_call(fun, name=name), vector_argnums=0)

def laplacian(u):
  scales = np.square(1 / np.array(u.grid.step, dtype=u.dtype))
  result = -2 * u.array * np.sum(scales)
  for axis in range(u.grid.ndim):
    result += stencil_sum(u.shift(-1, axis), u.shift(+1, axis)) * scales[axis]
  return result

def forward_difference(u, axis=None):
  if axis is None:
    axis = range(u.grid.ndim)
  if not isinstance(axis, int):
    return tuple(forward_difference(u, a) for a in axis)
  diff = stencil_sum(u.shift(+1, axis), -u.array)
  return diff / u.grid.step[axis]

def has_all_periodic_boundary_conditions(*arrays):
  for array in arrays:
    for lower_bc_type, upper_bc_type in array.bc.types:
      if lower_bc_type != BCType.PERIODIC or upper_bc_type != BCType.PERIODIC:
        return False
  return True

def get_pressure_bc_from_velocity(v):
  velocity_bc_types = consistent_boundary_conditions(*v).types
  pressure_bc_types = []
  for velocity_bc_lower, velocity_bc_upper in velocity_bc_types:
    if velocity_bc_lower == BCType.PERIODIC:
      pressure_bc_lower = BCType.PERIODIC
    elif velocity_bc_lower == BCType.DIRICHLET:
      pressure_bc_lower = BCType.NEUMANN
    else:
      raise ValueError('Expected periodic or dirichlete velocity BC, '
                       f'got {velocity_bc_lower}')
    if velocity_bc_upper == BCType.PERIODIC:
      pressure_bc_upper = BCType.PERIODIC
    elif velocity_bc_upper == BCType.DIRICHLET:
      pressure_bc_upper = BCType.NEUMANN
    else:
      raise ValueError('Expected periodic or dirichlete velocity BC, '
                       f'got {velocity_bc_upper}')
    pressure_bc_types.append((pressure_bc_lower, pressure_bc_upper))
  return HomogeneousBoundaryConditions(pressure_bc_types)

def diffuse(c, nu):
  return nu * laplacian(c)

def projection(v, solve):
  grid = consistent_grid(*v)
  pressure_bc = get_pressure_bc_from_velocity(v)

  q0 = GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = GridVariable(q0, pressure_bc)

  q = solve(v, q0)
  q = GridVariable(q, pressure_bc)
  q_grad = forward_difference(q)
  if has_all_periodic_boundary_conditions(*v):
    v_projected = tuple(GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
  else:
    v_projected = tuple(GridVariable(u.array - q_g, u.bc).enforce_edge_bc() for u, q_g in zip(v, q_grad))
  return v_projected


def laplacian_matrix(size, step):
  column = np.zeros(size)
  column[0] = -2 / step**2
  column[1] = column[-1] = 1 / step**2
  return scipy.linalg.circulant(column)


def _hermitian_matmul_transform(func, operators, dtype, precision):
  eigenvalues, eigenvectors = zip(*map(np.linalg.eigh, operators))

  summed_eigenvalues = functools.reduce(np.add.outer, eigenvalues)
  diagonals = jnp.asarray(func(summed_eigenvalues), dtype)
  eigenvectors = [jnp.asarray(vector, dtype) for vector in eigenvectors]

  shape = summed_eigenvalues.shape
  if diagonals.shape != shape:
    raise ValueError('output shape from func() does not match input shape: '
                     f'{diagonals.shape} vs {shape}')

  def apply(rhs: Array) -> Array:
    if rhs.shape != shape:
      raise ValueError(f'rhs.shape={rhs.shape} does not match shape={shape}')
    if rhs.dtype != dtype:
      raise ValueError(f'rhs.dtype={rhs.dtype} does not match dtype={dtype}')

    out = rhs
    for vectors in eigenvectors:
      out = jnp.tensordot(out, vectors, axes=(0, 0), precision=precision)
    out *= diagonals
    for vectors in eigenvectors:
      out = jnp.tensordot(out, vectors, axes=(0, 1), precision=precision)
    return out

  return apply

def _circulant_fft_transform(func, operators, dtype):
  eigenvalues = [np.fft.fft(op[:, 0]) for op in operators]
  summed_eigenvalues = functools.reduce(np.add.outer, eigenvalues)
  diagonals = jnp.asarray(func(summed_eigenvalues))

  shape = tuple(op.shape[0] for op in operators)
  if diagonals.shape != shape:
    raise ValueError('output shape from func() does not match input shape: '
                     f'{diagonals.shape} vs {shape}')

  def apply(rhs: Array) -> Array:
    if rhs.shape != shape:
      raise ValueError(f'rhs.shape={rhs.shape} does not match shape={shape}')
    return jnp.fft.ifftn(diagonals * jnp.fft.fftn(rhs)).astype(dtype)

  return apply


def _circulant_rfft_transform(func, operators, dtype):
  if operators[-1].shape[0] % 2:
    raise ValueError('implementation="rfft" currently requires an even size '
                     'for the last axis')
  eigenvalues = ([np.fft.fft(op[:, 0]) for op in operators[:-1]]
                 + [np.fft.rfft(operators[-1][:, 0])])
  summed_eigenvalues = functools.reduce(np.add.outer, eigenvalues)
  diagonals = jnp.asarray(func(summed_eigenvalues))

  if diagonals.shape != summed_eigenvalues.shape:
    raise ValueError('output shape from func() does not match input shape: '
                     f'{diagonals.shape} vs {summed_eigenvalues.shape}')

  def apply(rhs: Array) -> Array:
    if rhs.dtype != dtype:
      raise ValueError(f'rhs.dtype={rhs.dtype} does not match dtype={dtype}')
    return jnp.fft.irfftn(diagonals * jnp.fft.rfftn(rhs)).astype(dtype)

  return apply


def transform(func, operators, dtype, *, hermitian, circulant, precision, implementation=None):
  if any(op.ndim != 2 or op.shape[0] != op.shape[1] for op in operators):
    raise ValueError('operators are not all square matrices. Shapes are '
                     + ', '.join(str(op.shape) for op in operators))

  if implementation is None:
    if all(device.platform == 'tpu' for device in jax.local_devices()):
      size = max(op.shape[0] for op in operators)
      implementation = 'rfft' if size > 1024 else 'matmul'
    else:
      implementation = 'rfft'
    if implementation == 'rfft' and operators[-1].shape[0] % 2:
      implementation = 'matmul'

  if implementation == 'matmul':
    if not hermitian:
      raise ValueError('non-hermitian operators not yet supported with '
                       'implementation="matmul"')
    return _hermitian_matmul_transform(func, operators, dtype, precision)
  elif implementation == 'fft':
    if not circulant:
      raise ValueError('non-circulant operators not yet supported with '
                       'implementation="fft"')
    return _circulant_fft_transform(func, operators, dtype)
  elif implementation == 'rfft':
    if not circulant:
      raise ValueError('non-circulant operators not yet supported with '
                       'implementation="rfft"')
    return _circulant_rfft_transform(func, operators, dtype)
  else:
    raise ValueError(f'invalid implementation: {implementation}')

def psuedoinverse(operators, dtype, *, hermitian, circulant, implementation=None, precision=None, cutoff=None):
  if cutoff is None:
    cutoff = 10 * jnp.finfo(dtype).eps

  def func(v):
    with np.errstate(divide='ignore', invalid='ignore'):
      return np.where(abs(v) > cutoff, 1 / v, 0)

  return transform(func, operators, dtype, hermitian=hermitian,
                   circulant=circulant, implementation=implementation,
                   precision=precision)

#### PRESSURE SOLVES

def solve_fast_diag(v, q0):
  del q0  # unused
  if not has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic velocity BC')
  grid = consistent_grid(*v)
  rhs = divergence(v)
  laplacians = list(map(laplacian_matrix, grid.shape, grid.step))
  pinv = psuedoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=True)
  return applied(pinv)(rhs)


class ButcherTableau:
  a: Sequence[Sequence[float]]
  b: Sequence[float]

  def __init__(self, a, b):
    self.a = a
    self.b = b

  def __post_init__(self):
    if len(self.a) + 1 != len(self.b):
      raise ValueError("inconsistent Butcher tableau")



def navier_stokes_rk(tableau, equation, time_step):
  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  P = tree_math.unwrap(equation.pressure_projection)

  a = tableau.a
  b = tableau.b
  num_steps = len(b)

  @tree_math.wrap
  def step_fn(u0):
    u = [None] * num_steps
    k = [None] * num_steps

    u[0] = u0
    k[0] = F(u0)

    for i in range(1, num_steps):
      u_star = u0 + dt * sum(a[i-1][j] * k[j] for j in range(i) if a[i-1][j])
      u[i] = P(u_star)
      k[i] = F(u[i])

    u_star = u0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])

    u_final = P(u_star)

    return u_final

  return step_fn

class ExplicitNavierStokesODE:

  def __init__(self, explicit_terms, pressure_projection):
    self.explicit_terms = explicit_terms
    self.pressure_projection = pressure_projection

  def explicit_terms(self, state):
    raise NotImplementedError

  def pressure_projection(self, state):
    raise NotImplementedError

def forward_euler(equation, time_step):
  return jax.named_call(
      navier_stokes_rk(
          ButcherTableau(a=[], b=[1]),
          equation,
          time_step),
      name="forward_euler",
  )

####### EQUATION.PY ########


def navier_stokes_explicit_terms(viscosity, dt, forcing):

  def convect(v):
    return tuple(advect_van_leer_using_limiters(u, v, dt) for u in v)

  def diffuse_velocity(v, *args):
    return tuple(diffuse(u, *args) for u in v)

  convection = _wrap_term_as_vector(convect, name='convection')
  diffusion = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  @tree_math.wrap
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    dv_dt = convection(v)
    if viscosity > 0.0:
      print(viscosity)
      print(diffusion(v, viscosity))
      dv_dt += diffusion(v, viscosity)
    if forcing is not None:
      print(forcing(v))
      dv_dt += forcing(v)
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    dv_dt = _explicit_terms(v)
    return tuple(GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs

def semi_implicit_navier_stokes(viscosity, dt, pressure_solve, forcing):
  explicit_terms = navier_stokes_explicit_terms(viscosity, dt, forcing)

  pressure_projection = jax.named_call(projection, name='pressure')

  ode = ExplicitNavierStokesODE(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve)
  )
  step_fn = forward_euler(ode, dt)
  return step_fn

###### IMPORT THESE FUNCTIONS

def get_forcing(args, nx, ny):
  ff = lambda x, y, t: jnp.sin(4 * (2 * np.pi / args.Ly) * y)
  shape = (nx, ny)
  domain = ((0, args.Lx), (0, args.Ly))
  grid = Grid(shape, domain=domain)
  x_term = inner_prod_with_legendre(nx, ny, args.Lx, args.Ly, 0, ff, 0.0, n = 1)[:,:,0]
  x_term = args.forcing_coefficient * GridArray(x_term, (1, 1/2), grid)
  y_term = GridArray(jnp.zeros_like(x_term.data), (1/2, 1), grid)
  constant_term = (x_term, y_term)
  dx = args.Lx / nx
  dy = args.Ly / ny
  C = args.forcing_coefficient * dx * dy * args.damping_coefficient
  def f_forcing(v):
    return tuple( c_i - C * v_i.array for c_i, v_i in zip(constant_term, v))

  return f_forcing

def vorticity(v):
  # calculated at offset = (1, 1)
  u_x, u_y = v
  dx, dy = u_x.array.grid.step
  du_y_dx = (jnp.roll(u_y.array.data, -1, axis=0) - u_y.array.data) / dx
  du_x_dy = (jnp.roll(u_x.array.data, -1, axis=1) - u_x.array.data) / dy
  return (du_y_dx - du_x_dy)


def get_step_func(nu, dt, pressure_solve = solve_fast_diag, forcing=None):
  return semi_implicit_navier_stokes(nu, dt, pressure_solve, forcing)

@functools.partial(jax.jit, static_argnums=(1, 2))
def simulate_baseline(v, step_func, nt):
  def _scanf(v, x):
    return step_func(v), None
  vf, _ = jax.lax.scan(_scanf, v, None, length=nt)
  return vf

def get_velocity(args, u_x, u_y):
  assert u_x.shape == u_y.shape
  shape = u_x.shape
  nx, ny = shape
  dx = args.Lx / nx
  dy = args.Ly / ny
  step = (dx, dy)
  domain = ((0, args.Lx), (0, args.Ly))

  ndims = 2
  bcs = periodic_boundary_conditions(ndims)
  
  grid = Grid(shape, domain=domain)
  u_x = GridVariable(GridArray(u_x, grid.cell_faces[0], grid=grid), bcs)
  u_y = GridVariable(GridArray(u_y, grid.cell_faces[1], grid=grid), bcs)
  return (u_x, u_y)