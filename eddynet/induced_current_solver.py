"""Solve conductivity-contrast currents using a bulk-conductor approximation.

The interaction kernel describes an infinite, homogeneous conductor. It omits
surface reflections, specimen layers, and feedback from the ferrite core, so it
does not reproduce the paper's complete forward model. Cell integrals use Gauss
quadrature, a weighted sampling rule. Each cell's field at its own center also
includes an analytical static correction for its rectangular shape.
Fields follow the exp(+i*omega*t) convention used by incident_field_solver.

The bulk kernel follows the unbounded-domain part of Bowler et al. (1991):
https://class.ece.iastate.edu/jbowler/My%20Articles/JAP_Vol_Flaw.pdf
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from eddynet.incident_field_solver import (
    VACUUM_PERMEABILITY_H_PER_M,
    MockSpecimen,
)

# Illustrative cell dimensions along x, y, z; replace with physical dimensions.
DEFAULT_CRACK_CELL_SIZE_M = (0.1e-3, 0.1e-3, 0.1e-3)


def solve_induced_current(
    crack_points,
    e_inc,
    frequency_hz,
    *,
    cell_size_m=DEFAULT_CRACK_CELL_SIZE_M,
    specimen=None,
    crack_conductivity_s_per_m=0.0,
    field_operator=None,
):
    """Return equivalent current densities (n, 3) and volume weights (n,).

    crack_points contains XYZ cell centers in meters. e_inc contains complex
    electric fields in V/m at those same centers. Each cell has the uniform
    dimensions cell_size_m, ordered x, y, z. Returned currents are in A/m^2
    and weights are in m^3. Empty cracks return empty arrays.

    These are equivalent sources representing the conductivity change, not
    conduction currents inside the air gap. The solve is
    (I - delta_sigma * K) * j_ind = delta_sigma * e_inc.
    Only conductivity contrast is modeled; permeability contrast is omitted.

    field_operator may supply a precomputed (3*n, 3*n) matrix K for this exact
    geometry, material, and frequency. It maps current density to electric
    field and already includes source-cell volumes. Reuse it as the coil moves.
    """
    points, cell_size = _validate_cell_geometry(crack_points, cell_size_m)
    specimen = MockSpecimen() if specimen is None else specimen
    conductivity, _ = _material_properties(frequency_hz, specimen)
    e_inc = np.asarray(e_inc, dtype=complex)
    if e_inc.shape != points.shape or not np.all(np.isfinite(e_inc)):
        raise ValueError("e_inc must contain one finite XYZ field per crack point")
    if not np.isfinite(crack_conductivity_s_per_m) or crack_conductivity_s_per_m < 0:
        raise ValueError("crack conductivity must be finite and nonnegative")

    n_components = 3 * len(points)
    if field_operator is not None:
        field_operator = np.asarray(field_operator, dtype=complex)
        if field_operator.shape != (n_components, n_components) or not np.all(
            np.isfinite(field_operator)
        ):
            raise ValueError("field_operator must be a finite (3*n, 3*n) matrix")

    volumes = np.full(len(points), np.prod(cell_size))
    delta_sigma = crack_conductivity_s_per_m - conductivity
    if len(points) == 0 or delta_sigma == 0:
        return np.zeros_like(e_inc), volumes
    if field_operator is None:
        field_operator = build_electric_field_operator(
            points, frequency_hz, cell_size_m=cell_size, specimen=specimen
        )

    operator = np.eye(n_components, dtype=complex) - delta_sigma * field_operator
    rhs = (delta_sigma * e_inc).reshape(-1)
    j_ind = np.linalg.solve(operator, rhs).reshape(-1, 3)
    return j_ind, volumes


def make_crack_points(
    crack_profile, *, cell_size_m=DEFAULT_CRACK_CELL_SIZE_M, x_cells=1
):
    """Convert a binary (y, depth) profile into centers of uniform 3D crack cells.

    Each true pixel is extended through x_cells along x. The total crack width
    is x_cells * cell_size_m[0]. The crack is centered at x=y=0, with its top at
    z=0 and depth increasing toward negative z. Only true pixels become cells.
    """
    profile = np.asarray(crack_profile)
    if profile.ndim != 2 or not np.all(np.isin(profile, (0, 1))):
        raise ValueError("crack_profile must be a two-dimensional binary array")
    if not isinstance(x_cells, (int, np.integer)) or x_cells < 1:
        raise ValueError("x_cells must be a positive integer")
    cell_size = _validate_cell_size(cell_size_m)
    y_indices, depth_indices = np.nonzero(profile)
    x = (np.arange(x_cells) - (x_cells - 1) / 2) * cell_size[0]
    y = (y_indices - (profile.shape[0] - 1) / 2) * cell_size[1]
    z = -(depth_indices + 0.5) * cell_size[2]
    return np.column_stack(
        (np.repeat(x, len(y)), np.tile(y, x_cells), np.tile(z, x_cells))
    )


def build_electric_field_operator(
    crack_points,
    frequency_hz,
    *,
    cell_size_m=DEFAULT_CRACK_CELL_SIZE_M,
    specimen=None,
):
    """Build the approximate bulk-conductor matrix K, shaped (3*n, 3*n).

    Vectors flatten in cell order: x0, y0, z0, x1, y1, z1, ... . K includes
    integration over each source cell, so K @ j_ind.reshape(-1) gives electric
    fields without another volume multiplier. Each cell is uniformly excited.
    Surface and layer reflection terms are omitted.
    """
    points, cell_size = _validate_cell_geometry(crack_points, cell_size_m)
    specimen = MockSpecimen() if specimen is None else specimen
    conductivity, decay = _material_properties(frequency_hz, specimen)
    n = len(points)
    if n == 0:
        return np.empty((0, 0), dtype=complex)

    nodes, weights = leggauss(4)
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1)
    sample_offsets = grid.reshape(-1, 3) * cell_size / 2
    sample_weights = (
        (
            weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        ).ravel()
        * np.prod(cell_size)
        / 8
    )

    displacements = points[:, None, :] - points[None, :, :]
    tensor = np.zeros((n, n, 3, 3), dtype=complex)
    for offset, weight in zip(sample_offsets, sample_weights, strict=True):
        tensor += weight * _bulk_electric_kernel(
            displacements - offset, decay, conductivity
        )

    # Point sampling misses the singular self term; integrate that cell separately.
    indices = np.arange(n)
    tensor[indices, indices] = _self_interaction(
        cell_size, decay, conductivity, nodes, weights, sample_offsets, sample_weights
    )
    return tensor.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)


def _bulk_electric_kernel(displacements, decay, conductivity):
    """Map point-source currents to fields in an unbounded conductor.

    With g=exp(-decay*r)/(4*pi*r), the kernel is
    (Hessian(g) - decay^2*g*I) / conductivity away from the source. The Hessian
    is the matrix of second derivatives. _self_interaction handles the source
    singularity, where the point-source expression cannot be evaluated.
    """
    radius = np.linalg.norm(displacements, axis=-1)
    direction = displacements / radius[..., None]
    scaled_radius = decay * radius
    outer = direction[..., :, None] * direction[..., None, :]
    tensor = (3 + 3 * scaled_radius + scaled_radius**2)[..., None, None] * outer
    tensor -= (1 + scaled_radius + scaled_radius**2)[..., None, None] * np.eye(3)
    scale = np.exp(-scaled_radius) / (4 * np.pi * conductivity * radius**3)
    return scale[..., None, None] * tensor


def _self_interaction(
    cell_size, decay, conductivity, nodes, weights, sample_offsets, sample_weights
):
    """Integrate the singular kernel over the cuboid containing its own center.

    Integrating the Hessian becomes an integral over the six cell faces. The
    static contribution is computed analytically using each face's solid angle;
    quadrature supplies the frequency-dependent correction. Symmetry makes the
    self block diagonal. The static correction includes the contribution located
    exactly at the source, which ordinary point sampling would miss.
    """
    half_size = cell_size / 2
    radius = np.linalg.norm(sample_offsets, axis=-1)
    scalar_integral = np.sum(
        sample_weights * np.exp(-decay * radius) / (4 * np.pi * radius)
    )
    face_weights = weights[:, None] * weights[None, :]
    diagonal = np.empty(3, dtype=complex)
    for axis in range(3):
        other_axes = [i for i in range(3) if i != axis]
        a, b = half_size[other_axes]
        face_radius = np.sqrt(
            half_size[axis] ** 2 + (a * nodes[:, None]) ** 2 + (b * nodes[None, :]) ** 2
        )
        scaled_radius = decay * face_radius
        gradient_correction = -(
            np.expm1(-scaled_radius) + scaled_radius * np.exp(-scaled_radius)
        ) / (4 * np.pi * face_radius**3)
        depolarization = (
            2 / np.pi * np.arctan(a * b / (half_size[axis] * np.linalg.norm(half_size)))
        )
        hessian_integral = -depolarization + 2 * half_size[axis] * a * b * np.sum(
            face_weights * gradient_correction
        )
        diagonal[axis] = (hessian_integral - decay**2 * scalar_integral) / conductivity
    return np.diag(diagonal)


def _validate_cell_size(cell_size_m):
    cell_size = np.asarray(cell_size_m, dtype=float)
    if (
        cell_size.shape != (3,)
        or not np.all(np.isfinite(cell_size))
        or not np.all(cell_size > 0)
    ):
        raise ValueError(
            "cell_size_m must contain three finite positive XYZ dimensions"
        )
    return cell_size


def _validate_cell_geometry(crack_points, cell_size_m):
    """Require finite centers and distinct, nonoverlapping physical cells."""
    points = np.asarray(crack_points, dtype=float)
    cell_size = _validate_cell_size(cell_size_m)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(
            "crack_points must contain finite XYZ coordinates shaped (n, 3)"
        )
    offsets = np.abs(points[:, None, :] - points[None, :, :])
    # Allow rounding at shared cell faces when coordinates come from a grid.
    overlap = np.all(offsets < cell_size * (1 - 1e-12), axis=-1)
    np.fill_diagonal(overlap, False)
    if np.any(overlap):
        raise ValueError("crack cells must not overlap; check centers and cell_size_m")
    return points, cell_size


def _material_properties(frequency_hz, specimen):
    """Return conductivity and the decaying branch sqrt(i*omega*mu*sigma)."""
    conductivity = specimen.conductivity_s_per_m
    permeability = specimen.relative_permeability
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    if not np.isfinite(conductivity) or conductivity <= 0:
        raise ValueError("specimen conductivity must be finite and positive")
    if not np.isfinite(permeability) or permeability <= 0:
        raise ValueError("specimen relative permeability must be finite and positive")
    omega = 2 * np.pi * frequency_hz
    decay = np.sqrt(
        1j * omega * VACUUM_PERMEABILITY_H_PER_M * permeability * conductivity
    )
    return conductivity, decay
