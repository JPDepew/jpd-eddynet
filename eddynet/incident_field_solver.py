"""Incident-field solver scaffold with deterministic mock fields.

The mock kernels use smooth coil fields, artificial core coupling, and simple
depth attenuation. They omit the layered conductor's reflected fields and the
real ferrite Green functions. Use their output to exercise the solver, rather
than as physical training data.
"""

from dataclasses import dataclass

import numpy as np

VACUUM_PERMEABILITY_H_PER_M = 4e-7 * np.pi


@dataclass
class MockProbe:
    """Illustrative circular coil and cylindrical core with fixed permeability."""

    coil_xy: tuple[float, float] = (0.0, 0.0)
    inner_radius_m: float = 2.5e-3
    outer_radius_m: float = 3.15e-3
    coil_height_m: float = 2.0e-3
    turns: int = 15
    current_a: float = 1e-3
    lift_off_m: float = 0.5e-3
    core_radius_m: float = 2.0e-3
    core_height_m: float = 4.0e-3
    core_relative_permeability: float = 100.0


@dataclass
class MockSpecimen:
    """Illustrative conductive half-space occupying z <= 0."""

    conductivity_s_per_m: float = 35e6
    relative_permeability: float = 1.0


@dataclass
class CoreMesh:
    """Core cell centers shaped (n, 3), in meters, and volumes shaped (n,), in m^3."""

    centers: np.ndarray
    volumes: np.ndarray


def solve_incident_field(
    crack_points,
    frequency_hz,
    *,
    core_mesh=None,
    probe=None,
    specimen=None,
):
    """Return mock complex electric fields shaped (n_points, 3), in V/m.

    Points are XYZ coordinates in meters; z=0 is the specimen surface.
    Missing setup arguments use MockProbe, MockSpecimen, and a cylindrical mesh.
    The matrix solve follows (I - chi * G_HM * V) * M = chi * H_coil.
    The field kernels are mocks, so this is not a physical forward model.
    """
    probe = MockProbe() if probe is None else probe
    specimen = MockSpecimen() if specimen is None else specimen
    points = np.asarray(crack_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(
            "crack_points must contain finite XYZ coordinates shaped (n, 3)"
        )
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    if not np.isfinite(probe.core_relative_permeability) or (
        probe.core_relative_permeability <= 0
    ):
        raise ValueError("core_relative_permeability must be finite and positive")

    core_mesh = make_mock_core_mesh(probe) if core_mesh is None else core_mesh
    core_mesh = _validate_core_mesh(core_mesh)
    h_coil = mock_coil_magnetic_field(core_mesh.centers, probe)
    green_hm = mock_magnetic_interaction_tensor(core_mesh)
    chi = probe.core_relative_permeability - 1
    volumes = np.repeat(core_mesh.volumes, 3)
    n_components = len(volumes)

    weighted_green_hm = green_hm.reshape(n_components, n_components) * volumes
    operator = np.eye(n_components) - chi * weighted_green_hm
    magnetization = np.linalg.solve(operator, chi * h_coil.reshape(-1))

    e_coil = mock_coil_electric_field(points, probe, specimen, frequency_hz)
    green_em = mock_electric_field_from_core_tensor(
        points, core_mesh, specimen, frequency_hz
    )
    e_core = green_em.reshape(3 * len(points), n_components) @ (volumes * magnetization)
    return e_coil + e_core.reshape(-1, 3)


def make_mock_core_mesh(
    probe=None, *, radial_cells=2, azimuthal_cells=8, axial_cells=3
):
    """Divide a cylindrical core into ring sectors with correct cell volumes."""
    probe = MockProbe() if probe is None else probe
    if min(radial_cells, azimuthal_cells, axial_cells) < 1:
        raise ValueError("core mesh cell counts must be positive")
    if probe.core_radius_m <= 0 or probe.core_height_m <= 0:
        raise ValueError("core radius and height must be positive")

    radial_edges = np.linspace(0, probe.core_radius_m, radial_cells + 1)
    radii = np.sqrt((radial_edges[:-1] ** 2 + radial_edges[1:] ** 2) / 2)
    angles = (np.arange(azimuthal_cells) + 0.5) * 2 * np.pi / azimuthal_cells
    heights = (
        probe.lift_off_m
        + (np.arange(axial_cells) + 0.5) * probe.core_height_m / axial_cells
    )
    radius, angle, height = np.meshgrid(radii, angles, heights, indexing="ij")
    centers = np.column_stack(
        (
            probe.coil_xy[0] + radius.ravel() * np.cos(angle.ravel()),
            probe.coil_xy[1] + radius.ravel() * np.sin(angle.ravel()),
            height.ravel(),
        )
    )
    ring_volumes = (
        np.pi
        * np.diff(radial_edges**2)
        * probe.core_height_m
        / (axial_cells * azimuthal_cells)
    )
    volumes = np.broadcast_to(ring_volumes[:, None, None], radius.shape).ravel().copy()
    return CoreMesh(centers, volumes)


def mock_coil_magnetic_field(points, probe):
    """Return a smooth axial mock H field; conductor feedback is omitted."""
    _, scale = _mock_coil_profile(points, probe)
    field = np.zeros_like(points, dtype=np.complex128)
    field[:, 2] = scale / 2
    return field


def mock_coil_electric_field(points, probe, specimen, frequency_hz):
    """Return a circulating mock E field with simple depth attenuation."""
    offsets, scale = _mock_coil_profile(points, probe)
    directions = np.column_stack((-offsets[:, 1], offsets[:, 0], np.zeros(len(points))))
    amplitude = -1j * 2 * np.pi * frequency_hz * VACUUM_PERMEABILITY_H_PER_M * scale / 4
    amplitude *= _mock_depth_attenuation(points, specimen, frequency_hz)
    return amplitude[:, None] * directions


def mock_magnetic_interaction_tensor(core_mesh):
    """Return an artificial Gaussian core coupling tensor shaped (n, 3, n, 3).

    Negative coupling limits the mock magnetization response.
    Normalizing the Gaussian gives a bounded mock interaction operator.
    This tensor is not an electromagnetic Green function.
    """
    offsets = core_mesh.centers[:, None, :] - core_mesh.centers[None, :, :]
    cell_width = np.cbrt(np.mean(core_mesh.volumes))
    coupling = np.exp(-np.sum(offsets**2, axis=-1) / (2 * cell_width**2))
    coupling /= coupling.sum(axis=1).max()
    tensor = -coupling[:, None, :, None] * np.eye(3)[None, :, None, :]
    return tensor / (3 * core_mesh.volumes[None, None, :, None])


def mock_electric_field_from_core_tensor(points, core_mesh, specimen, frequency_hz):
    """Return softened mock dipole fields shaped (n_points, 3, n_core, 3).

    Each source-component column maps magnetization to M cross displacement.
    Cell width softens the point-source singularity; layer reflections are omitted.
    """
    offsets = points[:, None, :] - core_mesh.centers[None, :, :]
    source_radius = np.cbrt(core_mesh.volumes) / 2
    distance_cubed = (np.sum(offsets**2, axis=-1) + source_radius[None, :] ** 2) ** 1.5
    component_fields = np.cross(np.eye(3), offsets[:, :, None, :])
    tensor = component_fields.transpose(0, 3, 1, 2).astype(np.complex128)
    coefficient = (
        -1j
        * 2
        * np.pi
        * frequency_hz
        * VACUUM_PERMEABILITY_H_PER_M
        / (4 * np.pi * distance_cubed)
    )
    tensor *= coefficient[:, None, :, None]
    return (
        tensor
        * _mock_depth_attenuation(points, specimen, frequency_hz)[:, None, None, None]
    )


def _mock_coil_profile(points, probe):
    """Build a smooth coil profile whose amplitude scales with turns and current."""
    center = (*probe.coil_xy, probe.lift_off_m + probe.coil_height_m / 2)
    offsets = points - np.asarray(center)
    radius = (probe.inner_radius_m + probe.outer_radius_m) / 2
    if radius <= 0:
        raise ValueError("mean coil radius must be positive")
    distance_cubed = (radius**2 + np.sum(offsets**2, axis=-1)) ** 1.5
    scale = probe.turns * probe.current_a * radius**2 / distance_cubed
    return offsets, scale


def _mock_depth_attenuation(points, specimen, frequency_hz):
    """Apply a skin-depth-shaped mock decay below the surface, without reflections."""
    conductivity = specimen.conductivity_s_per_m
    permeability = specimen.relative_permeability
    if not np.isfinite(conductivity) or conductivity < 0:
        raise ValueError("specimen conductivity must be finite and nonnegative")
    if not np.isfinite(permeability) or permeability <= 0:
        raise ValueError("specimen relative permeability must be finite and positive")
    if conductivity == 0:
        return np.ones(len(points), dtype=np.complex128)
    omega = 2 * np.pi * frequency_hz
    skin_depth = np.sqrt(
        2 / (omega * VACUUM_PERMEABILITY_H_PER_M * permeability * conductivity)
    )
    depth = np.maximum(-points[:, 2], 0)
    return np.exp(-(1 + 1j) * depth / skin_depth)


def _validate_core_mesh(core_mesh):
    """Reject malformed cells before constructing the volume-weighted operator."""
    centers = np.asarray(core_mesh.centers, dtype=float)
    volumes = np.asarray(core_mesh.volumes, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("core centers must have shape (n, 3) with at least one cell")
    if volumes.shape != (len(centers),) or not np.all(volumes > 0):
        raise ValueError("core volumes must contain one positive volume per cell")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(volumes)):
        raise ValueError("core centers and volumes must be finite")
    return CoreMesh(centers, volumes)
