"""For crack shape, coil position, and freq, calculate the field the coil
sends to the crack, solve for the current the crack induces, and combine
those fields to get the coil's impedance change"""

# explain simulation equation code

import dataclasses
import functools
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

from eddynet.incident_field_solver import MockProbe, solve_incident_field
from eddynet.induced_current_solver import (
    build_electric_field_operator,
    make_crack_points,
    prepare_induced_current_solver,
    solve_induced_current,
)

# Starter values for the probe and specimen. The EddyNet paper does not give
# these dimensions or material values, so callers should replace them with
# values for the probe and material they want to simulate.
# maybe use 1, 3, and 6 for frequencies
DEFAULT_FREQUENCIES_MHZ = (2.0, 2.5, 3.0)
DEFAULT_COIL_POSITION_M = (0.0, 0.0)
DEFAULT_SCAN_STEP_M = 0.1e-3
DEFAULT_PROFILE_PIXEL_SPACING_Y_M = 0.1e-3
DEFAULT_PROFILE_PIXEL_SPACING_Z_M = 0.1e-3
DEFAULT_COIL_LIFT_OFF_M = 0.5e-3
DEFAULT_SOURCE_CURRENT_A = 1e-3
DEFAULT_COIL_INNER_RADIUS_M = 2.5e-3
DEFAULT_COIL_OUTER_RADIUS_M = 3.15e-3
DEFAULT_COIL_HEIGHT_M = 2.0e-3
DEFAULT_COIL_TURNS = 15
DEFAULT_FERRITE_CORE_RADIUS_M = 2.0e-3
DEFAULT_FERRITE_CORE_HEIGHT_M = 4.0e-3
DEFAULT_FERRITE_RELATIVE_PERMEABILITY = 100.0
DEFAULT_SPECIMEN_CONDUCTIVITY_S_PER_M = 35e6
DEFAULT_SPECIMEN_RELATIVE_PERMEABILITY = 1.0
DEFAULT_SPECIMEN_THICKNESS_M = None
DEFAULT_X_STEPS = 40
DEFAULT_Y_STEPS = 40


def impedance_change(e_inc, j_ind, weights, source_current):
    """Return one impedance change per field shaped (..., n_points, 3)."""
    pointwise_dot = np.sum(e_inc * j_ind, axis=-1)
    return -np.sum(pointwise_dot * weights, axis=-1) / source_current**2


def generate_crack_profile(x_dim=40, y_dim=12, *, plot=False):
    crack = np.random.choice([True, False], size=(x_dim, y_dim), replace=True)
    smoothed_crack = crack.copy()
    for _ in range(3):
        smoothed_crack = median_filter(smoothed_crack, size=(3, 3))
    if plot:
        plot_crack_profiles(crack, smoothed_crack)
    return smoothed_crack


def plot_crack_profiles(crack, smoothed_crack):
    """Display the generated crack profile beside its smoothed version."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, profile, title in zip(
        axes,
        (crack, smoothed_crack),
        ("Random crack", "Smoothed crack"),
        strict=True,
    ):
        axis.imshow(profile, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xlabel("Profile column")
        axis.set_ylabel("Profile row")
        axis.set_xticks([])
        axis.set_yticks([])

    plt.show()
    return figure, axes


def incident_field(points, coil_xy, frequency_hz, sample_a_phi):
    dx = points[:, 0] - coil_xy[0]
    dy = points[:, 1] - coil_xy[1]

    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)

    a_phi = sample_a_phi(radius, points[:, 2])
    e_phi = -1j * 2 * np.pi * frequency_hz * a_phi

    return np.column_stack(
        (
            -e_phi * np.sin(angle),
            e_phi * np.cos(angle),
            np.zeros_like(e_phi),
        )
    )


def precompute_incident_fields(
    points, frequencies_mhz, *, probe, x_steps, y_steps, scan_step_m
):
    """Cache fields shaped (scan_positions, grid_cells, 3) for each frequency."""
    fields_by_frequency = {}
    with Pool() as pool:
        for freq_mhz in frequencies_mhz:
            calc = functools.partial(
                calc_incident_field,
                points=points,
                freq=freq_mhz * 1e6,
                probe=probe,
                scan_step_m=scan_step_m,
            )
            # starmap preserves the row/column order produced by product.
            fields = pool.starmap(calc, product(range(x_steps), range(y_steps)))
            fields_by_frequency[freq_mhz] = np.asarray(fields, dtype=complex).reshape(
                x_steps * y_steps, len(points), 3
            )
    return fields_by_frequency


def generate_test_data(
    size=20_000,
    *,
    frequencies_mhz=DEFAULT_FREQUENCIES_MHZ,
    coil_position_m=DEFAULT_COIL_POSITION_M,
    scan_step_m=DEFAULT_SCAN_STEP_M,
    profile_pixel_spacing_y_m=DEFAULT_PROFILE_PIXEL_SPACING_Y_M,
    profile_pixel_spacing_z_m=DEFAULT_PROFILE_PIXEL_SPACING_Z_M,
    coil_lift_off_m=DEFAULT_COIL_LIFT_OFF_M,
    source_current_a=DEFAULT_SOURCE_CURRENT_A,
    coil_inner_radius_m=DEFAULT_COIL_INNER_RADIUS_M,
    coil_outer_radius_m=DEFAULT_COIL_OUTER_RADIUS_M,
    coil_height_m=DEFAULT_COIL_HEIGHT_M,
    coil_turns=DEFAULT_COIL_TURNS,
    ferrite_core_radius_m=DEFAULT_FERRITE_CORE_RADIUS_M,
    ferrite_core_height_m=DEFAULT_FERRITE_CORE_HEIGHT_M,
    ferrite_relative_permeability=DEFAULT_FERRITE_RELATIVE_PERMEABILITY,
    specimen_conductivity_s_per_m=DEFAULT_SPECIMEN_CONDUCTIVITY_S_PER_M,
    specimen_relative_permeability=DEFAULT_SPECIMEN_RELATIVE_PERMEABILITY,
    specimen_thickness_m=DEFAULT_SPECIMEN_THICKNESS_M,
    x_steps=DEFAULT_X_STEPS,
    y_steps=DEFAULT_Y_STEPS,
):
    """Reuse grid interactions and incident fields; factor each crack/frequency once."""
    root_dir = Path("data")
    profile_shape = (40, 12)
    full_grid_points = make_crack_points(np.ones(profile_shape, dtype=bool))
    full_field_operators = {
        freq_mhz: build_electric_field_operator(full_grid_points, freq_mhz * 1e6)
        for freq_mhz in frequencies_mhz
    }
    full_incident_fields = precompute_incident_fields(
        full_grid_points,
        frequencies_mhz,
        probe=MockProbe(current_a=source_current_a),
        x_steps=x_steps,
        y_steps=y_steps,
        scan_step_m=scan_step_m,
    )

    for i in tqdm(range(size)):
        crack_id = f"{i:06d}"
        crack = generate_crack_profile(*profile_shape)
        crack_path = root_dir / "cracks"
        crack_path.mkdir(exist_ok=True, parents=True)
        np.savez(crack_path / f"{crack_id}.npz", crack=crack)
        points = make_crack_points(crack)
        cell_indices = np.flatnonzero(crack)
        # Each cell occupies three consecutive matrix rows and columns for XYZ.
        component_indices = (3 * cell_indices[:, None] + np.arange(3)).ravel()

        for freq_mhz in frequencies_mhz:
            freq_path = root_dir / f"{freq_mhz}MHz"
            freq_path.mkdir(exist_ok=True, parents=True)

            freq = freq_mhz * 1e6
            field_operator = full_field_operators[freq_mhz][
                np.ix_(component_indices, component_indices)
            ]

            solver = prepare_induced_current_solver(
                points, freq, field_operator=field_operator
            )
            # The incident field does not depend on which cells form the crack.
            e_inc = full_incident_fields[freq_mhz][:, cell_indices, :]
            j_ind, weights = solver.solve(e_inc)
            response_map = impedance_change(
                e_inc, j_ind, weights, source_current_a
            ).reshape(x_steps, y_steps)

            np.savez(freq_path / f"{crack_id}.npz", delta_z=response_map)


def calc_impedance_change(x, y, points, freq, probe, source_current_a, response_map):
    e_inc = solve_incident_field(points, frequency_hz=freq, probe=probe)
    j_ind, weights = solve_induced_current(points, e_inc, freq)

    delta_z = impedance_change(e_inc, j_ind, weights, source_current_a)

    response_map[x, y] = delta_z


def calc_incident_field(row, column, points, freq, probe: MockProbe, scan_step_m):
    """Return crack-point fields; map rows follow probe y and columns follow x."""
    positioned_probe = dataclasses.replace(
        probe, coil_xy=(column * scan_step_m, row * scan_step_m)
    )
    return solve_incident_field(points, frequency_hz=freq, probe=positioned_probe)


if __name__ == "__main__":
    # generate_crack_profile(plot=True)
    generate_test_data(size=100)
