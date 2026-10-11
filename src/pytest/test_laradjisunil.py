# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Standalone tests for the Laradji-Sunil pair potential."""

import itertools

import hoomd
import numpy
import pytest

from hoomd.laradjisunil import LaradjiSunil


def ls_force_and_energy(dx, Umax, Umin, rm, rc):
    """Calculate the analytical pair force and potential energy."""
    dx = numpy.asarray(dx, dtype=numpy.float64)
    r = numpy.linalg.norm(dx)

    if r >= rc:
        return numpy.zeros(3, dtype=numpy.float64), 0.0

    if r == 0.0:
        return numpy.zeros(3, dtype=numpy.float64), Umax

    if r <= rm:
        # Repulsive core.
        dr = rm - r

        energy = (Umax - Umin) * (dr / rm) ** 2 + Umin

        force_r = (
            2.0 * (Umax - Umin) * dr / rm**2
        )

    else:
        # Smooth attractive tail.
        x = (rc - r) / (rc - rm)

        energy = Umin * x**2 * (3.0 - 2.0 * x)

        force_r = (
            6.0 * Umin * x * (1.0 - x) / (rc - rm)
        )

    # Vector force points along the displacement vector.
    force = force_r * dx / r

    return force, energy


distances = [0.25, 0.75, 1.0, 1.25, 1.75, 2.2]
Umax_values = [20.0, 100.0]
Umin_values = [0.0, -6.0]
modes = ["none", "shift"]

testdata = list(
    itertools.product(
        distances,
        Umax_values,
        Umin_values,
        modes,
    )
)


@pytest.mark.parametrize(
    ("distance", "Umax", "Umin", "mode"),
    testdata,
)
def test_force_and_energy(
    distance,
    Umax,
    Umin,
    mode,
):
    """Compare HOOMD's force and energy with the analytical result."""

    rm = 1.0
    rc = 2.0

    # Create a standalone CPU simulation.
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=1)

    # Construct a snapshot containing two particles of type A.
    snapshot = hoomd.Snapshot()

    snapshot.configuration.box = [
        10.0, 10.0, 10.0, 0.0, 0.0, 0.0
    ]

    snapshot.particles.N = 2
    snapshot.particles.types = ["A"]

    snapshot.particles.typeid[:] = [0, 0]

    snapshot.particles.position[:] = [
        [-distance / 2.0, 0.0, 0.0],
        [ distance / 2.0, 0.0, 0.0],
    ]

    sim.create_state_from_snapshot(snapshot)

    # Set up the integrator and pair force.
    nlist = hoomd.md.nlist.Cell(buffer=0.4)

    pair_force = LaradjiSunil(
        nlist=nlist,
        default_r_cut=rc,
        mode=mode,
    )

    pair_force.params[("A", "A")] = {
        "Umax": Umax,
        "Umin": Umin,
        "rm": rm,
    }

    integrator = hoomd.md.Integrator(dt=0.001)

    integrator.forces = [pair_force]

    integrator.methods = [
        hoomd.md.methods.ConstantVolume(
            filter=hoomd.filter.All()
        )
    ]

    sim.operations.integrator = integrator

    # Initialize and evaluate the forces without advancing dynamics.
    sim.run(0)

    # Get the force and energy calculated by the custom component.
    forces = numpy.asarray(pair_force.forces)
    energies = numpy.asarray(pair_force.energies)

    # Independently compute the expected values.
    displacement = numpy.array(
        [distance, 0.0, 0.0],
        dtype=numpy.float64,
    )

    expected_force, pair_energy = ls_force_and_energy(
        displacement,
        Umax,
        Umin,
        rm,
        rc,
    )

    expected_forces = numpy.array([
        -expected_force,
        expected_force,
    ])

    # HOOMD reports half of the pair energy on each particle.
    expected_energies = numpy.array([
        pair_energy / 2.0,
        pair_energy / 2.0,
    ])

    numpy.testing.assert_allclose(
        forces,
        expected_forces,
        rtol=1e-6,
        atol=1e-6,
    )

    numpy.testing.assert_allclose(
        energies,
        expected_energies,
        rtol=1e-6,
        atol=1e-6,
    )
