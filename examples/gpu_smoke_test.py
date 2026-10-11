import hoomd
import numpy as np

from hoomd.laradjisunil import LaradjiSunil


print("HOOMD version:", hoomd.version.version)
print("HOOMD module:", hoomd.__file__)
print("GPU enabled:", hoomd.version.gpu_enabled)
print("GPU platform:", hoomd.version.gpu_platform)

assert hoomd.version.gpu_enabled, "HOOMD is not GPU-enabled"

# Create an actual GPU device.
device = hoomd.device.GPU()
print("GPU device:", device)

sim = hoomd.Simulation(device=device, seed=123)

# Two particles separated by r = 0.8.
r = 0.8

snapshot = hoomd.Snapshot()
snapshot.configuration.box = [10, 10, 10, 0, 0, 0]

snapshot.particles.N = 2
snapshot.particles.types = ["A"]
snapshot.particles.typeid[:] = [0, 0]
snapshot.particles.position[:] = [
    [-r / 2, 0, 0],
    [ r / 2, 0, 0],
]

sim.create_state_from_snapshot(snapshot)

# Construct the custom Laradji-Sunil pair force.
nlist = hoomd.md.nlist.Cell(buffer=0.4)

ls = LaradjiSunil(
    nlist=nlist,
    default_r_cut=2.0,
    mode="none",
)

ls.params[("A", "A")] = {
    "Umax": 100.0,
    "Umin": -6.0,
    "rm": 1.0,
}

integrator = hoomd.md.Integrator(dt=0.001)
integrator.forces = [ls]

integrator.methods = [
    hoomd.md.methods.ConstantVolume(
        filter=hoomd.filter.All()
    )
]

sim.operations.integrator = integrator

# Initialize the force calculation without advancing dynamics.
sim.run(0)

forces = np.asarray(ls.forces)
energies = np.asarray(ls.energies)

print("\nForces:")
print(forces)

print("\nPer-particle energies:")
print(energies)

# Analytical result at r = 0.8:
#
# U = (100 - (-6)) * ((1 - 0.8) / 1)^2 - 6
#   = -1.76
#
# F = 2 * (100 - (-6)) * (1 - 0.8) / 1^2
#   = 42.4
#
# HOOMD distributes the pair energy equally between the particles.

expected_forces = np.array([
    [-42.4, 0.0, 0.0],
    [ 42.4, 0.0, 0.0],
])

expected_energies = np.array([-0.88, -0.88])

np.testing.assert_allclose(
    forces,
    expected_forces,
    rtol=1e-6,
    atol=1e-6,
)

np.testing.assert_allclose(
    energies,
    expected_energies,
    rtol=1e-6,
    atol=1e-6,
)

print("\nGPU Laradji-Sunil smoke test PASSED.")
