import numpy as np
from pyamg import ruge_stuben_solver
from pyamg.gallery import poisson
import poisson_viz
import poisson_solvers as psolve
import poisson_geometry

# from sisl.io.siesta import fdfSileSiesta
# from sisl.io.tbtrans import tbtncSileTBtrans

# Computational Parameters
tol, q_tol = 1e-6, 1e-6   # Convergence criteria for poisson, charge conservation
max_iter = 1000           # Max iterations for poisson solver
alpha = 0.15              # phi_e Mixing parameter
dos_threshold = 0         # DOS Cutoff threshold (for CBM, VBM Calc)

# Params
V = 0  # eV
extend_metal_x = 50  # Ang
extend_semi_x = 500  # Ang
metal_lattice_const = 1
semi_lattice_const = 2


fdfFile = "Device.fdf"
tbtFile = "Device.TBT.nc"
ldosFile = "calc_ldos.npy"
metalDOSFile = "bulk_dos_metal.npy"
semiDOSFile= "bulk_dos_semi.npy"

# System information. LDOS, axes and grids
sys = poisson_geometry.LDOSConfig(fdfFile, tbtFile, ldosFile, metalDOSFile, semiDOSFile, extend_metal_x, extend_semi_x)

# Guess Ef, En
E_Fn = (sys.E_c + sys.E_v + V) / 2
E_Fp = E_Fn - V
print(f'E_fn: {E_Fn:.3f}, F_fp: {E_Fp:.3f}')

c = np.zeros(sys.Nx, sys.Nz)

# Iterative poisson solver
calc = psolve.SolverConfig(sys, fn=E_Fn, fp=E_Fp)
print(f'Begin poisson iteration: ')
phi_history, rho_history, fn_history, fp_history, iters = calc.solve_iter(sys, c)


# === Visualization Function ===
poisson_viz.plot_convergence(calc, sys)
poisson_viz.plot_potential(calc, sys)
