# This file extracts the real space, energy resolved LDOS
# from the TSDE Files from electrode calculations
import os
import sisl
import numpy as np
import multiprocessing as mp

# FDF input file with geometry, basis and lattice information
fdf_L = sisl.io.siesta.fdfSileSiesta("Left/Left.fdf")
fdf_R = sisl.io.siesta.fdfSileSiesta("Right/Right.fdf")

# Setup geometry information for grid
geo_L = fdf_L.read_geometry()
geo_L.set_nsc(geo_L.find_nsc()) # Set optimal supercells for orbital calculations
geo_R = fdf_R.read_geometry()
geo_R.set_nsc(geo_R.find_nsc()) # Set optimal supercells for orbital calculations

# Get grid information
grid_L = fdf_L.read_grid("ElectrostaticPotential")
grid_L.fill(0)
shape_L = grid_L.shape
grid_R = fdf_R.read_grid("ElectrostaticPotential")
grid_R.fill(0)
shape_R = grid_R.shape

edm_L = fdf_L.read_energy_density_matrix()
edm_R = fdf_R.read_energy_density_matrix()

print(f"Left Grid: {shape_L}")
print(f"Right Grid: {shape_R}")
print(edm_L)
print(edm_R)