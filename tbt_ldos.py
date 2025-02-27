import os
import sys
import sisl
import numpy as np
import multiprocessing as mp


transport = sys.argv[1]
print(f"Transport direction: {transport}")


# TBTrans output file
tbt = sisl.io.tbtrans.tbtncSileTBtrans("Device.TBT.nc")
# FDF input file with geometry, basis and lattice information
fdf = sisl.io.siesta.fdfSileSiesta("Device.fdf")


# Get grid information
grid_template = fdf.read_grid("ElectrostaticPotential")
grid_template.fill(0)
shape = grid_template.shape


# Setup geometry information for grid
geo = fdf.read_geometry()
geo.set_nsc(geo.find_nsc()) # Set optimal supercells for orbital calculations

a1 = geo.cell[0]; dx = shape[0] / np.linalg.norm(a1)
a2 = geo.cell[1]; dy = shape[1] / np.linalg.norm(a2)
a3 = geo.cell[2]; dz = shape[2] / np.linalg.norm(a3)

print(f"Lattice Cell: \n{geo.cell}")
print(f"Grid Shape: {shape}") 

# Get energy resolved density matrix (DM)
density_matrices = [tbt.density_matrix(float(E_pt), kavg=True, geometry=geo) for E_pt in tbt.E]

# Projects energy resolved DM to real space densities
def compute_ldos(args):

    mat, E_pt = args
    temp_grid = grid_template.copy()
    # Process matrix density onto real space
    print(f"Processing Energy  {E_pt:.3f}", flush=True)
    mat.density(temp_grid)
    print(f"Processing Energy  {E_pt:.3f} Done", flush=True)

    return temp_grid.grid, E_pt

if __name__ == "__main__":

    # Limited memory bandwidth, deps on the grid spacings, size
    num_cores = int(os.getenv("SLURM_NTASKS", default=mp.cpu_count()))

    print(f"Number of cores: {num_cores}", flush=True)
    # Parallelize the projection onto real space across energies
    arg_list = list(zip(density_matrices, tbt.E))
    print(f"Total Energies: {tbt.nE}", flush=True)
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(compute_ldos,arg_list)

    calc, E = zip(*results)
    E = np.array(E)
    sort_indices = np.argsort(E)
    sort_calc = [calc[i] for i in sort_indices]
    sort_E = E[sort_indices]

    if transport == 'y':
        # x-summed grid, LDOS(y,z,E)
        ldos_grids = np.zeros((grid_template.shape[1], grid_template.shape[2], tbt.nE), dtype = np.float64)
        for idx, grid_data in enumerate(sort_calc):
            ldos_grids[:,:,idx] = np.sum(grid_data, axis = 0) * dx
    else:
        # y-summed grid, LDOS(x,z,E)
        ldos_grids = np.zeros((grid_template.shape[0], grid_template.shape[2], tbt.nE), dtype = np.float64)
        for idx, grid_data in enumerate(sort_calc):
            ldos_grids[:,:,idx] = np.sum(grid_data, axis = 1) * dy


    np.save("ldos_arr_par.npy", ldos_grids)
    #p = grid_data.plot.grid(axes = 'xz', show_cell = False)
    #p.write_image("test_par.png")
