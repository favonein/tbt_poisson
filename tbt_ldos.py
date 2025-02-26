import os
import sisl
import numpy as np
import multiprocessing as mp

# TBTrans output file
tbt = sisl.io.tbtrans.tbtncSileTBtrans("Device.TBT.nc")
# FDF input file with geometry, basis and lattice information
fdf = sisl.io.siesta.fdfSileSiesta("Device_long.fdf")

# Setup geometry information for grid
geo = fdf.read_geometry()
geo.set_nsc(geo.find_nsc()) # Set optimal supercells for orbital calculations

# Get grid information
grid_template = fdf.read_grid("ElectrostaticPotential")
grid_template.fill(0)
shape = grid_template.shape

density_matrices = [tbt.density_matrix(float(E_pt), kavg=True, geometry=geo) for E_pt in tbt.E]

def compute_ldos(args):

    mat, E_pt = args
    temp_grid = grid_template.copy()
    # Process matrix density onto real space
    print(f"Processing Energy  {E_pt:.3f}", flush=True)
    mat.density(temp_grid)
    print(f"Processing Energy  {E_pt:.3f} Done", flush=True)

    return temp_grid.grid, E_pt

if __name__ == "__main__":

    # on nautilus, memory limited to 8
    max_cores_tested = 8
    num_cores = min(int(os.getenv("SLURM_NTASKS", default=mp.cpu_count())), max_cores_tested)

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

    # x-summed grid, LDOS(x,z,E)
    ldos_grids = np.zeros((grid_template.shape[0], grid_template.shape[2], tbt.nE), dtype = np.float64)
    for idx, grid_data in enumerate(sort_calc):
        ldos_grids[:,:,idx] = np.sum(grid_data, axis = 1)


    np.save("ldos_arr_par.npy", ldos_grids)
    #p = grid_data.plot.grid(axes = 'xz', show_cell = False)
    #p.write_image("test_par.png")
