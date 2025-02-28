import os
import argparse
import sisl
import numpy as np
import multiprocessing as mp

dir_map = {'x': 0, 'a': 0, 'y': 1, 'b': 1, 'z': 2, 'c': 2}


# Read some system input for files and geometry orientation
parser = argparse.ArgumentParser(description="Compute LDOS from TBTrans Calculation")
parser.add_argument("syslabel", type=str,
                    help="System Label before file endings (e.g. LABEL for LABEL.fdf, LABEL.TBT.nc)")
parser.add_argument("transport_direction", type=str, choices=["x", "y", "z", "a", "b", "c"],
                    help="Transport direction (xyz or abc)")
parser.add_argument("vacuum_direction", type=str, choices=["x", "y", "z", "a", "b", "c"],
                    help="Vacuum direction (xyz or abc)")

# Parse arguments
args = parser.parse_args()
syslabel = args.syslabel
transport = dir_map[args.transport_direction]
vacuum = dir_map[args.vacuum_direction]

print(f"Transport axis: {transport}")
print(f"Vacuum axis:    {vacuum}")

if transport == vacuum:
    print("Transport and Vacuum direction cannot be same")
    os._exit(1)

ave = ({0, 1, 2} - {transport, vacuum}).pop()   # Direction that is not transport or vacuum

# TBTrans output file
tbt = sisl.io.tbtrans.tbtncSileTBtrans(syslabel + ".TBT.nc")
# FDF input file with geometry, basis and lattice information
fdf = sisl.io.siesta.fdfSileSiesta(syslabel + ".fdf")

print(f"Reading files: " + syslabel + ".TBT.nc " + syslabel + ".fdf")

# Get grid information
grid_template = fdf.read_grid("ElectrostaticPotential")
grid_template.fill(0)
shape = grid_template.shape


# Setup geometry information for grid
geo = fdf.read_geometry()
# Set supercell size, making sure that all indices are at least 3 for orbitals going beyond lattice cell
geo.set_nsc(geo.find_nsc())
geo.set_nsc(np.where(geo.nsc < 3, 3, geo.nsc))

a1 = geo.cell[0]
a2 = geo.cell[1]
a3 = geo.cell[2]
dx = shape[0] / np.linalg.norm(a1)
dy = shape[1] / np.linalg.norm(a2)
dz = shape[2] / np.linalg.norm(a3)

print(f"Lattice Cell: \n{geo.cell}")
print(f"Grid Shape: {shape}") 

# Get energy resolved density matrix (DM)
density_matrices = [tbt.density_matrix(float(E_pt), kavg=True, geometry=geo) for E_pt in tbt.E]

# Projects energy resolved DM to real space densities
def compute_ldos(args):
    
    try: 
        mat, E_pt = args
        temp_grid = grid_template.copy()
        # Process matrix density onto real space
        print(f"Processing Energy  {E_pt:.3f}", flush=True)
        mat.density(temp_grid)
        print(f"Processing Energy  {E_pt:.3f} Done", flush=True)

        return temp_grid.grid, E_pt
   
    except Exception as e:
        print(f"Error at energy {E_pt}: {e}")
        os._exit(1)


if __name__ == "__main__":

    # Limited memory bandwidth, deps on the grid spacings, size
    num_cores = int(os.getenv("SLURM_NTASKS", default=mp.cpu_count()))

    print(f"Number of cores: {num_cores}", flush=True)
    # Parallelize the projection onto real space across energies
    arg_list = list(zip(density_matrices, tbt.E))
    print(f"Total Energies: {tbt.nE}", flush=True)
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(compute_ldos, arg_list)

    calc, E = zip(*results)
    E = np.array(E)
    sort_indices = np.argsort(E)
    sort_calc = [calc[i] for i in sort_indices]
    sort_E = E[sort_indices]

    ldos_grids = np.zeros((grid_template.shape[transport], grid_template.shape[vacuum], tbt.nE), dtype=np.float64)
    for idx, grid_data in enumerate(sort_calc):
        ldos_grids[:, :, idx] = np.mean(grid_data, axis=ave)

    np.save("ldos_arr_par.npy", ldos_grids)
    #p = grid_data.plot.grid(axes = 'xz', show_cell = False)
    #p.write_image("test_par.png")
