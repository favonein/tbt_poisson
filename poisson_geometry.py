import numpy as np
from sisl.io.siesta import fdfSileSiesta
from sisl.io.tbtrans import tbtncSileTBtrans

# System parameters
#
# Nx, Nz, Nx_metal
# x_full, z_axis, energy_axis
# dx, dz, dE
# Ec, Ev
# D(x,z,E)


class LDOSConfig:
    def __init__(self, fdfFile, tbtFile, ldosFile, metalDOSFile, semiDOSFile,
                 extend_metal_x=50, extend_semi_x=500, ldos_tol=1e-6):
        # DOS Input files
        ldos_device = np.load(ldosFile)  # Junction LDOS d(x, z, E)
        bulk_dos_metal = np.load(metalDOSFile)  # Metal bulk DOS D(z, E)
        bulk_dos_semi = np.load(semiDOSFile)  # Semiconductor bulk DOS D(z, E)

        ldos_scatt, Nx_junction, x_len, z_len = self._extractFromLDOS(ldos_device, ldos_tol)

        # Create real-space grids
        x_junction = np.linspace(0, x_len, Nx_junction)
        self.z_axis = np.linspace(0, z_len, self.Nz)

        # Real space for extension region
        self.Nx_metal, Nx_semi = int(extend_metal_x / self.dx), int(extend_semi_x / self.dx)
        x_metal = np.linspace(-extend_metal_x, 0, self.Nx_metal)
        x_semi = np.linspace(x_len, x_len + extend_semi_x, Nx_semi)

        self.x_full = np.concatenate([x_metal, x_junction, x_semi])
        self.Nx = len(self.x_full)

        print(f"Updated grid: Nx = {self.Nx}, Nz = {self.Nz}, dx = {self.dx:.3f} Ang, dz = {self.dz:.3f} Ang")

        # Compute CBM, VBM
        self.E_c = self.energy_axis[np.where(bulk_dos_semi > ldos_tol)[0][0]]  # CBM
        self.E_v = self.energy_axis[np.where(bulk_dos_semi > ldos_tol)[0][-1]]  # VBM

        print(f"Computed E_c = {self.E_c:.3f} eV, E_v = {self.E_v:.3f} eV")

        D_E_xz_junction = self._macroscopicAve(ldos_scatt)
        # Combine bulk DOS and junction LDOS
        D_E_xz = np.zeros((len(self.x_full), len(self.z_axis), len(self.energy_axis)))
        for i in range(self.x_full):
            if i < self.Nx_metal:                    # Metal Region
                decay = np.exp(k * (self.Nx_metal - i) * self.dx)
                D_E_xz[i, :, :] = D_E_xz_junction[i, :, :] * decay + bulk_dos_metal * (1 - decay)
            elif i > self.Nx_metal + Nx_junction:    # SC Region
                decay = np.exp(-k * (i - self.Nx_metal - Nx_junction) * self.dx)
                D_E_xz[i, :, :] = D_E_xz_junction[i, :, :] * decay + bulk_dos_semi * (1 - decay)
            else:                               # Junction Region
                D_E_xz[i, :, :] = D_E_xz_junction[i, :, :]

    def _extractFromLDOS(self, ldos_junction, ldos_tol=1e-6):
        # Lattice Info
        fdf = fdfSileSiesta("Device_long.fdf")
        tbt = tbtncSileTBtrans("Device.TBT.nc")

        lat = fdf.read_lattice(True)  # Read lattice vectors
        a1, a2, a3 = lat.cell  # Extract lattice vectors
        self.energy_axis = tbt.E  # eV
        self.dE = self.energy_axis[1] - self.energy_axis[0]

        # Define real-space dimensions based on transport and out-of-plane directions
        x_len = np.linalg.norm(a1)  # Transport direction (a1)
        z_len = np.linalg.norm(a3)  # Out-of-plane direction (a3)

        Nx_junction, self.Nz = ldos_junction.shape[:2]
        self.dx = x_len / Nx_junction  # Corrected x spacing
        self.dz = z_len / self.Nz  # Corrected z spacing

        cut = int(3 / self.dx)
        # Extract scattering region from grid
        ldos_x = np.sum(ldos_junction, axis=(1, 2)) * self.dz * self.dE
        # Get valid indices where LDOS integral is above tolerance
        valid_idx = np.where(ldos_x >= ldos_tol)[0]
        # Remove one more angstrom from either side to get comfortably in the Scatt region
        valid_idx = valid_idx[cut:-cut]
        ldos_scatt = ldos_junction[valid_idx, :, :]  # Shape: (n_valid_x, nz, nE)

        Nx_junction_trim = ldos_scatt.shape[0]

        print(f"Lattice dimensions: x_len = {x_len:.3f} Ang, z_len = {z_len:.3f} Ang")

        return ldos_scatt, Nx_junction_trim, x_len, z_len


    def _macroscopicAve(self, ldos_junction, metal_lattice_const = 1, semi_lattice_const = 1):
        # Define square wave kernels
        metal_sq = np.ones(int(metal_lattice_const / self.dx)) / (metal_lattice_const / self.dx)
        semi_sq = np.ones(int(semi_lattice_const / self.dx)) / (semi_lattice_const / self.dx)
        kernel = np.convolve(metal_sq, semi_sq, mode="same")
        # Convolve along x-axis for metal and semiconductor regions
        D_E_xz_junction = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=ldos_junction)

        return D_E_xz_junction

