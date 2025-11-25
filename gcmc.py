import os
import numpy as np
from ase.io import write
from unit import *
from utilities import _random_rotation, random_position, vdw_overlap


class AI_GCMC:
    def __init__(self, model, atoms_frame, atoms_ads, T, P, fugacity, device, vdw_radii):
        self.model = model
        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.atoms_ads = atoms_ads
        self.n_ads = len(self.atoms_ads)
        self.cell = np.array(self.atoms_frame.get_cell())
        # ASE cell is in Å → volume already Å^3 in our unit system
        self.V = float(np.linalg.det(self.cell))
        self.T = T
        self.P = P
        self.fugacity = fugacity
        self.device = device
        self.beta = 1.0 / (boltzmann * T) if T != 0 else 0.0
        self.Z_ads = 0

        # vdw radii provided in Å; slight shrink to avoid immediate overlaps
        self.vdw = vdw_radii - 0.35
        if not os.path.exists("results"):
            os.mkdir("results")

    def _insertion_acceptance(self, e_trial, e):
        exp_value = self.beta * (e - e_trial)
        if exp_value > 100:
            return True
        elif exp_value < -100:
            return False
        else:
            pref = self.V * self.beta * self.fugacity / max(self.Z_ads, 1)
            acc = min(1.0, pref * np.exp(exp_value))
            return np.random.rand() < acc

    def _deletion_acceptance(self, e_trial, e):
        exp_value = -self.beta * (e_trial - e)
        if exp_value > 100:
            return True
        pref = (self.Z_ads + 1) / (self.V * self.beta * self.fugacity)
        acc = min(1.0, pref * np.exp(exp_value))
        return np.random.rand() < acc

    def run(self, N):
        atoms = self.atoms_frame.copy()
        atoms.calc = self.model
        e = atoms.get_potential_energy()

        uptake = []
        adsorption_energy = []

        for iteration in range(N):
            switch = np.random.rand()

            # Insertion
            if switch < 0.25:
                self.Z_ads += 1
                atoms_trial = atoms + self.atoms_ads.copy()
                pos = atoms_trial.get_positions()
                pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
                atoms_trial.set_positions(pos)

                if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads - 1):
                    e_trial = 1.0e10
                else:
                    atoms_trial.calc = self.model
                    e_trial = atoms_trial.get_potential_energy()

                if self._insertion_acceptance(e_trial, e):
                    atoms = atoms_trial
                    e = e_trial
                else:
                    self.Z_ads -= 1

            # Deletion
            elif switch < 0.5:
                if self.Z_ads > 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    self.Z_ads -= 1
                    del atoms_trial[self.n_frame + self.n_ads * i_ads: self.n_frame + self.n_ads * (i_ads + 1)]
                    atoms_trial.calc = self.model
                    e_trial = atoms_trial.get_potential_energy()

                    if self._deletion_acceptance(e_trial, e):
                        atoms = atoms_trial
                        e = e_trial
                    else:
                        self.Z_ads += 1

            # Translation
            elif switch < 0.75:
                if self.Z_ads > 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    pos = atoms_trial.get_positions()
                    disp = 0.5 * (np.random.rand(3) - 0.5)  # Å
                    pos[self.n_frame + self.n_ads * i_ads: self.n_frame + self.n_ads * (i_ads + 1)] += disp
                    atoms_trial.set_positions(pos)

                    if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                        e_trial = 1.0e10
                    else:
                        atoms_trial.calc = self.model
                        e_trial = atoms_trial.get_potential_energy()

                    dE = e_trial - e
                    acc = min(1.0, np.exp(-self.beta * dE))
                    if np.random.rand() < acc:
                        atoms = atoms_trial
                        e = e_trial

            # Rotation
            else:
                if self.Z_ads > 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    pos = atoms_trial.get_positions()
                    before = pos[self.n_frame + self.n_ads * i_ads: self.n_frame + self.n_ads * (i_ads + 1)].copy()
                    pos[self.n_frame + self.n_ads * i_ads: self.n_frame + self.n_ads * (i_ads + 1)] = _random_rotation(before, circlefrac=0.1)
                    atoms_trial.set_positions(pos)

                    if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                        e_trial = 1.0e10
                    else:
                        atoms_trial.calc = self.model
                        e_trial = atoms_trial.get_potential_energy()

                    dE = e_trial - e
                    acc = min(1.0, np.exp(-self.beta * dE))
                    if np.random.rand() < acc:
                        atoms = atoms_trial
                        e = e_trial

            uptake.append(self.Z_ads)
            adsorption_energy.append(e)

            if iteration % 1000 == 0:
                outpath = f"results/snapshot_{self.P / bar:.5f}bar_iteration_{iteration}.cif"
                write(outpath, atoms, format="cif")

        np.save(f"results/uptake_{self.P / bar:.5f}bar.npy", np.array(uptake))
        np.save(f"results/adsorption_energy_{self.P / bar:.5f}bar.npy", np.array(adsorption_energy))
