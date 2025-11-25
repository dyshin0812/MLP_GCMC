import numpy as np
from unit import *
import csv

def _random_rotation(pos, circlefrac=1.0):
    com = np.average(pos, axis=0)
    pos = pos - com

    randnums = np.random.uniform(size=(3,))
    theta, phi, z = randnums

    theta = theta * 2.0 * circlefrac * np.pi
    phi = phi * 2.0 * np.pi
    z = z * 2.0 * circlefrac

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    pos = np.einsum('ib,ab->ia', pos, M)
    return pos + com


def _random_translation(pos, rvecs):
    pos = pos - np.average(pos, axis=0)
    rnd = np.random.rand(3)
    new_cos = rnd[0] * rvecs[0] + rnd[1] * rvecs[1] + rnd[2] * rvecs[2]
    return pos + new_cos


def random_position(pos, rvecs):
    pos = _random_rotation(pos)
    pos = _random_translation(pos, rvecs)
    return pos


def vdw_overlap(atoms, vdw, n_frame, n_ads, select_ads):
    nat = len(atoms)
    pos, numbers = atoms.get_positions(), atoms.get_atomic_numbers()
    for i_ads in range(n_frame + n_ads * select_ads, n_frame + n_ads * (select_ads + 1)):
        dists = atoms.get_distances(i_ads, np.arange(nat), mic=True)
        for i, d in enumerate(dists):
            if n_frame + n_ads * select_ads <= i < n_frame + n_ads * (select_ads + 1):
                continue
            if d < vdw[numbers[i_ads]] + vdw[numbers[i]]:
                return True
    return False


# ==============================
#   Equations of State (EOS)
# ==============================
class EOS(object):
    def __init__(self, mass=0.0):
        self.mass = mass

    def calculate_fugacity(self, T, P):
        mu_ex, Pref = self.calculate_mu_ex(T, P)
        fugacity = np.exp(mu_ex / (boltzmann * T)) * Pref
        return fugacity

    def calculate_mu(self, T, P):
        mu_ex, Pref = self.calculate_mu_ex(T, P)
        mu_id = boltzmann * T * np.log(P / Pref)
        return mu_ex + mu_id

    def get_Pref(self, T, P0, deviation=1e-1, min_Pref=1e-3 * pascal):
        Pref = P0
        for _ in range(100):
            rhoref = self.calculate_rho(T, Pref)
            Zref = Pref / rhoref / boltzmann / T
            if np.abs(Zref - 1.0) > deviation:
                Pref *= 0.5
                if Pref < min_Pref:
                    raise ValueError(f"{Pref} Pa too small; check inputs")
            else:
                break
        if np.abs(Pref / self.calculate_rho(T, Pref) / boltzmann / T - 1.0) > deviation:
            raise ValueError("Failed to find ideal-like Pref")
        return Pref


class PREOS(EOS):
    """Peng–Robinson EOS (eV–Å–K units)"""

    def __init__(self, Tc, Pc, omega, mass=0.0, phase="vapour"):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.mass = mass
        self.phase = phase

        self.a = 0.457235 * (boltzmann ** 2) * (self.Tc ** 2) / self.Pc
        self.b = 0.0777961 * boltzmann * self.Tc / self.Pc
        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2

    @classmethod
    def from_name(cls, compound, csv_path="critical_acentric.csv"):
        """
        Load PREOS parameters from CSV file (column order only).
        Expected order: compound, molweight[g/mol], Tc[K], Pc[MPa], omega[-]
        """
        def norm(s):
            return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

        key = norm(compound)
        aliases = {"co2": "carbondioxide",
                   "carbon dioxide": "carbondioxide",
                   "carbon_dioxide": "carbondioxide"}
        key_alias = norm(aliases.get(key, key))

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows or len(rows) < 2:
            raise ValueError(f"{csv_path} is empty or invalid")

        data_rows = rows[1:]
        hit = None
        for r in data_rows:
            if not r or len(r) < 5 or r[0].lstrip().startswith("#"):
                continue
            name = norm(r[0])
            if name in (key, key_alias):
                hit = r
                break

        if hit is None:
            avail = sorted({norm(r[0]) for r in data_rows if r and len(r) >= 5})
            raise ValueError(f"Compound '{compound}' not found. Available: {avail}")

        try:
            molweight_g_per_mol = float(hit[1])
            Tc_K = float(hit[2]) * kelvin
            Pc_MPa = float(hit[3])
            omega = float(hit[4])

            Pc_Pa = Pc_MPa * 1.0e6
            Pc_eV_A3 = Pc_Pa * pascal
            mass_kg_per_molecule = molweight_g_per_mol * amu

        except Exception as e:
            raise ValueError(f"Failed to parse row for '{compound}': {hit}, err={e}")

        return cls(Tc_K, Pc_eV_A3, omega, mass=mass_kg_per_molecule)

    def set_conditions(self, T, P):
        self.Tr = T / self.Tc
        self.alpha = (1 + self.kappa * (1 - np.sqrt(self.Tr))) ** 2
        self.A = self.a * self.alpha * P / (boltzmann * T) ** 2
        self.B = self.b * P / (boltzmann * T)

    def polynomial(self, Z):
        return Z ** 3 - (1 - self.B) * Z ** 2 + (self.A - 2 * self.B - 3 * self.B ** 2) * Z - (
            self.A * self.B - self.B ** 2 - self.B ** 3
        )

    def polynomial_roots(self):
        a = -(1 - self.B)
        b = self.A - 2 * self.B - 3 * self.B ** 2
        c = -(self.A * self.B - self.B ** 2 - self.B ** 3)
        Q = (a ** 2 - 3 * b) / 9
        R = (2 * a ** 3 - 9 * a * b + 27 * c) / 54
        M = R ** 2 - Q ** 3
        if M > 0:
            S = np.cbrt(-R + np.sqrt(M))
            Tt = np.cbrt(-R - np.sqrt(M))
            Z = S + Tt - a / 3
        else:
            theta = np.arccos(R / np.sqrt(Q ** 3))
            x1 = -2 * np.sqrt(Q) * np.cos(theta / 3) - a / 3
            x2 = -2 * np.sqrt(Q) * np.cos((theta + 2 * np.pi) / 3) - a / 3
            x3 = -2 * np.sqrt(Q) * np.cos((theta - 2 * np.pi) / 3) - a / 3
            solutions = np.array([x1, x2, x3])
            solutions = solutions[solutions > 0.0]
            if self.phase == "vapour":
                Z = np.amax(solutions)
            elif self.phase == "liquid":
                Z = np.amin(solutions)
            else:
                raise NotImplementedError
        return Z

    def calculate_rho(self, T, P):
        self.set_conditions(T, P)
        Z = self.polynomial_roots()
        return P / Z / boltzmann / T

    def calculate_mu_ex(self, T, P):
        Pref = self.get_Pref(T, P)
        rho = self.calculate_rho(T, P)
        Z = P / rho / boltzmann / T
        mu_ex_dimless = (
            Z - 1 - np.log(Z - self.B)
            - (self.A / (np.sqrt(8) * self.B))
            * np.log((Z + (1 + np.sqrt(2)) * self.B) / (Z + (1 - np.sqrt(2)) * self.B))
        )
        mu_ex_dimless += np.log(P / Pref)
        mu_ex = mu_ex_dimless * boltzmann * T
        return mu_ex, Pref
