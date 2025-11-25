from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Structure, Element
import numpy as np
import os

# Utilities
def random_unit_vector():
    """
    Generate a random unit vector uniformly on a sphere.
    """
    phi = np.random.uniform(0, 2*np.pi)
    costheta = np.random.uniform(-1, 1)
    sintheta = np.sqrt(1 - costheta**2)
    return np.array([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta])

def wrap_to_cell(structure, cart):
    """
    Wrap a Cartesian coordinate back into the periodic simulation cell.
    """
    frac = structure.lattice.get_fractional_coords(cart)
    frac = frac % 1.0
    return structure.lattice.get_cartesian_coords(frac)

def _safe_radius(symbol, mode="vdw", default_radius=1.5):
    """
    Safely obtain an atomic radius.
    Priority: van der Waals radius → covalent radius → fallback default.
    """
    try:
        el = Element(symbol)
    except Exception:
        return default_radius

    if mode == "vdw":
        r = el.van_der_waals_radius
        if r is None:
            r = el.atomic_radius if el.atomic_radius is not None else default_radius
    elif mode == "covalent":
        r = el.atomic_radius if el.atomic_radius is not None else default_radius
    else:
        r = default_radius

    try:
        r = float(r)
    except Exception:
        r = default_radius
    return r

def min_pbc_dist_to_framework(structure, cart_point, framework_frac_coords):
    """
    Compute the minimum PBC (minimum image) distance from a point to all framework atoms.
    """
    p_frac = structure.lattice.get_fractional_coords(cart_point)
    diff = p_frac - framework_frac_coords
    diff -= np.round(diff)
    diff_cart = diff @ structure.lattice.matrix
    dists = np.linalg.norm(diff_cart, axis=1)
    return float(dists.min()), int(np.argmin(dists))

# Placement validity check
def is_valid_molecule_position_with_inflated_framework(
    structure,
    atom_positions_cart,
    framework_symbols,
    framework_frac_coords,
    inflated_radii,
    min_distance_between_molecules=2.5,
):
    """
    Validate that a CO₂ molecule does not overlap with:
    1) Inflated framework exclusion zones
    2) Previously added CO₂ molecules
    """
    # Framework avoidance
    for pos in atom_positions_cart:
        dmin, idx = min_pbc_dist_to_framework(structure, pos, framework_frac_coords)
        if dmin < inflated_radii[idx] - 1e-8:
            return False

    # CO₂–CO₂ minimum distance check
    for pos in atom_positions_cart:
        pos_wrapped = wrap_to_cell(structure, pos)
        neighbors = structure.get_sites_in_sphere(pos_wrapped, min_distance_between_molecules - 1e-6)
        if len(neighbors) > 0:
            return False

    return True

# CO₂ placement
def add_co2_molecules_by_count_inflated(
    structure,
    num_molecules,
    min_distance_between_molecules=2.5,
    max_attempts=60000,
    bond_length=1.16,
    radius_mode="vdw",
    radius_scale=1.4,
    radius_padding=0.6,
    default_radius=1.5,
):
    """
    Randomly place CO₂ molecules while respecting inflated framework radii
    and CO₂–CO₂ minimum spacing.
    """
    framework_symbols = np.array([str(site.specie) for site in structure.sites])
    framework_frac_coords = np.array([site.frac_coords for site in structure.sites], dtype=float)

    base_radii = np.array([
        _safe_radius(sym, mode=radius_mode, default_radius=default_radius)
        for sym in framework_symbols
    ], dtype=float)
    inflated_radii = base_radii * float(radius_scale) + float(radius_padding)

    added = 0
    attempts_left = int(max_attempts)

    while added < num_molecules and attempts_left > 0:
        attempts_left -= 1

        frac_center = np.random.rand(3)
        center_cart = structure.lattice.get_cartesian_coords(frac_center)
        axis = random_unit_vector()

        o1 = center_cart - bond_length * axis
        o2 = center_cart + bond_length * axis
        c  = center_cart

        c_wrapped  = wrap_to_cell(structure, c)
        o1_wrapped = wrap_to_cell(structure, o1)
        o2_wrapped = wrap_to_cell(structure, o2)
        atom_positions = [c_wrapped, o1_wrapped, o2_wrapped]

        if is_valid_molecule_position_with_inflated_framework(
            structure=structure,
            atom_positions_cart=atom_positions,
            framework_symbols=framework_symbols,
            framework_frac_coords=framework_frac_coords,
            inflated_radii=inflated_radii,
            min_distance_between_molecules=min_distance_between_molecules,
        ):
            structure.append("C", c_wrapped,  coords_are_cartesian=True)
            structure.append("O", o1_wrapped, coords_are_cartesian=True)
            structure.append("O", o2_wrapped, coords_are_cartesian=True)
            added += 1

    return structure

# Unique structure generation utilities
def _structure_signature_added_sites(structure: Structure, original_num_sites: int, frac_round: int = 3):
    """
    Create a signature for added atoms: (species, rounded fractional coords).
    Used to determine uniqueness of generated structures.
    """
    added = structure.sites[original_num_sites:]
    sig = []
    for site in added:
        f = site.frac_coords % 1.0
        f = np.round(f, frac_round)
        sig.append((str(site.specie), float(f[0]), float(f[1]), float(f[2])))
    sig.sort()
    return tuple(sig)

def reorder_structure_by_species(structure: Structure, order=("C", "Mg", "O", "H")):
    """
    Reorder atomic species according to a specified order.
    """
    ordered_sites = []
    for elem in order:
        ordered_sites.extend([site for site in structure.sites if str(site.specie) == elem])

    known = set(order)
    ordered_sites.extend([site for site in structure.sites if str(site.specie) not in known])

    return Structure.from_sites(ordered_sites)

def generate_unique_structures_for_counts_inflated(
    base_structure: Structure,
    molecule_counts,
    save_dir,
    n_unique_per_count=50,
    min_distance_between_molecules=2.0,
    bond_length=1.16,
    radius_mode="vdw",
    radius_scale=1.4,
    radius_padding=0.6,
    default_radius=1.5,
    frac_round=3,
    max_global_attempts=500000,
):
    """
    For each number of CO₂ molecules, generate a specified number of unique structures.
    Uniqueness is determined by fractional positions of added atoms.
    """
    os.makedirs(save_dir, exist_ok=True)

    for n in molecule_counts:
        print(f"\n=== Target: {n} CO2 → Need {n_unique_per_count} unique structures ===")

        out_dir = os.path.join(save_dir, f"{n}CO2")
        os.makedirs(out_dir, exist_ok=True)

        unique_sigs = set()
        made = 0
        attempts = 0
        original_num_sites = len(base_structure.sites)

        while made < n_unique_per_count and attempts < max_global_attempts:
            attempts += 1
            s = base_structure.copy()

            s = add_co2_molecules_by_count_inflated(
                s, n,
                min_distance_between_molecules=min_distance_between_molecules,
                max_attempts=60000,
                bond_length=bond_length,
                radius_mode=radius_mode,
                radius_scale=radius_scale,
                radius_padding=radius_padding,
                default_radius=default_radius,
            )

            if len(s.sites) - original_num_sites != 3 * n:
                continue

            sig = _structure_signature_added_sites(s, original_num_sites, frac_round=frac_round)
            if sig in unique_sigs:
                continue

            unique_sigs.add(sig)
            made += 1
            out = os.path.join(out_dir, f"POSCAR_{made:03d}")
            s = reorder_structure_by_species(s)
            Poscar(s).write_file(out)
            print(f"✔ [{n}] saved {made}/{n_unique_per_count}: {out}")

        if made < n_unique_per_count:
            print(f"⚠️ [{n}] Only {made}/{n_unique_per_count} unique structures generated "
                  f"(Attempts={attempts}, max_global_attempts={max_global_attempts})")
        else:
            print(f"✅ [{n}] Successfully generated {n_unique_per_count} unique structures "
                  f"(Total attempts={attempts})")



if __name__ == "__main__":
    # Example: Input POSCAR path
    input_poscar = "path/POSCAR_xx"
    mof_structure = Poscar.from_file(input_poscar).structure

    # Example: Output directory
    save_dir = "path/save_dir"

    # Example CO2 counts
    molecule_counts = [1, 2, 3, 5, 6, 7, 9, 12, 15]

    # Number of unique structures per count
    n_unique_per_count = 50

    generate_unique_structures_for_counts_inflated(
        base_structure=mof_structure,
        molecule_counts=molecule_counts,
        save_dir=save_dir,
        n_unique_per_count=n_unique_per_count,
        min_distance_between_molecules=2.0,
        bond_length=1.16,
        radius_mode="vdw",
        radius_scale=2.3,
        radius_padding=0.6,
        default_radius=1.5,
        frac_round=3,
        max_global_attempts=500000,
    )
