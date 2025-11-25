import sys
import numpy as np
import torch
torch.set_num_threads(6)
from ase import Atoms
from ase.io import read, write
from ase.data import vdw_radii
from time import time
from gcmc import AI_GCMC
from unit import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = 'cuda'

# Preferably run on GPUs
from sevenn.calculator import SevenNetCalculator
model = SevenNetCalculator(model='checkpoint_best.pth', device=device)

atoms_frame = read('POSCAR_dobpdc')
#atoms_frame.set_pbc((True, True, True))
# C and O were renamed to Co and Os to differentiate them from framework atoms during training
atoms_ads = read('co2.xyz')
vdw_radii = vdw_radii.copy()
# Mg radius is set to 1.0 A
vdw_radii[12] = 1.0

T = 298 * kelvin
Pressures = [0.01 * bar, 0.1 * bar, 0.2 * bar, 0.4 * bar, 0.6 * bar, 0.8 * bar, 1.0 * bar, 1.2 * bar]

from utilities import PREOS
eos = PREOS.from_name('carbondioxide')

for P in Pressures:
    fugacity = eos.calculate_fugacity(T,P)
    gcmc = AI_GCMC(model, atoms_frame, atoms_ads, T, P, fugacity, device, vdw_radii)
    gcmc.run(600000)

