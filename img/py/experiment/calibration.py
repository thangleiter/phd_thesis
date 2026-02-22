# %% Imports
import pathlib
import sys
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
from qutil import itertools
from mjolnir.calibration import PowerCalibration

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import MARGINSTYLE, MARGINWIDTH, PATH, init  # noqa

DATA_PATH = PATH.parent / 'data/calibration'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/experiment'
SAVE_PATH.mkdir(exist_ok=True)

init(MARGINSTYLE, backend := 'pgf')
# %% Load
with gzip.open(str(DATA_PATH / 'power_calibration_795,0nm_2025-01-27_10-18-31.p'), 'rb') as f:
    data = pickle.load(f)

angles = np.fromiter(data.keys(), np.float64)
powers = np.fromiter(itertools.chain.from_iterable((v.values() for v in data.values())),
                     np.float64).reshape(angles.size, -1)

calib = PowerCalibration(795, None, angles, powers.mean(-1), powers.std(-1))
# %%% Plot
fig, ax = plt.subplots(layout='constrained', figsize=(MARGINWIDTH, 1.3))

ax.errorbar(angles, powers.mean(-1)*1e6, powers.std(-1)*1e6, fmt='.', markersize=2.5)
ax.plot(x := np.linspace(*angles[[0, -1]], 1001), calib(x)*1e6)

ax.set_xlim(75)
ax.grid()
ax.set_yscale('log')
match backend:
    case 'pgf':
        ax.set_xlabel(r'Rotator angle (\unit{\degree})')
        ax.set_ylabel(r'$P$ (\unit{\micro\watt})')
    case _:
        ax.set_xlabel(r'Rotator angle ($\degree$)')
        ax.set_ylabel('$P$ (μW)')

fig.savefig(SAVE_PATH / 'power_calibration.pdf')
