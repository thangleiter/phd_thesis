# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_25, RWTH_COLORS_50, RWTH_COLORS_75

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MARGINSTYLE, PATH, init

ORIG_DATA_PATH = pathlib.Path(
    r'\\janeway\User AG Bluhm\Common\GaAs\Hangleiter\characterization\vibrations'
)
DATA_PATH = PATH.parent / 'data/vibrations'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

init(MARGINSTYLE, backend := 'pgf')
# %% Functions


def H(ω, ω0, m=1, γ=1, σ=0):
    # The damping ratio ζ = b/2√km where k = mω_0² and b = 2γm
    s = σ + 1j*ω
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1 / abs(1 + s**2 / (2*γ*s + ω0**2))


def T(ω, ω0, M=1, γ=1, σ=0):
    # The damping ratio ζ = b/2√kM
    with np.errstate(invalid='ignore', divide='ignore'):
        k = M*ω0**2
        b = 2*γ*M
        s = σ + 1j*ω
        return abs((b*s + k) / (M*s**2 + b*s + k))


# %% Air spring sketch
m = 1
ω0 = 1
ω = np.geomspace(1e-1, 1e1, 1001)
γs = [1/200, 1/2]
linestyles = ['-', '--']
colors = [RWTH_COLORS_25, RWTH_COLORS_50]

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained')

    ax.axhline(1, color=RWTH_COLORS_75['black'])
    ax.axvline(np.sqrt(2)*ω0, color=RWTH_COLORS_50['black'], ls=':')

    for γ, color, ls in zip(γs, colors, linestyles):
        ax.loglog(ω, t := abs(H(ω, ω0, m, γ)), color=RWTH_COLORS['black'], ls=ls)
        ax.fill_between(ω[t <= 1], 1, t[t <= 1], color=color['green'])
        ax.fill_between(ω[t >= 1], 1, t[t >= 1], color=color['magenta'])

    match backend:
        case 'pgf':
            ax.set_xlabel(r'$\flatfrac{\omega}{\omega_0}$')
            ax.set_ylabel(r'$\abs{H(i\omega)}$')
        case 'qt':
            ax.set_xlabel(r'$\omega/\omega_0$')
            ax.set_ylabel(r'$|H(i\omega)|$')

    ax.grid()
    ax.set_yticks([1e-2, 1e0, 1e2])
    # ax.set_ylim(1e-2, 1e2)

    fig.savefig(SAVE_PATH / 'spring_tf.pdf')
