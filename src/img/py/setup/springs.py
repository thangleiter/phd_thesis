# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutil.plotting import changed_plotting_backend
from qutil.plotting.colors import (
    RWTH_COLORS, RWTH_COLORS_75, RWTH_COLORS_25
)

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import PATH, MARGINSTYLE, init

ORIG_DATA_PATH = pathlib.Path(
    r'\\janeway\User AG Bluhm\Common\GaAs\Hangleiter\characterization\vibrations'
)
DATA_PATH = PATH.parent / 'data/vibrations'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

init(MARGINSTYLE, backend := 'pgf')
# %% Functions


def H(ω, ω0=1, M=1, γ=1, σ=0):
    # M d²y + 2γM dy + Mω0² y = u
    # M d²y + b dy + k y = u
    # Simple LTI solution
    b = 2*γ*M
    k = M*ω0**2
    s = σ + 1j*ω
    return k / (M * s**2 + b * s + k)


def T(ω, ω0=1, M=1, γ=1):
    # https://www.fabreeka.com/wp-content/uploads/2025/04/Fabreeka%20Low%20Frequency%20Pneumatic-2019-EN.pdf
    # ζ is the damping ratio, b/2√kM
    Fn = ω0/(2*np.pi)
    Fd = ω/(2*np.pi)
    ζ = γ / ω0
    return np.sqrt(
        (1 + (2*Fd/Fn*ζ)**2)
        / ((1 - Fd**2/Fn**2)**2 + (2*Fd/Fn*ζ)**2)
    )


# %% Air spring sketch
gammainv = 10
ω0 = 1
M = 1

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained')

    ω = np.geomspace(1e-1, 1e1, 1001)
    ax.axhline(1, color=RWTH_COLORS_75['black'])
    ax.axvline(np.sqrt(2)*ω0, color=RWTH_COLORS_75['black'], ls=':')
    ax.loglog(ω, t := T(ω, ω0, M, 1/gammainv), color=RWTH_COLORS_75['black'], ls='--')
    ax.loglog(ω, h := np.abs(H(ω, ω0, M, 1/gammainv)), color=RWTH_COLORS['black'])
    ax.fill_between(ω[h <= 1], 1, h[h <= 1], color=RWTH_COLORS_25['green'])
    ax.fill_between(ω[h >= 1], 1, h[h >= 1], color=RWTH_COLORS_25['magenta'])
    ax.grid()
    ax.set_ylim(1e-2)

    match backend:
        case 'pgf':
            ax.set_xlabel(r'$\flatfrac{\omega}{\omega_0}$')
            ax.set_ylabel(r'$\abs{\flatfrac{H(i\omega)}{H(0)}}$')
            ax.annotate(rf'$\gamma=\flatfrac{{\omega_0}}{{{gammainv}}}$', (1.25e-1, 2.5e-2))
        case 'qt':
            ax.set_xlabel(r'$\omega/\omega_0$')
            ax.set_ylabel(r'$|H(i\omega)/H(0)|$')
            ax.annotate(rf'$\gamma=\omega_0/{gammainv}$', (1.25e-1, 2.5e-2))

    with changed_plotting_backend('pgf'):
        fig.savefig(SAVE_PATH / 'spring_tf.pdf')
