import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from qopt.noise import fast_colored_noise
from qutil import const, itertools
from qutil.plotting import RWTH_COLORS

from common import apply_sketch_style

TEXTWIDTH = 4.2134
MARGINWIDTH = 1.87831
TOTALWIDTH = TEXTWIDTH + TEXTWIDTH + 0.24414
PATH = pathlib.Path(__file__).parents[1]

mpl.use('pgf')
# %%


def rect(x, T=1):
    arg = .5 * x * T / np.pi
    return T * np.sinc(arg)


def hann(x, T=1):
    arg = .5 * x * T / np.pi
    with np.errstate(invalid='ignore', divide='ignore'):
        y = np.atleast_1d(T * np.sinc(arg) / (2 * (1 - arg) * (1 + arg)))
    y[(arg == 1) | (arg == -1)] = T / 4
    return y


# %%
x = np.linspace(-11, 11, 1001)*np.pi
xn = np.linspace(-11, 11, 11)*np.pi

alpha = 0.5
T = 1

with plt.style.context('./margin.mplstyle', after_reset=True):
    for window in (rect, hann):
        fig, ax = plt.subplots(figsize=(MARGINWIDTH, 1.4))

        ax.plot(x, window(x, T), color=RWTH_COLORS['magenta'])
        for xnn in xn:
            ax.vlines(xnn, *itertools.minmax(0, window(xnn, T).item()),
                      color=RWTH_COLORS['green'] + (alpha,))
            ax.plot([xnn], window(xnn, T), 'o',
                    markeredgecolor=RWTH_COLORS['green'],
                    markerfacecolor=RWTH_COLORS['green'] + (alpha,))

        ax.set_xlim(-12.5*np.pi, 12.5*np.pi)
        ax.set_xticks([-10 * np.pi / T, 10 * np.pi / T])
        ax.set_xticklabels([r'$\flatfrac{-10\pi}{T}$', r'$\flatfrac{+10\pi}{T}$'])

        ax.set_xlabel(r'$\omega_n$')
        ax.set_ylabel(r'$\hat{w}_n$')

        apply_sketch_style(ax)

        if window is rect:
            ax.set_yticks([T])
            ax.set_yticklabels([r'$T$'])
            ax.set_ylim(-0.25, 1.2*T)
            ax.xaxis.set_tick_params(pad=7.5)
            ax.xaxis.set_label_coords(12.5*np.pi, .2*T, transform=ax.transData)
            ax.yaxis.set_label_coords(2.25*np.pi, 1.1*T, transform=ax.transData)
        elif window is hann:
            ax.set_yticks([T/2])
            ax.set_yticklabels([r'$\flatfrac{T}{2}$'])
            ax.set_ylim(-0.25/4, .6*T)
            ax.xaxis.set_label_coords(12.5*np.pi, .115*T, transform=ax.transData)
            ax.yaxis.set_label_coords(2.25*np.pi, .55*T, transform=ax.transData)

        fig.tight_layout()
        fig.savefig(PATH / f'pdf/{window.__name__}.pdf', backend='pgf')
        plt.close(fig)

# %%
L = 300
N = 100
K = 50
M = int(2*L/N - 1)

np.random.seed(0)
noise = fast_colored_noise(lambda f: 1/f, 1, L, ())

with plt.style.context('./main.mplstyle', after_reset=True):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2))

    n = np.arange(N)
    ax.hlines(0, 0, L, color='tab:gray')
    for i in range(M):
        ax.fill_between(n + i*K, y0 := -np.sin(np.pi*n/N)**2, y1 := np.sin(np.pi*n/N)**2,
                        color='tab:gray', alpha=0.33 if i == 1 else 0.2)
        ax.plot(n + i*K, y0, color='tab:gray', alpha=1 if i == 1 else 0.5)
        ax.plot(n + i*K, y1, color='tab:gray', alpha=1 if i == 1 else 0.5)

    ax2 = ax.twinx()
    ax2.plot(n := np.arange(0, N - K + 1), noise[n], color=RWTH_COLORS['magenta'] + (0.33,))
    ax2.plot(n := np.arange(N - K, 2*N - K), noise[n], color=RWTH_COLORS['magenta'])
    ax2.plot(n := np.arange(2*N - K - 1, L), noise[n], color=RWTH_COLORS['magenta'] + (0.33,))

    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax.set_xticks([i*(N-K) for i in range(M + 2)])
    ax.set_xticklabels([r'$0$', r'$N-K$'] + ['$...$']*(M - 2) + [r'$L - N + K$', r'$L$'])
    ax.set_yticks([])
    ax2.set_yticks([])
    ax.margins(y=0.15)
    ax.tick_params(direction='inout')

    fig.tight_layout()
    fig.savefig(PATH / 'pdf/welch.pdf', backend='pgf')
    plt.close(fig)
