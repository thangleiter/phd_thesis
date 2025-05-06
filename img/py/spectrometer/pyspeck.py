# %%
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qopt.noise import fast_colored_noise
from qutil import itertools
from qutil.plotting.colors import RWTH_COLORS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (  # noqa
    apply_sketch_style, markerprops, MARGINWIDTH, PATH, TEXTWIDTH, MARGIN_STYLE, MAIN_STYLE
)

mpl.use('pgf')
# %%


def rect(x, T=1):
    arg = .5 * x * T
    return T * np.sinc(arg / np.pi)


def hann(x, T=1):
    arg = .5 * x * T
    with np.errstate(invalid='ignore', divide='ignore'):
        y = .5 * np.atleast_1d(rect(x, T)) / (1 - (arg / np.pi)**2)
    y[(arg == np.pi) | (arg == -np.pi)] = T / 4
    return y


# %%
alpha = 0.5
T = 1

x = np.linspace(-5.5, 5.5, 1001)*2*np.pi/T
xn = np.arange(-5, 6)*2*np.pi/T
xm = np.arange(-4.5, 5.5)*2*np.pi/T

colors = mpl.color_sequences['rwth'][1:]

with plt.style.context(MARGIN_STYLE, after_reset=True):
    for window in (rect, hann):
        fig, ax = plt.subplots(figsize=(MARGINWIDTH, 1.4))

        ax.plot(x, window(x, T), color=colors[0])
        for xnn in xn:
            ax.plot([xnn], window(xnn, T), **markerprops(colors[1], marker='o', markersize=5))
        for xmm in xm:
            ax.plot([xmm], window(xmm, T), **markerprops(colors[2], marker='D', markersize=4))

        ax.set_xlim(-12.5*np.pi, 12.5*np.pi)
        ax.set_xticks([-10 * np.pi / T, 10 * np.pi / T])
        ax.set_xticklabels([r'$\flatfrac{-10\pi}{T}$', r'$\flatfrac{+10\pi}{T}$'])

        ax.set_xlabel(r'$\omega_n$')
        ax.set_ylabel(r'$\hat{w}_n$')

        apply_sketch_style(ax)

        if window is rect:
            ax.set_yticks([T])
            ax.set_yticklabels([r'$T$'])
            ax.set_ylim(-0.3, 1.2*T)
            ax.xaxis.set_tick_params(pad=5, length=7.5)
            ax.yaxis.set_tick_params(length=7.5)
            ax.xaxis.set_label_coords(12.5*np.pi/T, .2*T, transform=ax.transData)
            ax.yaxis.set_label_coords(2.25*np.pi/T, 1.1*T, transform=ax.transData)
        elif window is hann:
            ax.set_yticks([T/2])
            ax.set_yticklabels([r'$\flatfrac{T}{2}$'])
            ax.set_ylim(-0.25/4, .6*T)
            ax.xaxis.set_tick_params(length=7.5)
            ax.yaxis.set_tick_params(length=7.5)
            ax.xaxis.set_label_coords(12.5*np.pi/T, .1*T, transform=ax.transData)
            ax.yaxis.set_label_coords(2.25*np.pi/T, .55*T, transform=ax.transData)

        fig.tight_layout()
        fig.savefig(PATH / f'pdf/spectrometer/{window.__name__}.pdf', backend='pgf')
        plt.close(fig)

# %%
L = 300
N = 100
K = 50
M = int(2*L/N - 1)

np.random.seed(0)
noise = fast_colored_noise(lambda f: 1/f, 1, L, ())

with plt.style.context(MAIN_STYLE, after_reset=True):
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
    fig.savefig(PATH / 'pdf/spectrometer/welch.pdf', backend='pgf')
    plt.close(fig)
