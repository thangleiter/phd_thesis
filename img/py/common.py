import pathlib

import IPython
import matplotlib as mpl

from qutil import const

TEXTWIDTH = 4.2134
MARGINWIDTH = 1.87831
TOTALWIDTH = TEXTWIDTH + TEXTWIDTH + 0.24414
PATH = pathlib.Path(__file__).parents[1]
MAINSTYLE = PATH / 'py/main.mplstyle'
MARGINSTYLE = PATH / 'py/margin.mplstyle'


def init(style, backend):
    if (ipy := IPython.get_ipython()) is not None:
        ipy.run_line_magic('matplotlib', backend)
    else:
        mpl.use('qtagg' if backend == 'qt' else backend)

    mpl.rcdefaults()
    mpl.style.use(style)


def apply_sketch_style(ax):
    ax.spines.left.set_position(("data", 0))
    ax.spines.bottom.set_position(("data", 0))
    ax.spines.left.set_zorder(0.5)
    ax.spines.bottom.set_zorder(0.5)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    ax.tick_params(direction='inout')
    ax.set_xlabel(ax.get_xlabel(), rotation='horizontal')
    ax.set_ylabel(ax.get_ylabel(), rotation='horizontal')


def markerprops(color, marker='o', markersize=5, markeredgealpha=1.0, markerfacealpha=0.5,
                markeredgewidth=None):
    return dict(
        ls='',
        marker=marker,
        markersize=markersize,
        markeredgewidth=markeredgewidth or mpl.rcParams['lines.markeredgewidth'],
        markeredgecolor=mpl.colors.to_rgba(color, markeredgealpha),
        markerfacecolor=mpl.colors.to_rgba(color, markerfacealpha)
    )


def n_GaAs(T=0):
    # 800 nm
    # https://refractiveindex.info/?shelf=other&book=AlAs-GaAs&page=Papatryfonos-0
    n0 = 3.6520 + 1j*0.075663
    # https://doi.org/10.1063/1.114204, we assume they measured at 25C
    return n0 - 2.67e-4 * (25 + const.zero_Celsius - T)
