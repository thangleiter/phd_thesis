import pathlib

import IPython
import matplotlib as mpl
import numpy as np

from qutil import const, functools, misc

TEXTWIDTH = 4.2134
MARGINWIDTH = 1.87831
TOTALWIDTH = TEXTWIDTH + MARGINWIDTH + 0.24414
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


def sliceprops(color, alpha=0.66, linestyle='-.', linewidth=0.75):
    return dict(
        color=color,
        alpha=alpha,
        linestyle=linestyle,
        linewidth=linewidth
    )


def _lambda2eV(lambda_):
    with misc.filter_warnings('ignore', RuntimeWarning):
        result = const.lambda2eV(lambda_)
    result[np.isinf(result)] = 1e16
    return result


def _eV2lambda(eV):
    with misc.filter_warnings('ignore', RuntimeWarning):
        result = const.eV2lambda(eV)
    result[np.isinf(result)] = 1e16
    return result


def secondary_axis(ax, unit: str = 'eV'):
    match unit:
        case 'nm':
            functions = (functools.scaled(1e+9)(_lambda2eV),
                         functools.scaled(1e-9)(_eV2lambda))
            secondary_unit = 'eV'
        case 'eV':
            functions = (functools.scaled(1e+9)(_eV2lambda),
                         functools.scaled(1e-9)(_lambda2eV))
            secondary_unit = 'nm'
        case _:
            return ax, ''

    return ax.secondary_xaxis('top', functions=functions), secondary_unit


def n_GaAs(T=0):
    # 800 nm
    # https://refractiveindex.info/?shelf=other&book=AlAs-GaAs&page=Papatryfonos-0
    n0 = 3.6520 + 1j*0.075663
    # https://doi.org/10.1063/1.114204, we assume they measured at 25C
    return n0 - 2.67e-4 * (25 + const.zero_Celsius - T)


def E_AlGaAs(x):
    # https://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/bandstr.html#Temperature
    return 1.519 + 1.155*x + 0.37*x**2


def effective_mass(lh=False):
    # masses from 10.1103/PhysRevB.29.7085
    # Although 10.1038/s41598-017-05139-w find a much larger electron mass in QWs
    # due to non-parabolicities: m_ep = 0.169
    m_ep = 0.0665
    m_hp = 0.34
    m_lp = 0.094
    if lh:
        return np.array([[m_ep, m_lp]]).T * const.m_e
    else:
        return np.array([[m_ep, m_hp]]).T * const.m_e


def reduced_mass(lh=False):
    return (np.multiply.reduce(effective_mass(lh)) / np.add.reduce(effective_mass(lh))).item()
