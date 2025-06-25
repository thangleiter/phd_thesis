# %% Imports
import pathlib
import sys

import PyMoosh as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from PyMoosh import green
from qutil import functools, itertools
from qutil.plotting.colors import (
    make_sequential_colormap, make_diverging_colormap, RWTH_COLORS, RWTH_COLORS_50
)

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, TEXTWIDTH, PATH, init)  # noqa

SAVE_PATH = PATH / 'pdf/experiment'
SAVE_PATH.mkdir(exist_ok=True)
with np.errstate(divide='ignore', invalid='ignore'):
    SEQUENTIAL_CMAP = make_sequential_colormap('magenta', endpoint='blackwhite')
    DIVERGING_CMAP = make_diverging_colormap(['magenta', 'green'], endpoint='white')

WAV = 800
INC = 0.
POL = 0.

init(MAINSTYLE, backend := 'pgf')
# %% Functions


def eps_to_nk(eps):
    κ = np.sqrt(.5*(np.sqrt(eps.real**2 + eps.imag**2) - eps.real))
    n = .5*eps.imag/κ
    return n, κ


def nk_to_eps(n, κ):
    return n**2 - κ**2 + 2j*n*κ


def get_index(structure, material):
    return structure.layer_type.index(materials.index(material))


def get_source_interface(structure):
    return np.diff(structure.layer_type).tolist().index(0) + 1


def setup_bare_structure(barrier_thickness=90):
    stack = [
        materials.index(air),
        materials.index(gaas),
        materials.index(algaas),
        materials.index(gaas),
        materials.index(gaas),
        materials.index(algaas),
        materials.index(gaas),
        materials.index(epoxy),
        # materials.index(si),
        # materials.index(air)
    ]
    thickness = [
        1,  # irrelevant
        10,
        barrier_thickness,
        10,
        10,
        barrier_thickness,
        10,
        1,  # irrelevant
        # 2e6,
        # 1
    ]
    return pm.Structure(materials, stack, thickness, verbose=False)


def setup_tbg_structure(barrier_thickness):
    bare_structure = setup_bare_structure(barrier_thickness)
    tbg_stack = bare_structure.layer_type.copy()
    tbg_stack.insert(1, materials.index(au_thin))
    tbg_stack.insert(2, materials.index(ti))
    tbg_stack.insert(-1, materials.index(au_med))
    tbg_stack.insert(-2, materials.index(ti))

    tbg_thickness = bare_structure.thickness.copy()
    tbg_thickness.insert(1, 7)
    tbg_thickness.insert(2, 2)
    tbg_thickness.insert(-1, 25)
    tbg_thickness.insert(-2, 5)
    return pm.Structure(materials, tbg_stack, tbg_thickness, verbose=False)


def analyze_absorptance(structure, wavelength=WAV, incidence=INC, polarization=POL):
    if not np.isscalar(wavelength):
        print('Sweeping wavelength.')
        rs, ts = np.empty((2, np.size(wavelength)), dtype=complex)
        Rs, Ts = np.empty((2, np.size(wavelength)))
        As = np.empty((np.size(wavelength), len(structure.layer_type)))
        for i, wav in enumerate(wavelength):
            A, r, t, R, T = pm.absorption(structure, wav, incidence, polarization)
            rs[i] = r
            ts[i] = t
            As[i] = np.real(A)
            Rs[i] = np.real(R)
            Ts[i] = np.real(T)
    else:
        source_interface = get_source_interface(structure)
        As, rs, ts, Rs, Ts = pm.absorption(structure, wavelength, incidence, polarization)
        print(f'R = {Rs:.2g}')
        print(f'A = {sum(As[source_interface-1:source_interface+1]).real:.2g}')
    return As, rs, ts, Rs, Ts


def mask_data(structure, En, window, mat, xlim, func):
    yacc = np.add.accumulate(structure.thickness)
    yoff = yacc[0]
    ylim = (-yoff, yacc[get_index(structure, mat)] - yoff)
    x = np.arange(-window.nx // 2, window.nx // 2) * window.px
    y = np.arange(0, sum(structure.thickness), window.py) - yoff
    xmask = (x >= xlim[0]) & (x <= xlim[1])
    ymask = (y >= ylim[0]) & (y <= ylim[1])
    masked = func(En)[ymask][:, xmask]
    return yacc, yoff, ylim, x, y, xmask, ymask, masked


def plot_interfaces(axs, yacc, yoff, structure):
    for j, yy in enumerate(yacc):
        if j < len(yacc) - 1 and structure.layer_type[j] == structure.layer_type[j+1]:
            # Skip double layers
            continue
        for ax in axs:
            ax.axhline(yy - yoff, color=RWTH_COLORS_50['black'], ls=':', lw=0.5, alpha=0.66)


def plot_field(structures, Es, window, xlim, mat):
    func = functools.chain(np.abs)
    yaccs, yoffs, ylims, xs, ys, xmasks, ymasks, maskeds = zip(
        *[mask_data(structure, En, window, mat, xlim, func)
          for structure, En in zip(structures, Es)]
    )

    absmax = itertools.absmax(itertools.chain.from_iterable((masked.flat for masked in maskeds)))

    with mpl.style.context(MAINSTYLE, after_reset=True):
        fig = plt.figure()
        grid = ImageGrid(fig, 111, (2, 2), aspect=False, share_all=False, cbar_mode='single',
                         cbar_location='right', cbar_size='7.5%', axes_pad=0.1)

        for i, axs in enumerate(grid.axes_row):
            img = axs[1].pcolormesh(
                xs[i][xmasks[i]],
                ys[i][ymasks[i]],
                maskeds[i],
                cmap=SEQUENTIAL_CMAP,
                norm=mpl.colors.Normalize(vmin=0, vmax=absmax),
                rasterized=True
            )
            ln, = axs[0].plot(func(Es[i])[:, window.nx // 2], ys[i], color=RWTH_COLORS['magenta'])

            axs[1].set_aspect(8)
            axs[1].set_ylim(ylims[i])
            axs[0].set_aspect(1/50)
            axs[0].set_ylim(ylims[i])
            axs[0].grid(axis='x')
            axs[0].set_ylabel('$z$ (nm)')
            if i == 1:
                axs[1].set_xlabel('$x$ (nm)')
                axs[0].set_xlabel(
                    label := r'$\lvert E_y\rvert$ (a.u.)' if backend == 'pgf' else '$|E_y|$ (a.u.)'
                )

            plot_interfaces(axs, yaccs[i], yoffs[i], structures[i])

            # line cut indicator
            axs[1].axvline(0, color=RWTH_COLORS_50['black'], ls='-.', lw=0.75, alpha=0.66)

        cb = grid.cbar_axes[0].colorbar(img)
        cb.set_label(label)

        grid.axes_column[0][0].invert_yaxis()
        grid.axes_column[0][1].invert_yaxis()
        grid.axes_column[0][0].invert_xaxis()
        grid.axes_column[0][0].set_xlim(right=0)
        grid.axes_column[0][1].set_xticks([0, 1, 2, 3])

    return fig


def plot_dipole(structures, Ens, window, xlim, mat):
    func = functools.chain(np.real)

    yaccs, yoffs, ylims, xs, ys, xmasks, ymasks, maskeds = zip(
        *[mask_data(structure, En, window, mat, xlim, func)
          for structure, En in zip(structures, Ens)]
    )

    absmax = itertools.absmax(itertools.chain.from_iterable((masked.flat for masked in maskeds)))

    with mpl.style.context(MARGINSTYLE, after_reset=True):
        fig = plt.figure()
        grid = ImageGrid(fig, 111, (2, 1), cbar_mode='single', cbar_location='top', cbar_pad=0.05,
                         cbar_size='10%', axes_pad=0.075)

        for i, ax in enumerate(grid):
            img = ax.pcolormesh(xs[i][xmasks[i]], ys[i][ymasks[i]], maskeds[i],
                                cmap=DIVERGING_CMAP,
                                norm=mpl.colors.Normalize(vmin=-absmax, vmax=+absmax),
                                rasterized=True)

            ax.set_xlim(xlim)
            ax.set_ylim(ylims[i])
            ax.invert_yaxis()
            ax.set_ylabel('$z$ (nm)')
            if ax is grid[1]:
                ax.set_xlabel('$x$ (nm)')

            plot_interfaces([ax], yaccs[i], yoffs[i], structures[i])

        cb = grid.cbar_axes[0].colorbar(img)
        cb.set_label(r'$\mathrm{Re}\,E_y$ (a.u.)')
        return fig


# %% Materials
air = pm.Material(1.)

materials = []
materials.append(air)
# %%% GaAs
gaas = pm.Material(['main', 'GaAs', 'Papatryfonos'], specialType="RII")
materials.append(gaas)
# %%% AlGaAs
algaas = pm.Material(['other', 'AlAs-GaAs', 'Papatryfonos-34.2'], specialType="RII")
materials.append(algaas)
# %%% Gold
au_thin = pm.Material(['main', 'Au', 'Yakubovsky-6nm'], specialType="RII")
au_med = pm.Material(['main', 'Au', 'Yakubovsky-25nm'], specialType="RII")
au_thick = pm.Material(['main', 'Au', 'Yakubovsky-117nm'], specialType='RII')
materials.append(au_thin)
materials.append(au_med)
materials.append(au_thick)
# %%% Titanium
ti = pm.Material(['main', 'Ti', 'Palm'], specialType="RII")
materials.append(ti)
# %%% Epoxy
# Thickness of a few microns. Since the thickness varies significantly, let's assume that there is
# no coherent backscattering at the epoxy/Si interface, which we implement by a large imaginary
# part of the permittivity
# https://www.epotek.com/docs/en/Related/Tech%20Tip%2018%20Understanding%20Optical%20Properties%20of%20Epoxy%20Applications.pdf
# https://www.epotek.com/docs/en/Datasheet/353ND.pdf
epoxy = pm.Material(nk_to_eps(1.55, 0))
materials.append(epoxy)
# %%% Si
si = pm.Material(['main', 'Si', 'Franta-10K'], specialType='RII')
materials.append(si)
# %% Simulate
materials = [air, gaas, algaas, au_thin, au_med, au_thick, ti, epoxy, si]

# ebeam buried gate: 5/25
# ebeam etched gate: 2/7
# optical: 7/150
# %%% Bare stack
bare_structure = setup_bare_structure(barrier_thickness=90)
A, r, t, R, T = analyze_absorptance(bare_structure)
# %%% Topgate optical
tg_stack = bare_structure.layer_type.copy()
tg_stack.insert(1, materials.index(au_thick))
tg_stack.insert(2, materials.index(ti))

tg_thickness = bare_structure.thickness.copy()
tg_thickness.insert(1, 150)
tg_thickness.insert(2, 7)

tg_structure = pm.Structure(materials, tg_stack, tg_thickness)
A, r, t, R, T = analyze_absorptance(tg_structure)
# %%% Bottomgate optical
bg_stack = bare_structure.layer_type.copy()
bg_stack.insert(-1, materials.index(au_thick))
bg_stack.insert(-2, materials.index(ti))

bg_thickness = bare_structure.thickness.copy()
bg_thickness.insert(-1, 150)
bg_thickness.insert(-2, 7)

bg_structure = pm.Structure(materials, bg_stack, bg_thickness)
A, r, t, R, T = analyze_absorptance(bg_structure)
# %%% Topgate ebeam
tg_stack = bare_structure.layer_type.copy()
tg_stack.insert(1, materials.index(au_thin))
tg_stack.insert(2, materials.index(ti))

tg_thickness = bare_structure.thickness.copy()
tg_thickness.insert(1, 7)
tg_thickness.insert(2, 2)

tg_structure = pm.Structure(materials, tg_stack, tg_thickness)
A, r, t, R, T = analyze_absorptance(tg_structure)
# %%% Bottomgate ebeam
bg_stack = bare_structure.layer_type.copy()
bg_stack.insert(-1, materials.index(au_med))
bg_stack.insert(-2, materials.index(ti))

bg_thickness = bare_structure.thickness.copy()
bg_thickness.insert(-1, 25)
bg_thickness.insert(-2, 5)

bg_structure = pm.Structure(materials, bg_stack, bg_thickness)
A, r, t, R, T = analyze_absorptance(bg_structure)
# %%% Top and Bottom ebeam
tbg_structure = setup_tbg_structure(barrier_thickness=90)
A, r, t, R, T = analyze_absorptance(tbg_structure)
# %%%% optimize barrier thickness


def objective_function(barrier_thickness, wavelength=WAV, incidence=INC, polarization=POL):
    structure = setup_tbg_structure(barrier_thickness[0])
    source_interface = get_source_interface(structure)
    A, *_ = pm.absorption(structure, wavelength, incidence, polarization)
    return 1 - sum(A[source_interface-1:source_interface+1]).real


budget = 1000
best, convergence = pm.differential_evolution(objective_function, budget,
                                              X_min=np.array([50]), X_max=np.array([150]))
best = best.round()

tbg_structure_opt = setup_tbg_structure(barrier_thickness=best[0])
A_opt, *_ = analyze_absorptance(tbg_structure_opt)
# %%%%% Plot wavelengths
PLOT_R = False

source_interface = get_source_interface(tbg_structure_opt)
wavelengths = np.arange(750, 851)
As, rs, ts, Rs, Ts = analyze_absorptance(tbg_structure, wavelengths)
As = np.sum(As[:, source_interface-1:source_interface+1], axis=1)
As_opt, rs_opt, ts_opt, Rs_opt, Ts_opt = analyze_absorptance(tbg_structure_opt, wavelengths)
As_opt = np.sum(As_opt[:, source_interface-1:source_interface+1], axis=1)

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained')

    ax.semilogy(wavelengths, As)
    ax.semilogy(wavelengths, As_opt)
    ax.set_ylabel('$A$')
    ax.set_xlabel(r'$\lambda$ (nm)')
    arr = mpl.patches.FancyArrowPatch((800, As[wavelengths == 800].item()),
                                      (800, As_opt[wavelengths == 800].item()),
                                      arrowstyle='->', mutation_scale=10, zorder=5)
    ax.add_patch(arr)
    ax.annotate(rf'$\times {(As_opt[wavelengths == 800] / As[wavelengths == 800]).item():.2g}$',
                (.75, .5), xycoords=arr, ha='left', va='center')
    if PLOT_R:
        ax2 = ax.twinx()
        ax2.plot(wavelengths, Rs, ls='--')
        ax2.plot(wavelengths, Rs_opt, ls='--')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('$R$')
    else:
        ax.grid()

    fig.savefig(SAVE_PATH / 'tmm_wavelengths.pdf')

# %%%% field
depth = None
structure = pm.Structure(materials, tbg_structure.layer_type[:depth],
                         tbg_structure.thickness[:depth], verbose=False)
structure_opt = pm.Structure(materials, tbg_structure_opt.layer_type[:depth],
                             tbg_structure_opt.thickness[:depth], verbose=False)

beam = pm.Beam(WAV, INC, POL, 2*624)
window = pm.Window(7.5*WAV, 0.5, 50, 1)

E = pm.field(structure, beam, window)
E_opt = pm.field(structure_opt, beam, window)

# %%%%% Plotit
xlim = (-window.nx * window.px // 2, window.nx * window.px // 2)
fig = plot_field([structure, structure_opt], [E, E_opt], window, xlim, au_med)
fig.savefig(SAVE_PATH / 'tmm_field.pdf')

# %%%% Dipole emitter
depth = None
# Window needs to be wide enough for fields to have attenuated before leaking into other side (PBC)
window = pm.Window(50*WAV, 0.5, 5, 1)

thick = tbg_structure.thickness[:depth]
thick[0] = 0.6*WAV
structure = pm.Structure(materials, tbg_structure.layer_type[:depth], thick, verbose=False)

thick = tbg_structure_opt.thickness[:depth]
thick[0] = 0.6*WAV
structure_opt = pm.Structure(materials, tbg_structure_opt.layer_type[:depth], thick, verbose=False)

En = green.green(structure, window, WAV, get_source_interface(structure))
En_opt = green.green(structure_opt, window, WAV, get_source_interface(structure_opt))

# %%%%% Plot it
fig = plot_dipole([structure, structure_opt], [En, En_opt], window, xlim=(-1000, 1000), mat=au_med)
fig.savefig(SAVE_PATH / 'tmm_green.pdf')
