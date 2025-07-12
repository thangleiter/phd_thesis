# %% Imports
import pathlib
import sys
from typing import Literal

import PyMoosh as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyMoosh import green
from mpl_toolkits.axes_grid1 import ImageGrid
from qutil import functools, itertools
from qutil.plotting.colors import (RWTH_COLORS, RWTH_COLORS_50, make_diverging_colormap,
                                   make_sequential_colormap)

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, TEXTWIDTH, PATH, init)  # noqa

FILE_PATH = pathlib.Path(__file__).relative_to(pathlib.Path(__file__).parents[3])
DATA_PATH = PATH.parent / 'data/tmm'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/experiment'
SAVE_PATH.mkdir(exist_ok=True)
with np.errstate(divide='ignore', invalid='ignore'):
    SEQUENTIAL_CMAP = make_sequential_colormap('magenta', endpoint='blackwhite').reversed()
    DIVERGING_CMAP = make_diverging_colormap(('magenta', 'green'), endpoint='white')

# Lengths are in units of nm
WAV = 825
w_0 = 624
INC = 0.
POL = 0

INCLUDE_SI = True

init(MAINSTYLE, backend := 'pgf')
# %% Functions


def eps_to_nk(eps):
    n = np.sqrt(.5*(np.abs(eps) + eps.real))
    κ = np.sqrt(.5*(np.abs(eps) - eps.real))
    return n, κ


def nk_to_eps(n, κ):
    return (n + 1j*κ)**2


def get_index(structure, material):
    return structure.layer_type.index(materials.index(material))


def get_source_interface(structure):
    return np.diff(structure.layer_type).tolist().index(0) + 1


def setup_bare_structure(barrier_thickness=90., air_thickness=1., epoxy_thickness=None,
                         verbose=False):
    stack = [
        materials.index(air),
        materials.index(gaas),
        materials.index(algaas),
        materials.index(gaas),
        materials.index(gaas),
        materials.index(algaas),
        materials.index(gaas),
        materials.index(epoxy)
    ]
    thickness = [
        air_thickness,
        10,
        barrier_thickness,
        10,
        10,
        barrier_thickness,
        10,
        1,  # irrelevant
    ]
    if epoxy_thickness is not None:
        stack.append(materials.index(si))
        thickness.insert(-1, epoxy_thickness)

    return pm.Structure(materials, stack, thickness, verbose=verbose)


def setup_tg_structure(which: Literal['optical', 'ebeam'] = 'ebeam', barrier_thickness=90.,
                       air_thickness=1., epoxy_thickness=None, verbose=False):
    bare_structure = setup_bare_structure(barrier_thickness, air_thickness, epoxy_thickness,
                                          verbose)

    tg_stack = bare_structure.layer_type.copy()
    tg_stack.insert(1, materials.index(au_thin if which == 'ebeam' else au_thick))
    tg_stack.insert(2, materials.index(ti))

    tg_thickness = bare_structure.thickness.copy()
    tg_thickness.insert(1, 7 if which == 'ebeam' else 150)
    tg_thickness.insert(2, 2 if which == 'ebeam' else 7)
    return pm.Structure(materials, tg_stack, tg_thickness, verbose=False)


def setup_bg_structure(which: Literal['optical', 'ebeam'] = 'ebeam', barrier_thickness=90.,
                       air_thickness=1., epoxy_thickness=None, verbose=False):
    bare_structure = setup_bare_structure(barrier_thickness, air_thickness, epoxy_thickness,
                                          verbose)

    offset = int(epoxy_thickness is not None)
    bg_stack = bare_structure.layer_type.copy()
    bg_stack.insert(-1 - offset, materials.index(au_med if which == 'ebeam' else au_thick))
    bg_stack.insert(-2 - offset, materials.index(ti))

    bg_thickness = bare_structure.thickness.copy()
    bg_thickness.insert(-1 - offset, 25 if which == 'ebeam' else 150)
    bg_thickness.insert(-2 - offset, 5 if which == 'ebeam' else 7)
    return pm.Structure(materials, bg_stack, bg_thickness, verbose=False)


def setup_tgbg_structure(barrier_thickness=90., air_thickness=1., epoxy_thickness=None,
                         verbose=False):
    bare_structure = setup_bare_structure(barrier_thickness, air_thickness, epoxy_thickness,
                                          verbose)

    offset = int(epoxy_thickness is not None)
    tgbg_stack = bare_structure.layer_type.copy()
    tgbg_stack.insert(1, materials.index(au_thin))
    tgbg_stack.insert(2, materials.index(ti))
    tgbg_stack.insert(-1 - offset, materials.index(au_med))
    tgbg_stack.insert(-2 - offset, materials.index(ti))

    tgbg_thickness = bare_structure.thickness.copy()
    tgbg_thickness.insert(1, 7)
    tgbg_thickness.insert(2, 2)
    tgbg_thickness.insert(-1 - offset, 25)
    tgbg_thickness.insert(-2 - offset, 5)
    return pm.Structure(materials, tgbg_stack, tgbg_thickness, verbose=verbose)


def extract_absorptance(structure, As):
    source_interface = get_source_interface(structure)
    return sum(As[source_interface-1:source_interface+1]).real


def analyze_absorptance(structure, wavelength=WAV, incidence=INC, polarization=POL, verbose=True):
    if not np.isscalar(wavelength):
        if verbose:
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
        As, rs, ts, Rs, Ts = pm.absorption(structure, wavelength, incidence, polarization)
        if verbose:
            print(f'R = {Rs:.2g}')
            print(f'A = {extract_absorptance(structure, As):.2g}')
    return As, rs, ts, Rs, Ts


def analyze_dipole(idx=range(4), barrier_thickness=90., air_thickness=0.6*WAV,
                   horizontal_pixel_size=10, vertical_pixel_size=1):
    # Window needs to be wide enough for fields to have attenuated before leaking into other side
    # (PBC)
    window = pm.Window(width=50 * WAV, beam_relative_position=0.5,
                       horizontal_pixel_size=horizontal_pixel_size,
                       vertical_pixel_size=vertical_pixel_size)
    structures = []
    Ens = []
    if 0 in idx:
        structures.append(setup_bare_structure(barrier_thickness=barrier_thickness,
                                               air_thickness=air_thickness))
        Ens.append(green.green(structures[-1], window, WAV, get_source_interface(structures[-1])))
    if 1 in idx:
        structures.append(setup_tg_structure(barrier_thickness=barrier_thickness,
                                             air_thickness=air_thickness))
        Ens.append(green.green(structures[-1], window, WAV, get_source_interface(structures[-1])))
    if 2 in idx:
        structures.append(setup_bg_structure(barrier_thickness=barrier_thickness,
                                             air_thickness=air_thickness))
        Ens.append(green.green(structures[-1], window, WAV, get_source_interface(structures[-1])))
    if 3 in idx:
        structures.append(setup_tgbg_structure(barrier_thickness=barrier_thickness,
                                               air_thickness=air_thickness))
        Ens.append(green.green(structures[-1], window, WAV, get_source_interface(structures[-1])))
    return structures, Ens, window


def mask_data(structure, En, window, mat, xlim, func):
    yacc = np.add.accumulate(structure.thickness)
    yoff = yacc[0]
    ylim = (-yoff, yacc[get_index(structure, mat) - 1] - yoff)
    x = np.arange(-window.nx // 2 + window.nx % 2, window.nx // 2 + window.nx % 2) * window.px
    y = np.arange(0, sum(structure.thickness), window.py) - yoff
    xmask = (x >= xlim[0]) & (x <= xlim[1])
    ymask = (y >= ylim[0]) & (y <= ylim[1])
    masked = func(En)[ymask][:, xmask]
    return yacc, yoff, ylim, x, y, xmask, ymask, masked


def extract_params(Ens, func, mat, structures, window, xlim):
    yaccs, yoffs, ylims, xs, ys, xmasks, ymasks, maskeds = zip(
        *[mask_data(structure, En, window, mat, xlim, func)
          for structure, En in zip(structures, Ens)]
    )
    absmax = itertools.absmax(itertools.chain.from_iterable((masked.flat for masked in maskeds)))
    return absmax, maskeds, xmasks, xs, yaccs, ylims, ymasks, yoffs, ys


def plot_interfaces(axs, yacc, yoff, structure, show_emitter=False):
    for j, yy in enumerate(yacc):
        for ax in axs:
            if j < len(yacc) - 1 and structure.layer_type[j] == structure.layer_type[j+1]:
                # QW layer
                if show_emitter:
                    ax.scatter(0, yy - yoff, s=0.5, color=RWTH_COLORS['black'])
            else:
                ax.axhline(yy - yoff, color=RWTH_COLORS_50['black'], ls=':', lw=0.5, alpha=0.66)


def plot_field(fig, structures, Es, window, beam, xlim, mat):
    func = functools.chain(np.abs)
    absmax, maskeds, xmasks, xs, yaccs, ylims, ymasks, yoffs, ys = extract_params(
        Es, func, mat, structures, window, xlim
    )

    grid = ImageGrid(fig, 111, (2, 2), aspect=False, share_all=False, cbar_mode='single',
                     cbar_location='right', cbar_size='7.5%', axes_pad=0.1)

    for i, axs in enumerate(grid.axes_row):
        img = axs[1].pcolormesh(
            xs[i][xmasks[i]] / (beam.waist * 0.5),
            ys[i][ymasks[i]],
            maskeds[i] / absmax,
            cmap=SEQUENTIAL_CMAP,
            norm=mpl.colors.Normalize(vmin=0, vmax=1),
            rasterized=True
        )
        ln, = axs[0].plot(func(Es[i])[:, window.nx // 2] / absmax, ys[i],
                          color=RWTH_COLORS['magenta'])

        axs[1].set_aspect(1/95)
        axs[1].set_yticks([0, 100, 200, 300])
        axs[1].set_ylim(ylims[i])
        axs[0].set_aspect(1/150)
        axs[0].set_ylim(ylims[i])
        axs[0].grid(axis='x')
        axs[0].set_ylabel('$z$ (nm)')
        if i == 1:
            if backend == 'pgf':
                axs[1].set_xlabel(r'$\flatfrac{x}{w_0}$')
                axs[0].set_xlabel(label := r'$\lvert E_y\rvert$ (a.u.)')
            else:
                axs[1].set_xlabel('$x/w_0$')
                axs[0].set_xlabel(label := '$|E_y|$ (a.u.)')

        plot_interfaces(axs, yaccs[i], yoffs[i], structures[i])

        # line cut indicator
        axs[1].axvline(0, color=RWTH_COLORS_50['black'], ls='-.', lw=0.75, alpha=0.66)

    cb = grid.cbar_axes[0].colorbar(img)
    cb.set_label(label)

    grid.axes_column[0][0].invert_yaxis()
    grid.axes_column[0][1].invert_yaxis()
    grid.axes_column[0][0].invert_xaxis()
    grid.axes_column[0][0].set_xlim(left=1.05, right=-0.05)
    grid.axes_column[0][1].set_xticks([0, 0.5, 1])


def plot_dipole(fig, structures, Ens, window, xlim, mat):
    func = functools.chain(np.real)
    absmax, maskeds, xmasks, xs, yaccs, ylims, ymasks, yoffs, ys = extract_params(
        Ens, func, mat, structures, window, xlim
    )

    grid = ImageGrid(fig, 111, (len(Ens), 1), cbar_mode='single', cbar_location='top',
                     cbar_pad=0.05, cbar_size='10%', axes_pad=0.075)

    for i, ax in enumerate(grid):
        img = ax.pcolormesh(xs[i][xmasks[i]], ys[i][ymasks[i]], maskeds[i] / absmax,
                            cmap=DIVERGING_CMAP,
                            norm=mpl.colors.Normalize(vmin=-1, vmax=+1),
                            rasterized=True)

        ax.set_xlim(xlim)
        ax.set_ylim(ylims[i])
        ax.invert_yaxis()

        plot_interfaces([ax], yaccs[i], yoffs[i], structures[i])

    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel('$z$ (nm)')

    cb = grid.cbar_axes[0].colorbar(img)
    cb.set_label(r'$\mathrm{Re}\,E_y$ (a.u.)')


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
# https://www.epotek.com/docs/en/Related/Tech%20Tip%2018%20Understanding%20Optical%20Properties%20of%20Epoxy%20Applications.pdf
# https://www.epotek.com/docs/en/Datasheet/353ND.pdf
epoxy = pm.Material(nk_to_eps(1.55, 0))
materials.append(epoxy)
# %%% Si
si = pm.Material(['main', 'Si', 'Franta-10K'], specialType='RII')
materials.append(si)
# %% Simulate
materials = [air, gaas, algaas, au_thin, au_med, au_thick, ti, epoxy, si]

tbl = {'A': {}, 'R': {}}
# ebeam buried gate: 5/25
# ebeam etched gate: 2/7
# optical: 7/150
# epoxy: thickness of a few microns. Since the thickness varies significantly, let's assume that
#        there is no coherent backscattering at the epoxy/Si interface, which we implement by a
#        large imaginary part of the permittivity
# %%% Bare stack
print('Bare stack:\n --> 10/90/20/90/10')
structure = setup_bare_structure(barrier_thickness=90, epoxy_thickness=None, verbose=False)
A, r, t, R, T = analyze_absorptance(structure)
tbl['A']['Bare'] = extract_absorptance(structure, A)
tbl['R']['Bare'] = R
# %%% Epoxy-Si interface
R_epoxy_Si = pm.coefficient(pm.Structure([epoxy, si], [0, 1], [0, 0], verbose=False),
                            WAV, INC, POL)[2]
T_reverse = pm.coefficient(pm.Structure(materials, structure.layer_type[::-1],
                                        structure.thickness[::-1], verbose=False),
                           WAV, INC, POL)[3]
print('Bare stack -- incoherent reflection at Epoxy/Si interface:\n --> 10/90/20/90/10')
print(f'R = {R + T_reverse*R_epoxy_Si:.2g}')
# %%%% Sweep epoxy thickness
thickness = np.linspace(000, 1000, 1001)
Rs = np.empty_like(thickness)
for i, t in enumerate(thickness):
    structure = setup_bare_structure(barrier_thickness=90, epoxy_thickness=t, verbose=False)
    Rs[i] = pm.coefficient(structure, WAV, INC, POL)[2]

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained')

    ax.axhline(R, ls='--', color=RWTH_COLORS_50['blue'])
    ax.plot(thickness, Rs)
    ax.grid()
    ax.set_xlabel('Epoxy thickness (nm)')
    ax.set_ylabel(r'$\mathcal{R}$')

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ticks=[R], labels=[r'$\mathdefault{\mathcal{R}_{\infty}}$'])

    fig.savefig(SAVE_PATH / 'reflectance_epoxy.pdf')
# %%% Topgate optical
print('Optical top gate:\n --> 150/7/10/90/20/90/10')
tg_structure = setup_tg_structure('optical', barrier_thickness=90, verbose=False)
A, r, t, R, T = analyze_absorptance(tg_structure)
# %%% Bottomgate optical
print('Optical bot gate:\n --> 10/90/20/90/10/7/150')
bg_structure = setup_bg_structure('optical', barrier_thickness=90, verbose=False)
A, r, t, R, T = analyze_absorptance(bg_structure)
# %%% Topgate ebeam
print('Ebeam top gate:\n --> 7/2/10/90/20/90/10')
tg_structure = setup_tg_structure('ebeam', barrier_thickness=90, verbose=False)
A, r, t, R, T = analyze_absorptance(tg_structure)
tbl['A']['TG'] = extract_absorptance(tg_structure, A)
tbl['R']['TG'] = R
# %%% Bottomgate ebeam
print('Ebeam bot gate:\n --> 10/90/20/90/10/5/25')
bg_structure = setup_bg_structure('ebeam', barrier_thickness=90, verbose=False)
A, r, t, R, T = analyze_absorptance(bg_structure)
tbl['A']['BG'] = extract_absorptance(bg_structure, A)
tbl['R']['BG'] = R
# %%% Top and Bottom ebeam
print('Ebeam top and bot gate:\n --> 7/2/10/90/20/90/10/5/25')
tgbg_structure = setup_tgbg_structure(barrier_thickness=90, verbose=False)
A, r, t, R, T = analyze_absorptance(tgbg_structure)
tbl['A']['TG+BG'] = extract_absorptance(tgbg_structure, A)
tbl['R']['TG+BG'] = R
# %%%% Save to latex
df = pd.DataFrame.from_dict(tbl)
with (DATA_PATH / 'absorptance_reflectance.tex').open('w') as file:
    file.write(f'% This table is automatically generated by {FILE_PATH}\n'.replace('\\', '/'))
    (df * 100).to_latex(
        file,
        header=[r'{{$\mathcal{{' + x + r'}}$ (\unit{{\percent}})}}' for x in df.columns],
        column_format='lSS',
        float_format="%.2f"
    )
# %%% Optimize barrier thickness


def objective_function(barrier_thickness, wavelength=WAV, incidence=INC, polarization=POL):
    structure = setup_tgbg_structure(barrier_thickness[0])
    source_interface = get_source_interface(structure)
    A, *_ = pm.absorption(structure, wavelength, incidence, polarization)
    return 1 - sum(A[source_interface-1:source_interface+1]).real


budget = 1000
best, convergence = pm.differential_evolution(objective_function, budget,
                                              X_min=np.array([25]), X_max=np.array([200]))
best = round(best.item())

print(f'Ebeam top and bot gate (opt @ {WAV} nm):\n -> 7/2/10/{best}/20/{best}/10/5/25')
tgbg_structure_opt = setup_tgbg_structure(barrier_thickness=best, verbose=False)
A_opt, *_ = analyze_absorptance(tgbg_structure_opt)
# %%% Plot absorptance as function of wavelengths
PLOT_R = False

source_interface = get_source_interface(tgbg_structure_opt)
wavelengths = np.arange(775, 876)
As, rs, ts, Rs, Ts = analyze_absorptance(tgbg_structure, wavelengths, verbose=False)
As = np.sum(As[:, source_interface-1:source_interface+1], axis=1)
As_opt, rs_opt, ts_opt, Rs_opt, Ts_opt = analyze_absorptance(tgbg_structure_opt, wavelengths,
                                                             verbose=False)
As_opt = np.sum(As_opt[:, source_interface-1:source_interface+1], axis=1)

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained')

    ax.semilogy(wavelengths, As)
    ax.semilogy(wavelengths, As_opt)
    ax.set_ylabel(r'$\mathcal{A}$')
    ax.set_xlabel(r'$\lambda$ (nm)')
    arr = mpl.patches.FancyArrowPatch((WAV, As[wavelengths == WAV].item()),
                                      (WAV, As_opt[wavelengths == WAV].item()),
                                      arrowstyle='->', mutation_scale=7.5, zorder=5)
    ax.add_patch(arr)
    ax.annotate(rf'$\times {(As_opt[wavelengths == WAV] / As[wavelengths == WAV]).item():.2g}$',
                (1, .5), xycoords=arr, ha='left', va='center')
    if PLOT_R:
        ax2 = ax.twinx()
        ax2.plot(wavelengths, Rs, ls='--')
        ax2.plot(wavelengths, Rs_opt, ls='--')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('$R$')
    else:
        ax.grid()

    fig.savefig(SAVE_PATH / 'tmm_absorptance.pdf')

# %%% Gaussian illumination
structure = setup_tgbg_structure(barrier_thickness=90, air_thickness=1)
structure_opt = setup_tgbg_structure(barrier_thickness=best, air_thickness=1)

beam = pm.Beam(wavelength=WAV, incidence=INC, polarization=POL, horizontal_waist=2*w_0)
window = pm.Window(width=8*w_0, beam_relative_position=0.5, horizontal_pixel_size=25,
                   vertical_pixel_size=1)

E = pm.field(structure, beam, window)
E_opt = pm.field(tgbg_structure_opt, beam, window)

# %%% Plotit
xlim = ((-window.nx // 2 + window.nx % 2) * window.px,
        (+window.nx // 2 + window.nx % 2) * window.px)

with mpl.style.context(MAINSTYLE, after_reset=True):
    fig = plt.figure(figsize=(TEXTWIDTH, 2.35))
    plot_field(fig, [structure, structure_opt], [E, E_opt], window, beam, xlim, au_med)
    fig.savefig(SAVE_PATH / 'tmm_field.pdf')

# %%% Dipole emitter
structures, Ens, window = analyze_dipole(idx=(1, 2), barrier_thickness=90)

# %%% Plot different gate stacks
with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig = plt.figure(figsize=(MARGINWIDTH, 5))
    plot_dipole(fig, structures, Ens, window, xlim=(-1000, 1000), mat=epoxy)
    fig.savefig(SAVE_PATH / 'tmm_green.pdf')

# %%% Dipole emitter optimized
structures_opt, Ens_opt, window = analyze_dipole(barrier_thickness=best)

# %%% Plot different gate stacks
with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig = plt.figure(figsize=(MARGINWIDTH, 5))
    plot_dipole(fig, structures_opt, Ens_opt, window, xlim=(-1000, 1000), mat=epoxy)
    fig.savefig(SAVE_PATH / 'tmm_green_opt.pdf')

# %%% Plot optimized
with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig = plt.figure(figsize=(MARGINWIDTH, 2))
    plot_dipole(fig, [structures[-1], structures_opt[-1]], [Ens[-1], Ens_opt[-1]], window,
                xlim=(-1000, 1000), mat=epoxy)
    fig.savefig(SAVE_PATH / 'tmm_green_opt_tgbg.pdf')
