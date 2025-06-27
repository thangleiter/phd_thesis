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

WAV = 825
INC = 0.
POL = 0.

INCLUDE_SI = True

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


def setup_bare_structure(barrier_thickness=90, epoxy_thickness=None, verbose=False):
    stack = [
        materials.index(air),
        materials.index(gaas),
        materials.index(algaas),
        materials.index(gaas),
        materials.index(gaas),
        materials.index(algaas),
        materials.index(gaas),
        materials.index(epoxy)
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
        # 1
    ]
    if epoxy_thickness is not None:
        stack.append(materials.index(si))
        thickness.insert(-1, epoxy_thickness)

    return pm.Structure(materials, stack, thickness, verbose=verbose)


def setup_tbg_structure(barrier_thickness=90, epoxy_thickness=None, verbose=False):
    bare_structure = setup_bare_structure(barrier_thickness, epoxy_thickness, verbose)
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
    return pm.Structure(materials, tbg_stack, tbg_thickness, verbose=verbose)


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
        source_interface = get_source_interface(structure)
        As, rs, ts, Rs, Ts = pm.absorption(structure, wavelength, incidence, polarization)
        if verbose:
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
            axs[1].set_yticks([0, 100, 200, 300])
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
print('Bare stack:\n -> 10/90/20/90/10')
bare_structure = setup_bare_structure(barrier_thickness=90, epoxy_thickness=None, verbose=False)
A, r, t, R, T = analyze_absorptance(bare_structure)
# %%%% Sweep epoxy thickness
thickness = np.linspace(0, 500, 1001)
Rs = np.empty_like(thickness)
for i, t in enumerate(thickness):
    structure = setup_bare_structure(barrier_thickness=90, epoxy_thickness=t, verbose=False)
    Rs[i] = pm.coefficient(structure, WAV, INC, POL)[2]

fig, ax = plt.subplots(layout='constrained')
ax.plot(thickness, Rs)
ax.scatter([0], [R])
ax.set_xlabel('Epoxy thickness (nm)')
ax.set_ylabel('$R$')
# %%% Topgate optical
tg_stack = bare_structure.layer_type.copy()
tg_stack.insert(1, materials.index(au_thick))
tg_stack.insert(2, materials.index(ti))

tg_thickness = bare_structure.thickness.copy()
tg_thickness.insert(1, 150)
tg_thickness.insert(2, 7)

print('Optical top gate:\n -> 150/7/10/90/20/90/10')
tg_structure = pm.Structure(materials, tg_stack, tg_thickness, verbose=False)
A, r, t, R, T = analyze_absorptance(tg_structure)
# %%% Bottomgate optical
bg_stack = bare_structure.layer_type.copy()
bg_stack.insert(-1, materials.index(au_thick))
bg_stack.insert(-2, materials.index(ti))

bg_thickness = bare_structure.thickness.copy()
bg_thickness.insert(-1, 150)
bg_thickness.insert(-2, 7)

print('Optical bot gate:\n -> 10/90/20/90/10/7/150')
bg_structure = pm.Structure(materials, bg_stack, bg_thickness, verbose=False)
A, r, t, R, T = analyze_absorptance(bg_structure)
# %%% Topgate ebeam
tg_stack = bare_structure.layer_type.copy()
tg_stack.insert(1, materials.index(au_thin))
tg_stack.insert(2, materials.index(ti))

tg_thickness = bare_structure.thickness.copy()
tg_thickness.insert(1, 7)
tg_thickness.insert(2, 2)

print('Ebeam top gate:\n -> 7/2/10/90/20/90/10')
tg_structure = pm.Structure(materials, tg_stack, tg_thickness, verbose=False)
A, r, t, R, T = analyze_absorptance(tg_structure)
# %%% Bottomgate ebeam
bg_stack = bare_structure.layer_type.copy()
bg_stack.insert(-1, materials.index(au_med))
bg_stack.insert(-2, materials.index(ti))

bg_thickness = bare_structure.thickness.copy()
bg_thickness.insert(-1, 25)
bg_thickness.insert(-2, 5)

print('Ebeam bot gate:\n -> 10/90/20/90/10/5/25')
bg_structure = pm.Structure(materials, bg_stack, bg_thickness, verbose=False)
A, r, t, R, T = analyze_absorptance(bg_structure)
# %%% Top and Bottom ebeam
print('Ebeam top and bot gate:\n -> 7/2/10/90/20/90/10/5/25')
tbg_structure = setup_tbg_structure(barrier_thickness=90, verbose=False)
A, r, t, R, T = analyze_absorptance(tbg_structure)
# %%%% optimize barrier thickness


def objective_function(barrier_thickness, wavelength=WAV, incidence=INC, polarization=POL):
    structure = setup_tbg_structure(barrier_thickness[0])
    source_interface = get_source_interface(structure)
    A, *_ = pm.absorption(structure, wavelength, incidence, polarization)
    return 1 - sum(A[source_interface-1:source_interface+1]).real


budget = 1000
best, convergence = pm.differential_evolution(objective_function, budget,
                                              X_min=np.array([25]), X_max=np.array([200]))
best = round(best.item())

print(f'Ebeam top and bot gate (opt @ {WAV} nm):\n -> 7/2/10/{best}/20/{best}/10/5/25')
tbg_structure_opt = setup_tbg_structure(barrier_thickness=best, verbose=False)
A_opt, *_ = analyze_absorptance(tbg_structure_opt)
# %%%%% Plot wavelengths
PLOT_R = False

source_interface = get_source_interface(tbg_structure_opt)
wavelengths = np.arange(775, 876)
As, rs, ts, Rs, Ts = analyze_absorptance(tbg_structure, wavelengths, verbose=False)
As = np.sum(As[:, source_interface-1:source_interface+1], axis=1)
As_opt, rs_opt, ts_opt, Rs_opt, Ts_opt = analyze_absorptance(tbg_structure_opt, wavelengths,
                                                             verbose=False)
As_opt = np.sum(As_opt[:, source_interface-1:source_interface+1], axis=1)

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained')

    ax.semilogy(wavelengths, As)
    ax.semilogy(wavelengths, As_opt)
    ax.set_ylabel('$A$')
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

    fig.savefig(SAVE_PATH / 'tmm_wavelengths.pdf')

# %%%% field
depth = None
structure = pm.Structure(materials, tbg_structure.layer_type[:depth],
                         tbg_structure.thickness[:depth], verbose=False)
structure_opt = pm.Structure(materials, tbg_structure_opt.layer_type[:depth],
                             tbg_structure_opt.thickness[:depth], verbose=False)

beam = pm.Beam(wavelength=WAV, incidence=INC, polarization=POL, horizontal_waist=2*624)
window = pm.Window(width=7.5*WAV, beam_relative_position=0.5, horizontal_pixel_size=50,
                   vertical_pixel_size=1)

E = pm.field(structure, beam, window)
E_opt = pm.field(structure_opt, beam, window)

# %%%%% Plotit
xlim = (-window.nx * window.px // 2, window.nx * window.px // 2)
fig = plot_field([structure, structure_opt], [E, E_opt], window, xlim, au_med)
fig.savefig(SAVE_PATH / 'tmm_field.pdf')

# %%%% Dipole emitter
depth = None
# Window needs to be wide enough for fields to have attenuated before leaking into other side (PBC)
window = pm.Window(width=50*WAV, beam_relative_position=0.5, horizontal_pixel_size=5,
                   vertical_pixel_size=1)

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
