import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import PyMoosh as pm
from PyMoosh import green

wavelength = 800
angle_inc = 0
pol = 0.
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
au_thin = pm.Material(['main', 'Au', 'Rosenblatt-11nm'], specialType="RII")
au_med = pm.Material(['main', 'Au', 'Rosenblatt-21nm'], specialType="RII")
au_thick = pm.Material(['main', 'Au', 'Yakubovsky-117nm'], specialType='RII')
materials.append(au_thin)
materials.append(au_med)
materials.append(au_thick)
# %%% Titanium
ti = pm.Material(['main', 'Ti', 'Palm'], specialType="RII")
materials.append(ti)
# %%% Epoxy
# Thickness of a few microns
# https://www.epotek.com/docs/en/Related/Tech%20Tip%2018%20Understanding%20Optical%20Properties%20of%20Epoxy%20Applications.pdf
# https://www.epotek.com/docs/en/Datasheet/353ND.pdf
epoxy = pm.Material(1.55**2)
materials.append(epoxy)
# %%% Si
si = pm.Material(['main', 'Si', 'Franta-10K'], specialType='RII')
materials.append(si)
# %% Functions


def get_index(structure, material):
    return structure.layer_type.index(materials.index(material))


# %% Simulate
materials = [air, gaas, algaas, au_thin, au_med, au_thick, ti, epoxy, si]

# ebeam buried gate: 5/25
# ebeam etched gate: 2/7
# optical: 7/150
# %%% Topgate
stack = [
    materials.index(air),
    materials.index(au_thin),
    materials.index(ti),
    materials.index(gaas),
    materials.index(algaas),
    materials.index(gaas),
    materials.index(gaas),
    materials.index(algaas),
    materials.index(gaas),
    materials.index(epoxy),
    materials.index(si),
]
thicknesses = [
    100,  # irrelevant
    7,
    2,
    10,
    90,
    10,
    10,
    90,
    10,
    1e3,
    2e6,
]

structure = pm.Structure(materials, stack, thicknesses)
r, t, R, T = pm.coefficient(structure, wavelength, angle_inc, pol)
print(f'R = {R:.2g}')

# %%% Bottom
stack = [
    materials.index(air),
    materials.index(gaas),
    materials.index(algaas),
    materials.index(gaas),
    materials.index(gaas),
    materials.index(algaas),
    materials.index(gaas),
    materials.index(ti),
    materials.index(au_med),
    materials.index(epoxy),
    materials.index(si),
]
thicknesses = [
    100,  # irrelevant
    10,
    90,
    10,
    10,
    90,
    10,
    5,
    25,
    1e3,
    2e6,
]

structure = pm.Structure(materials, stack, thicknesses)
r, t, R, T = pm.coefficient(structure, wavelength, angle_inc, pol)
print(f'R = {R:.2g}')

# %%% Top and Bottom
stack = [
    materials.index(air),
    materials.index(au_thin),
    materials.index(ti),
    materials.index(gaas),
    materials.index(algaas),
    materials.index(gaas),
    materials.index(gaas),
    materials.index(algaas),
    materials.index(gaas),
    materials.index(ti),
    materials.index(au_thick),
    materials.index(epoxy),
    materials.index(si),
]
thicknesses = [
    1*wavelength,  # Air, irrelevant
    7,  # Au
    2,  # Ti
    10,  # GaAs
    90,  # AlGaAs
    10,  # QW
    10,  # QW
    90,  # AlGaAs
    10,  # GaAs
    5,  # Ti
    25,  # Au
    1e3,  # Epoxy
    1,  # Si
    # 1*wavelength,  # Air, irrelevant
]

structure = pm.Structure(materials, stack, thicknesses)
r, t, R, T = pm.coefficient(structure, wavelength, angle_inc, pol)
print(f'R = {R:.2g}')

# %%%% angular
incidence, r, t, R, T = pm.angular(structure, wavelength, pol, 0, 45, 46)
incidence, r_p, t_p, R_p, T_p = pm.angular(structure, wavelength, 1, 0, 45, 46)

fig, ax = plt.subplots()
ax.plot(incidence, R, label="TE polarisation")
ax.plot(incidence, R_p, label="TM polarisation")
ax.set_ylabel('Reflectance')
ax.legend()
ax.grid()

# %%%% spectrum
incidence, r, t, R2, T = pm.spectrum(structure, 0, pol, 750, 900, 151)
incidence, r_p, t_p, R_p2, T_p = pm.spectrum(structure, 0, pol, 750, 900, 151)

fig, ax = plt.subplots()
ax.plot(incidence, R2, label="TE polarisation")
ax.plot(incidence, R_p2, label="TM polarisation")
ax.set_ylabel('Reflectance')
ax.legend()
ax.grid()

# %%%% field
depth = None
structure = pm.Structure(materials, stack[:depth], thicknesses[:depth])
beam = pm.Beam(wavelength, 0, pol, 2*624)
window = pm.Window(7.5*wavelength, 0.5, 50, 1)

E = pm.field(structure, beam, window)

# %%%%% Plot
fig, axs = plt.subplot_mosaic([['linecut', 'image']], width_ratios=(1, 4), layout='constrained',
                              sharey=True)

ylim = (ys := np.add.accumulate(structure.thickness))[[0, get_index(structure, au_thick)]]
img = axs['image'].pcolormesh(
    x := np.arange(-window.nx // 2, window.nx // 2) * window.px,
    y := np.arange(0, sum(structure.thickness), window.py),
    abs(E),
    norm=mpl.colors.Normalize(vmax=abs(E)[(y >= ylim[0]) & (y <= ylim[1])].max())
)
cb = fig.colorbar(img)

axs['linecut'].plot(abs(E)[:, window.nx // 2], y)
axs['linecut'].invert_xaxis()
axs['linecut'].grid(axis='x')

for i, y in enumerate(ys):
    if i < len(ys) - 1 and stack[i] == stack[i+1]:
        # Skip double layers
        continue
    axs['linecut'].axhline(y, color='tab:gray', ls='--')
    axs['image'].axhline(y, color='tab:gray', ls='--')

axs['image'].set_ylim(ylim)

# %%%% Dipole emitter
depth = None
structure = pm.Structure(materials, stack[:depth], thicknesses[:depth])
window = pm.Window(50*wavelength, 0.5, 5, 1)
source_interface = np.diff(stack).tolist().index(0) + 1

En = green.green(structure, window, wavelength, source_interface)

# %%%%% Plot
fig, axs = plt.subplot_mosaic([['linecut', 'image']], width_ratios=(1, 4), layout='constrained',
                              sharey=True)
ylim = (ys := np.add.accumulate(structure.thickness))[[0, get_index(structure, au_thick)]]
img = axs['image'].pcolormesh(
    x := np.arange(-window.nx // 2, window.nx // 2) * window.px,
    y := np.arange(0, sum(structure.thickness), window.py),
    np.real(En),
    norm=mpl.colors.Normalize(vmin=0, vmax=np.real(En)[(y >= ylim[0]) & (y <= ylim[1])].max())
)
cb = fig.colorbar(img)

axs['linecut'].plot(np.real(En)[:, window.nx // 2], y)
axs['linecut'].invert_xaxis()
axs['linecut'].grid(axis='x')

for i, y in enumerate(ys):
    if i < len(ys) - 1 and stack[i] == stack[i+1]:
        # Skip double layers
        continue
    axs['linecut'].axhline(y, color='tab:gray', ls='--')
    axs['image'].axhline(y, color='tab:gray', ls='--')

# axs['image'].set_ylim(ylim)
