"""
Created on Tue Nov 19 16:23:43 2024

@author: Flash
"""
import addcopyfighandler
import numpy as np
import tifffile
from qutil import plotting
import matplotlib.pyplot as plt
import pathlib
from PIL import Image

path = pathlib.Path('C:/Users/Flash/Pictures/Camera Roll')

PIXEL_SIZE = 5.2e-6
NOMINAL_O_SIZE = np.array([4.2175e-6, 4.7455e-6])
# %% White light
file = '24-11-19_white_light_on_top_left.tif'
img = tifffile.imread(path / file)
img = img / img.max()
x = np.arange(img.shape[1])
y = np.arange(img.shape[0])
# %%%
fig = plt.figure(figsize=(8, 7), layout='constrained')
axes_dict = fig.subplot_mosaic(
    [
         ['.', 'xslice', '.'],
         ['yslice', 'image', 'cbar']
     ],
    width_ratios=[1, 3, .15],
    height_ratios=[1, 3]
)

xslice, = axes_dict['xslice'].plot(x, img[0, :], color='k')
yslice, = axes_dict['yslice'].plot(img[:, 0], y, color='k')
im = axes_dict['image'].pcolormesh(x, y, img, cmap='inferno')
cbar = fig.colorbar(im, cax=axes_dict['cbar'])

axes_dict['yslice'].invert_xaxis()
axes_dict['image'].sharey(axes_dict['yslice'])
axes_dict['image'].sharex(axes_dict['xslice'])
axes_dict['image'].label_outer()
# %%%
clicker = plotting.CoordClicker(fig, plot_click=True)
# %%%
clicker.disconnect()
xslice.set_ydata(img[round(clicker.ys[0]), :])
yslice.set_xdata(img[:, round(clicker.xs[0])])
axes_dict['xslice'].relim()
axes_dict['yslice'].relim()
axes_dict['xslice'].autoscale_view()
axes_dict['yslice'].autoscale_view()

# %% Magnification
xclicker = plotting.CoordClicker(fig, plot_click=True)
# %%%
xclicker.disconnect()
yclicker = plotting.CoordClicker(fig, plot_click=True)
# %%%
yclicker.disconnect()

magnification = np.abs(np.concatenate((np.diff(xclicker.xs), np.diff(yclicker.ys))))
magnification *= PIXEL_SIZE / NOMINAL_O_SIZE
# %% Detection
file = '24-11-19_detection_spot_on_optical_gate_less_power.tif'
img = tifffile.imread(path / file)
img = Image.fromarray(img, mode='RGBA')
img_array = np.array(img.convert('F'))
img_array /= img_array.max()
# %%%
fig = plt.figure(figsize=(8, 7), layout='constrained')
axes_dict = fig.subplot_mosaic(
    [
         ['.', 'xslice',],
         ['yslice', 'image',]
     ],
    width_ratios=[1, 3,],
    height_ratios=[1, 3]
)

xslice, = axes_dict['xslice'].plot(x, img_array[0, :], color='k')
yslice, = axes_dict['yslice'].plot(img_array[:, 0], y, color='k')
im = axes_dict['image'].imshow(img, origin='lower', aspect='auto',
                               extent=(0, img.size[0], 0, img.size[1]))

axes_dict['yslice'].invert_xaxis()
axes_dict['image'].sharey(axes_dict['yslice'])
axes_dict['image'].sharex(axes_dict['xslice'])
axes_dict['image'].label_outer()

fig.suptitle('Detection Path')
# %%%
clicker = plotting.CoordClicker(fig, plot_click=True)
# %%%
clicker.disconnect()
xslice.set_ydata(img_array[round(clicker.ys[0]), :])
yslice.set_xdata(img_array[:, round(clicker.xs[0])])
axes_dict['xslice'].relim()
axes_dict['yslice'].relim()
axes_dict['xslice'].autoscale_view()
axes_dict['yslice'].autoscale_view()

# %% Excitation
file = '24-11-19_excitation_spot_on_optical_gate_less_power.tif'
img = tifffile.imread(path / file)
img = Image.fromarray(img, mode='RGBA')
img_array = np.array(img.convert('F'))
img_array /= img_array.max()
# %%%
fig = plt.figure(figsize=(8, 7), layout='constrained')
axes_dict = fig.subplot_mosaic(
    [
         ['.', 'xslice',],
         ['yslice', 'image',]
     ],
    width_ratios=[1, 3,],
    height_ratios=[1, 3]
)

xslice, = axes_dict['xslice'].plot(x, img_array[0, :], color='k')
yslice, = axes_dict['yslice'].plot(img_array[:, 0], y, color='k')
im = axes_dict['image'].imshow(img, origin='lower', aspect='auto',
                               extent=(0, img.size[0], 0, img.size[1]))

axes_dict['xslice'].grid()
axes_dict['yslice'].grid()
axes_dict['yslice'].invert_xaxis()
axes_dict['image'].sharey(axes_dict['yslice'])
axes_dict['image'].sharex(axes_dict['xslice'])
axes_dict['image'].label_outer()

fig.suptitle('Excitation Path')
# %%%
clicker = plotting.CoordClicker(fig, plot_click=True)
# %%%
clicker.disconnect()
xslice.set_ydata(img_array[round(clicker.ys[0]), :])
yslice.set_xdata(img_array[:, round(clicker.xs[0])])
axes_dict['xslice'].relim()
axes_dict['yslice'].relim()
axes_dict['xslice'].autoscale_view()
axes_dict['yslice'].autoscale_view()


"""
Created on Mon Mar 11 18:18:56 2024

@author: Flash
"""
import addcopyfighandler  # noqa
import pathlib
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import optimize, ndimage, special
from qutil import functools, itertools

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.dpi'] = 144

d = pathlib.Path.home() / 'Pictures/Camera Roll'
imgs = tifffile.imread([d / 'excitation_spot_cloverleaf.tif', d / 'detection_spot_cloverleaf.tif'])
imgs = tifffile.imread([d / 'excitation_spot_no_halfwave.tif',
                        d / 'detection_spot_no_halfwave.tif'])
imgs = tifffile.imread([d / 'excitation_spot_test.tif', d / 'detection_spot_test.tif'])
imgs = tifffile.imread([d / 'excitation_spot.tif', d / 'detection_spot.tif'])


def gaussian(x, a, μ, σ):
    return a * np.exp(-(x - μ)**2 / 2 / σ**2)


@jax.jit
def gaussian_2d(xy, a, σ_x, σ_y):
    x, y = xy
    return a * jnp.exp(-x**2 / (2 * σ_x**2) - y**2 / (2 * σ_y**2))


@jax.jit
def rotated_gaussian_2d(xy, a, μ_x, μ_y, σ_x, σ_y, θ):
    R = jnp.array([[jnp.cos(θ), -jnp.sin(θ)],
                   [jnp.sin(θ), +jnp.cos(θ)]])
    xpyp = jnp.tensordot(R, xy - jnp.array([[[μ_x, μ_y]]]).T, axes=[1, 0])
    return gaussian_2d(xpyp, a, σ_x, σ_y).ravel()


def TEM_ℓm_field(ℓ, m, xy, μ_x, μ_y, w, w_0, E_0):
    """Hermite-Gaussian modes."""
    x, y = xy
    x = x - μ_x
    y = y - μ_y
    return (
        E_0
        * w_0 / w
        * special.eval_hermite(ℓ, np.sqrt(2) / w * x)
        * special.eval_hermite(m, np.sqrt(2) / w * y)
        * np.exp(-(x**2 + y**2) / w**2)
    ).ravel()


def TEM_ℓm_intensity(ℓs, ms, xy, μ_x, μ_y, w, w_0, *E_0s):
    res = np.zeros(xy[0].size)
    for ℓ, m, E_0 in zip(ℓs, ms, E_0s):
        res += TEM_ℓm_field(ℓ, m, xy, μ_x, μ_y, w, w_0, E_0)
    return res**2


# def TEM_pℓ(p, ℓ, rho, phi, μ_x, μ_y, w, phi_0, b, I_0):
#     """Laguerre-Gaussian modes.

#     Incorrect with cosine factor (from Wikipedia)
#     """
#     return b + (
#         I_0
#         * rho**ℓ
#         * special.eval_genlaguerre(p, ℓ, rho)**2
#         * np.cos(ℓ * (phi - phi_0))**2
#         * np.exp(-rho)
#     ).ravel()


# def TEM_02(xy, μ_x, μ_y, w, phi_0, b, I_0):
#     rho, phi = to_polar(xy, μ_x, μ_y, w)
#     return TEM_pℓ(0, 2, rho, phi, μ_x, μ_y, w, phi_0, b, I_0)


# def TEM_pℓ_sum(ps, ℓs, xy, μ_x, μ_y, w, phi_0, b, *I_0s):
#     rho, phi = to_polar(xy, μ_x, μ_y, w)
#     res = np.zeros(rho.size)
#     for p, ℓ, I_0 in zip(ps, ℓs, I_0s):
#         res += TEM_pℓ(p, ℓ, rho, phi, μ_x, μ_y, w, phi_0, b, I_0)
#     return res


def to_polar(xy, μ_x, μ_y, w):
    xiy = xy[0] - μ_x + 1j*(xy[1] - μ_y)
    r = np.abs(xiy)
    rho = 2 * r**2 / w**2
    phi = np.angle(xiy)
    return rho, phi


jac = jax.jit(jax.jacobian(rotated_gaussian_2d, [1, 2, 3, 4, 5, 6]))


def R2(obs, fit):
    RSS = ((obs - fit)**2).sum()
    TSS = ((obs - obs.mean())**2).sum()
    return 1 - RSS / TSS


def eccentricity(σ_x, σ_y):
    return np.sqrt(1 - min(σ_x, σ_y)**2 / max(σ_x, σ_y)**2)


# %% 1d
PIXEL_SIZE = 5.2
magnification = np.array([27.45, 30])

px_row = np.arange(0, imgs.shape[-2], dtype=np.float_)
px_col = np.arange(0, imgs.shape[-1], dtype=np.float_)

fig, ax = plt.subplots(len(imgs), 2, sharex='col', sharey='row', layout='constrained')
for i, img in enumerate(imgs):
    # row, col = map(round, ndimage.center_of_mass(img))
    row, col = np.unravel_index(img.argmax(), img.shape)
    popt_row, pcov_row = optimize.curve_fit(
        gaussian, px_row, img[:, col],
        p0=[img.max(), row, 50],
        bounds=list(zip(*[(0, np.inf), (0, px_row.max()), (0, px_row.max())]))
    )
    popt_col, pcov_col = optimize.curve_fit(
        gaussian, px_col, img[row, :],
        p0=[img.max(), col, 50],
        bounds=list(zip(*[(0, np.inf), (0, px_col.max()), (0, px_col.max())]))
    )

    R_sq_col = R2(img[row, :], gaussian(px_col, *popt_col))
    R_sq_row = R2(img[:, col], gaussian(px_row, *popt_row))

    e = eccentricity(popt_col[2], popt_row[2])
    w_x_px = 4*popt_col[2]
    w_y_px = 4*popt_row[2]
    w_x_um = w_x_px * PIXEL_SIZE / magnification[1]
    w_y_um = w_y_px * PIXEL_SIZE / magnification[0]

    ax[i, 0].plot(px_col, img[row, :], px_col, gaussian(px_col, *popt_col))
    ax[i, 1].plot(px_row, img[:, col], px_row, gaussian(px_row, *popt_row))
    ax[i, 0].set_title(rf'$1-R^2$ = {1-R_sq_col:.1e}, $w_x$ = {w_x_px:.1f} px = {w_x_um:.1f} μm')
    ax[i, 1].set_title(rf'$1-R^2$ = {1-R_sq_row:.1e}, $w_y$ = {w_y_px:.1f} px = {w_y_um:.1f} μm')
    ax[i, 0].set_xlim(popt_col[1] - 5 * popt_col[2], popt_col[1] + 5 * popt_col[2])
    ax[i, 1].set_xlim(popt_row[1] - 5 * popt_row[2], popt_row[1] + 5 * popt_row[2])
    txt = ax[i, 1].text(0.9, 0.75, rf'$e = {e:.2f}$', transform=ax[i, 1].transAxes)
    txt.set_ha('right')

ax[1, 0].set_xlabel('Horizontal px')
ax[1, 1].set_xlabel('Vertical px')
ax[0, 0].set_ylabel('Excitation')
ax[1, 0].set_ylabel('Detection')

# %% 2d TEM_00
# rows, cols = zip(*[map(int, ndimage.center_of_mass(img)) for img in imgs])
rows, cols = np.unravel_index([img.argmax() for img in imgs], imgs.shape[1:])
diff = np.diff([rows, cols])
dist = np.linalg.norm(diff, ord=2).astype(np.int_)
center_row, center_col = ((diff / 2).astype(np.int_) + np.array([rows[0:1], cols[0:1]])).squeeze()

# at least n pixels
width = max(15, 5 * dist)

px_row = jnp.arange(center_row - width, center_row + width)
px_col = jnp.arange(center_col - width, center_col + width)
# px_row = jnp.arange(0, imgs.shape[1])
# px_col = jnp.arange(0, imgs.shape[2])
ij = jnp.ix_(px_row, px_col)
xy = jnp.array(jnp.meshgrid(px_col, px_row))

fig = plt.figure(layout='constrained')
grid = ImageGrid(fig, 111, (len(imgs), 3), axes_pad=0.1, share_all=True, cbar_mode='edge')
for i, (img, arm) in enumerate(zip(imgs, ['Excitation', 'Detection'])):
    # row, col = ndimage.center_of_mass(img)
    row, col = np.unravel_index(img.argmax(), img.shape)
    popt, pcov = optimize.curve_fit(
        rotated_gaussian_2d, xy, img[ij].ravel(),
        # jac=lambda *_, **__: jnp.array(jac(*_, **__)).T,
        p0=[img.max(), col, row, min(50, width), min(50, width), jnp.pi/4],
        bounds=np.transpose([(0, jnp.inf),
                             (px_col[0], px_col[-1]),
                             (px_row[0], px_row[-1]),
                             (1, px_col.size),
                             (1, px_row.size),
                             (0, jnp.pi/2)])
    )
    fit = rotated_gaussian_2d(xy, *popt).reshape(img[ij].shape)

    vmin, vmax = itertools.minmax(itertools.chain(itertools.minmax(fit.flatten()),
                                                  itertools.minmax(img[ij].flatten())))

    ax = grid.axes_row[i]

    im = ax[0].pcolormesh(px_col, px_row, img[ij], cmap='inferno', vmin=vmin, vmax=vmax)
    im = ax[1].pcolormesh(px_col, px_row, fit, cmap='inferno', vmin=vmin, vmax=vmax)
    im = ax[2].pcolormesh(px_col, px_row, 100*(img[ij] - fit)/img[ij].max(), cmap='RdBu',
                          norm=colors.CenteredNorm())
    cb = ax[2].cax.colorbar(im, ax=ax[2])
    cb.set_label('Deviation (%)')

    w_px = 4*popt[3:5]
    w_um = w_px * PIXEL_SIZE / np.mean(magnification)

    ax[0].set_ylabel('\n'.join([
        arm,
        rf'$1-R^2$ = {1-R2(img[ij], fit):.1e}',
        rf'$e$ = {eccentricity(*popt[3:5]):.2f}, $\theta$ = {np.rad2deg(popt[-1]):.1f}°',
        r'$\vec{{w}}$ = ({:.1f}, {:.1f}) px'.format(*w_px),
        r'$\vec{{w}}$ = ({:.1f}, {:.1f}) μm'.format(*w_um)
    ]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

grid.axes_row[0][0].set_title('Image')
grid.axes_row[0][1].set_title('Rotated Gaussian Fit')
grid.axes_row[0][2].set_title('Residuals')

# %% 2d TEM_01 + TEM_10
rows, cols = zip(*[map(int, ndimage.center_of_mass(img)) for img in imgs])
diff = np.diff([rows, cols])
dist = np.linalg.norm(diff, ord=2).astype(np.int_)
center_row, center_col = ((diff / 2).astype(np.int_) + np.array([rows[0:1], cols[0:1]])).squeeze()

scale = 5

px_row = jnp.arange(center_row - scale * dist, center_row + scale * dist)
px_col = jnp.arange(center_col - scale * dist, center_col + scale * dist)
# px_row = jnp.arange(0, imgs.shape[1])
# px_col = jnp.arange(0, imgs.shape[2])
ij = jnp.ix_(px_row, px_col)
xy = jnp.array(jnp.meshgrid(px_col, px_row)).astype(jnp.float_)

fig = plt.figure()
grid = ImageGrid(fig, 111, (len(imgs), 3), axes_pad=0.1, share_all=True, cbar_mode='edge')
for i, (img, arm) in enumerate(zip(imgs, ['Excitation', 'Detection'])):
    row, col = ndimage.center_of_mass(img)
    func = functools.partial(TEM_ℓm_intensity, [0, 1], [1, 0])
    popt, pcov = optimize.curve_fit(func, xy, img[ij].ravel(),
                                    p0=[col, row, 50, 5] + [img.max() / 2] * 2,
                                    bounds=np.transpose([(px_col[0], px_col[-1]),
                                                         (px_row[0], px_row[-1]),
                                                         (1, px_col.size),
                                                         (0, px_col.size)] + [(0, jnp.inf)] * 2))
    fit = func(xy, *popt).reshape(img[ij].shape)

    ax = grid.axes_row[i]

    im = ax[0].pcolormesh(px_col, px_row, img[ij], cmap='inferno')
    im = ax[1].pcolormesh(px_col, px_row, fit, cmap='inferno')
    im = ax[2].pcolormesh(px_col, px_row, img[ij] - fit, cmap='bwr', norm=colors.CenteredNorm())
    cb = ax[2].cax.colorbar(im, ax=ax[2])

    ax[0].set_ylabel('\n'.join([
        arm,
        rf'$R^2 = {R2(img[ij], fit):.3g}$',
        rf'$w = {popt[2]:.1f}$ px'
    ]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

grid.axes_row[0][0].set_title('Image')
grid.axes_row[0][1].set_title('TEM$_{01}$ + TEM$_{10}$ Fit')
grid.axes_row[0][2].set_title('Residuals')


# %% 2d TEM_11
rows, cols = zip(*[map(int, ndimage.center_of_mass(img)) for img in imgs])
diff = np.diff([rows, cols])
dist = np.linalg.norm(diff, ord=2).astype(np.int_)
center_row, center_col = ((diff / 2).astype(np.int_) + np.array([rows[0:1], cols[0:1]])).squeeze()

scale = 5
n = 4

px_row = jnp.arange(center_row - scale * dist, center_row + scale * dist)
px_col = jnp.arange(center_col - scale * dist, center_col + scale * dist)
# px_row = jnp.arange(0, imgs.shape[1])
# px_col = jnp.arange(0, imgs.shape[2])
ij = jnp.ix_(px_row, px_col)
xy = jnp.array(jnp.meshgrid(px_col, px_row)).astype(jnp.float_)

fig = plt.figure()
grid = ImageGrid(fig, 111, (len(imgs), 3), axes_pad=0.1, share_all=True, cbar_mode='edge')
for i, (img, arm) in enumerate(zip(imgs, ['Excitation', 'Detection'])):
    row, col = ndimage.center_of_mass(img)
    func = functools.partial(TEM_ℓm_intensity, [1], [1])
    popt, pcov = optimize.curve_fit(func, xy, img[ij].ravel(),
                                    p0=[col, row, 50, 5] + [img.max()],
                                    bounds=np.transpose([(px_col[0], px_col[-1]),
                                                         (px_row[0], px_row[-1]),
                                                         (1, px_col.size),
                                                         (0, px_col.size)] + [(0, jnp.inf)]))
    fit = func(xy, *popt).reshape(img[ij].shape)

    vmin, vmax = itertools.minmax(itertools.chain(itertools.minmax(fit.flat),
                                                  itertools.minmax(img[ij].flat)))

    ax = grid.axes_row[i]

    im = ax[0].pcolormesh(px_col, px_row, img[ij], cmap='inferno', vmin=vmin, vmax=vmax)
    im = ax[1].pcolormesh(px_col, px_row, fit, cmap='inferno', vmin=vmin, vmax=vmax)
    im = ax[2].pcolormesh(px_col, px_row, img[ij] - fit, cmap='bwr', norm=colors.CenteredNorm())
    cb = ax[2].cax.colorbar(im, ax=ax[2])

    ax[0].set_ylabel('\n'.join([
        arm,
        rf'$R^2 = {R2(img[ij], fit):.3g}$',
        rf'$w = {popt[2]:.1f}$ px'
    ]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    print(f'Relative intensities {arm.lower()}:\t{popt[-n:]}')

grid.axes_row[0][0].set_title('Image')
grid.axes_row[0][1].set_title(rf'$\sum_{{\ell=0}}^{n-1} a_\ell\mathrm{{TEM}}_{{0\ell}}$')
grid.axes_row[0][2].set_title('Residuals')
