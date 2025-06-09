# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutil.plotting import colors
import tifffile
from PIL import Image
from scipy import optimize
from qutil import itertools, misc
from mpl_toolkits.axes_grid1 import ImageGrid
import uncertainties.unumpy as unp

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, PATH, TEXTWIDTH, TOTALWIDTH, init,  # noqa
                    markerprops, n_GaAs)

DATA_PATH = PATH.parent / 'data/imaging'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

with misc.filter_warnings(action='ignore', category=RuntimeWarning):
    SEQUENTIAL_CMAP = colors.make_sequential_colormap('blue', endpoint='white').reversed()
    DIVERGING_CMAP = colors.make_diverging_colormap(['magenta', 'green'], endpoint='blackwhite')

jax.config.update("jax_enable_x64", True)
init(MAINSTYLE, backend := 'pgf')
# %% Functions


@jax.jit
def gaussian_2d(xy, a, w_x, w_y):
    x, y = xy
    return a * jnp.exp(-2 * x**2 / w_x**2 - 2 * y**2 / w_y**2)


@jax.jit
def rotated_gaussian_2d(xy, a, μ_x, μ_y, w_x, w_y, θ):
    R = jnp.array([[jnp.cos(θ), -jnp.sin(θ)],
                   [jnp.sin(θ), +jnp.cos(θ)]])
    xpyp = jnp.tensordot(R, xy - jnp.array([[[μ_x, μ_y]]]).T, axes=[1, 0])
    return gaussian_2d(xpyp, a, w_x, w_y).ravel()


def fit_n_plot(arms, magnification):
    imgs = []
    fits = []
    for arm in arms:
        img = tifffile.imread(DATA_PATH / f'24-11-19_{arm}_spot_on_optical_gate_less_power.tif')
        img = Image.fromarray(img, mode='RGBA')
        img_array = np.array(img.convert('F'))
        img_array /= img_array.max()
        imgs.append(img_array)

    imgs = jnp.array(imgs)

    # rows, cols = zip(*[map(int, ndimage.center_of_mass(img)) for img in imgs])
    rows, cols = np.unravel_index([img.argmax() for img in imgs], imgs.shape[1:])
    diff = np.diff([rows, cols])
    dist = np.linalg.norm(diff, ord=2).astype(np.int_)
    center_row, center_col = (
        (diff / 2).astype(np.int_) + np.array([rows[0:1], cols[0:1]])
    ).squeeze()

    # at least n pixels
    width = max(15, 5 * dist)

    px_row = jnp.arange(center_row - width, center_row + width)
    px_col = jnp.arange(center_col - width, center_col + width)
    ij = jnp.ix_(px_row, px_col)
    xy = jnp.array(jnp.meshgrid(px_col, px_row))

    fig = plt.figure()
    grid = ImageGrid(fig, 111, (len(imgs), 3), axes_pad=0.1, share_all=True, cbar_mode='edge')

    for i, (img, arm) in enumerate(zip(imgs, ['Excitation', 'Detection'])):
        # row, col = ndimage.center_of_mass(img)
        row, col = np.unravel_index(img.argmax(), img.shape)
        popt, pcov = optimize.curve_fit(
            rotated_gaussian_2d, xy, img[ij].ravel(),
            p0=[img.max(), col, row, min(50, width), min(50, width), jnp.pi/4],
            bounds=np.transpose([(0, jnp.inf),
                                 (px_col[0], px_col[-1]),
                                 (px_row[0], px_row[-1]),
                                 (1, px_col.size),
                                 (1, px_row.size),
                                 (0, jnp.pi/2)])
        )
        fits.append((popt, pcov))
        fit = rotated_gaussian_2d(xy, *popt).reshape(img[ij].shape)

        vmin, vmax = itertools.minmax(itertools.chain(itertools.minmax(fit.flatten()),
                                                      itertools.minmax(img[ij].flatten())))

        ax = grid.axes_row[i]

        im = ax[0].pcolormesh(px_col, px_row, img[ij], cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax)
        im = ax[1].pcolormesh(px_col, px_row, fit, cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax)
        im = ax[2].pcolormesh(px_col, px_row, 100*(img[ij] - fit)/img[ij].max(),
                              cmap=DIVERGING_CMAP, norm=mpl.colors.CenteredNorm())
        cb = ax[2].cax.colorbar(im, ax=ax[2])
        cb.set_label('Deviation (%)')

        ax[0].set_ylabel(arm.capitalize())
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_yticks([])

    grid.axes_row[0][0].set_title('Image')
    grid.axes_row[0][1].set_title('Fit')
    grid.axes_row[0][2].set_title('Residuals')

    fig.savefig(SAVE_PATH / 'spots.pdf')

    return grid, imgs, fits


# %% Calibrate magnification from the "O" in TOP LEFT
file = '24-11-19_white_light_on_top_left.tif'

PIXEL_SIZE = 5.2e-6
NOMINAL_O_SIZE = np.array([4.2175e-6, 4.7455e-6])
# previously extracted values
dx = 24.36
dy = 25.05
magnification = np.abs((dx, dy))
magnification *= PIXEL_SIZE / NOMINAL_O_SIZE
# %% Plot fits
grid, imgs, fits = fit_n_plot(arms := ['detection', 'excitation'], magnification)
# %% Analyze
w_px = np.array([unp.uarray(popt[3:5], np.sqrt(np.diag(pcov))[3:5]) for popt, pcov in fits])
w_um = w_px * PIXEL_SIZE / magnification
for i, (arm, ws) in enumerate(zip(arms, w_um)):
    print(f'{arm}\t(w_x, w_y) = ({ws[0]:.3g}, {ws[1]:.3g})')
