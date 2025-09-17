# %% Imports
import pathlib
import sys

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from qutil.plotting.colors import RWTH_COLORS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import MARGINSTYLE, PATH, init  # noqa

DATA_PATH = PATH.parent / 'data/imaging'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/experiment'
SAVE_PATH.mkdir(exist_ok=True)

init(MARGINSTYLE, backend := 'pgf')
qty = r'\qty{{{}}}{{\micro\meter}}' if backend == 'pgf' else r'{} μm'
# %% Load
files = ['image856', 'image845']
sizes = [1, 10]
images = []
images_rgba = []
scale_bars = []
mins, maxs = [], []
for file in files:
    rgba = Image.open(DATA_PATH / f'{file}.png')
    r, g, b, a = rgba.split()
    images_rgba.append(rgba)
    images.append(rgba.convert('L'))
    scale_bars.append(np.array(r) - np.array(g))
    mins.append(scale_bars[-1].nonzero()[1].min())
    maxs.append(scale_bars[-1].nonzero()[1].max())

scales = np.array(maxs) - np.array(mins)
# %% Match detail
detail = np.array(images[0])[:690]
large = np.array(images[1])[:690]

scale = np.divide(*sizes) / np.divide(*scales)
h, w = detail.shape
detail_down = cv2.resize(detail, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# Perform template matching
result = cv2.matchTemplate(large, detail_down, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Extract best match location
top_left = np.array(max_loc)

# %% Plot
rect_kwargs = dict(facecolor='none', linestyle='--', linewidth=0.3)

large = images_rgba[1].crop((200, 100, 900, 660))
medium = images_rgba[0].crop((0, 0, 1024, 660))
small = medium.crop((165, 120, 405, 290))
height_ratios = (large.height/large.width, medium.height/medium.width, small.height/small.width)
scalex = (1, medium.width/large.width, small.width/large.width)
scaley = (1, medium.height/large.height, small.height/medium.height)

fig, axes = plt.subplots(3, gridspec_kw=dict(height_ratios=height_ratios))

ax = axes[0]
ax.imshow(large)
rect = mpl.patches.Rectangle(top_left - [200, 100],
                             detail_down.shape[1], detail_down.shape[0],
                             edgecolor=RWTH_COLORS['blue'], **rect_kwargs)
ax.add_patch(rect)
bar = mpl.patches.Rectangle(
    xy := (30, large.height - 50), scales[1], 10,
    color='white', linewidth=0
)
ax.add_patch(bar)
ax.annotate(qty.format(10), xy, color='white', fontsize=3, ha='left', va='bottom')
ax.axis('off')
ref = ax.get_window_extent().height

ax = axes[1]
ax.imshow(medium)
rect = mpl.patches.Rectangle((165, 120), small.width, small.height,
                             edgecolor=RWTH_COLORS['magenta'], **rect_kwargs)
ax.add_patch(rect)
bar = mpl.patches.Rectangle(
    xy := (30*scalex[1], medium.height - 50*scaley[1]),
    scales[0]*scalex[1], 10*ref/ax.get_window_extent().height/large.height*medium.height,
    color='white', linewidth=0
)
ax.add_patch(bar)
ax.annotate(qty.format(1), xy, color='white', fontsize=3, ha='left', va='bottom')
ax.set_xticks([])
ax.set_yticks([])
for spine in axes[1].spines.values():
    spine.set_linestyle((0, (3, 3)))
    spine.set_linewidth(0.5)
    spine.set_color(RWTH_COLORS['blue'])

ax = axes[2]
ax.imshow(small)
colors = [RWTH_COLORS['orange'], RWTH_COLORS['green'], RWTH_COLORS['teal'],
          RWTH_COLORS['green'], RWTH_COLORS['teal']]
bar = mpl.patches.Rectangle(
    xy := (30*scalex[2], small.height - 50*scaley[2]),
    scales[0]*scalex[2]*medium.width/small.width,
    10*ref/ax.get_window_extent().height/large.height*small.height,
    color='white', linewidth=0
)
ax.add_patch(bar)
ax.annotate(qty.format(1), xy, color='white', fontsize=3, ha='left', va='bottom')
ax.set_xticks([])
ax.set_yticks([])
for spine in axes[2].spines.values():
    spine.set_linestyle((0, (3, 3)))
    spine.set_linewidth(0.5)
    spine.set_color(RWTH_COLORS['magenta'])

fig.tight_layout(pad=0, h_pad=0, w_pad=0)
fig.savefig(SAVE_PATH / 'sem.pdf', pad_inches=2/72)
