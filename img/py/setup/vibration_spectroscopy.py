# %% Imports
import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import uncertainties.unumpy as unp
import xarray as xr
from cycler import cycler
from python_spectrometer import Spectrometer
from qcodes.utils.json_utils import NumpyJSONEncoder
from qutil import const, functools, io
from qutil.misc import filter_warnings
from qutil.plotting import changed_plotting_backend
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_50, RWTH_COLORS_75, get_rwth_color_cycle
from qutil.signal_processing import fourier_space, real_space
from scipy import interpolate, odr, special
from uncertainties import ufloat

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (  # noqa
    MAINSTYLE, MARGINSTYLE, MARGINWIDTH, PATH, TEXTWIDTH, init, markerprops, n_GaAs
)

ORIG_DATA_PATH = pathlib.Path(
    r'\\janeway\User AG Bluhm\Common\GaAs\Hangleiter\characterization\vibrations'
)
DATA_PATH = PATH.parent / 'data/vibrations'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

init(MAINSTYLE, backend := 'qt')
# %% Functions


def to_relative_paths(spect, file, *keys):
    with io.changed_directory(DATA_PATH):
        spect.savepath = '.'
        spect.relative_paths = True
        for key in keys:
            # 'x' in the metadata corresponds to 'y' in the thesis.
            spect[key]['comment'] = spect[key]['comment'].replace('x Valhalla puck ', '')
            spect[key]['comment'] = spect[key]['comment'].replace('Optical gate x ', '')
            spect[key]['comment'] = spect[key]['comment'].replace('cold head resting ', '')
            spect[key]['comment'] = spect[key]['comment'].replace(' suspension', ', susp.')
            spect[key]['filepath'] = pathlib.Path(spect[key]['filepath']).name
            if 'station_snapshot' in spect[key]['settings']:
                spect[key]['settings']['station_snapshot'] = json.dumps(
                    spect[key]['settings']['station_snapshot'],
                    cls=NumpyJSONEncoder
                )
            spect.reprocess_data(key, save='overwrite')
        for key in set(spect.keys()).difference(spect._parse_keys(*keys)):
            spect.drop(key)
        spect.serialize_to_disk(file, verbose=True)


# %%% Accelerometer


def sensitivity_deviation(temp_in_kelvin):
    return 1 - 0.05 / 325 * (const.convert_temperature(temp_in_kelvin, 'K', 'F') - 75)


def sensitivity(x, f, temperatures: dict[str: float] | None = None, sensor=None, **kwargs):
    temperatures = temperatures or {}
    match sensor:
        case 'Wilcoxon Research 731-207':
            return x / (9.9 / const.g), f
        case 'PCB 351B42':
            if (T_MC := temperatures.get('MC Plate Cernox', 293)) < 50:
                T_MC = temperatures.get('MC RuO2', 293)
            with filter_warnings(action='ignore', category=RuntimeWarning):
                return x / (_spline(np.log10(f) * sensitivity_deviation(T_MC))), f
        case None:
            raise RuntimeError


def comp_gain(x, gain=1, **_):
    return x / gain


# %%% Optical


def sort_files(s):
    return sorted(s, key=lambda s: int(pathlib.Path(s).stem.split("_")[-1]))


def pos_vs_vdc(vdc, a, b):
    return (a*vdc + b)*1e6


def vdc_vs_pos(pos, a, b):
    return (pos*1e-6 - b)/a


def pos_vs_cps(cps, a, b):
    return (cps*1e-6 - b)/a


def cps_calib(cts, fs, pos_vs_cps_calibration, **_):
    return pos_vs_cps(cts*fs, *pos_vs_cps_calibration)


def erf_theory(x, I0, w0, x0, R):
    return 0.5*I0*w0*np.sqrt(0.5*np.pi)*(1 - (1 - R)*special.erf((x - x0)*np.sqrt(2)/w0))


# %% Calibrations
# %%% Accelerometer
# This is for the PCB 351B42
f_calib = np.array([10, 15, 30, 50, 100, 300, 500, 1000, 2000])  # Hz
s_calib = np.array([0.995, 0.997, 0.998, 0.999, 1, 1.001, 1.002, 1.005, 1.012])*9.99e-3  # V/(m/s²)
_spline = interpolate.interp1d(np.log10(f_calib), s_calib, 'quadratic', fill_value='extrapolate')

f_calib_cond = np.array([1, 10, 100, 1000, 10000])
v_calib_cond = np.array([1.3, 0.1, 0.08, 0.07, 0.07])*1e-6  # V/sqrt(Hz)
x_calib_cond = abs(functools.chain(sensitivity, fourier_space.derivative, n_args=2)(
    v_calib_cond, f_calib_cond, order=-2, sensor='PCB 351B42'
)[0])

f_noise_floor = np.array([1, 10, 100, 1000])
S_noise_floor = abs(fourier_space.derivative(np.array([980, 216, 58.9, 15.7])*1e-6,
                                             f_noise_floor, order=-2)[0])

# %%% Optical
# %%%% Camera calibration
vdc_calib = np.arange(11)
seq = tifffile.FileSequence(tifffile.imread, str(DATA_PATH / 'vdc_mapping/*.tif'),
                            sort=sort_files).asarray()[..., 0] / 255
row = 470
col = np.arange(480, 540)

I_min = 94 / 255
I_max = 121 / 255
seqnan = np.copy(seq)
seqnan[(I_min > seq) | (I_max < seq)] = np.nan

popt = np.empty((len(vdc_calib), len(vdc_calib), 2))
pcov = np.empty((len(vdc_calib), len(vdc_calib), 2, 2))
for j, r in enumerate(np.arange(-5, 6) + row):
    for i, line in enumerate(seqnan[:, r, col]):
        x = col[~np.isnan(line)]
        y = line[~np.isnan(line)]
        popt[i, j], pcov[i, j] = np.polyfit(x, y, 1, cov=True)

pavg, sow = np.average(popt, axis=1, weights=1/pcov[..., range(2), range(2)], returned=True)
perr = 1 / np.sqrt(sow)
a, b = unp.uarray(pavg, perr).T

gate_width_camera = ufloat(116, 3)  # px
gate_width_actual = 14e-6  # m
magnification = gate_width_camera / gate_width_actual  # px/m
I_set = ufloat((I_max + I_min) / 2, 1/255/np.sqrt(12))  # normalized intensity
position = (I_set - b) / a / magnification  # m
position -= position.mean()  # we don't know anything about absolute positions

popt_posvdc, pcov_posvdc = np.polyfit(vdc_calib, unp.nominal_values(position), 1, cov=True,
                                      w=1/unp.std_devs(position))

# %%%% Count rate calibration
ds = xr.load_dataset(DATA_PATH / 'vdc_calibration.h5', engine='h5netcdf')
count_rate = ds.counter_countrate
# 'y_axis' somewhat surprisingly correctly corresponds to 'y' in the thsis.
x = count_rate['positioners_y_axis_voltage']
y1 = count_rate * 1e-6

v_min = 0.5
v_max = 7.0
mask = (v_min <= x) & (x <= v_max)

vdc = x[mask]
pos = pos_vs_vdc(vdc, *popt_posvdc)
poserr = pos_vs_vdc(vdc, *(popt_posvdc + np.sqrt(np.diag(pcov_posvdc)))) - pos
cps = y1[mask].mean('counter_time_axis')
cpserr = y1[mask].std('counter_time_axis') / np.sqrt(count_rate.sizes['counter_time_axis'])

linear_data = odr.RealData(pos, cps, sx=poserr, sy=cpserr)
linear_model = odr.Model(lambda beta, x: beta[0]*x + beta[1])
linear_fit = odr.ODR(linear_data, linear_model, beta0=[2.5, 0])
linear_output = linear_fit.run()

# s has units of Mcps/μm
s, b = unp.uarray(linear_output.beta, linear_output.sd_beta)

# %%%%% Plot camera image
erroralpha = 0.5
errorcolor = RWTH_COLORS['blue']
ix = (np.where(~np.isnan(seqnan[0, row, col]))[0][[0, -1]] + (col[0] - 400))

with mpl.style.context(MARGINSTYLE, after_reset=True), changed_plotting_backend('pgf'):
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(MARGINWIDTH, 2.35),
                            gridspec_kw={'height_ratios': [2*const.golden, 1, 1]})
    ax = axs[0]
    ax.imshow(seq[0, row-75:row+75, 400:600], cmap='binary', aspect='equal')
    ax.plot([0, 200], [75-5]*2, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([0, 200], [75+5]*2, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([ix[0]]*2, [0, 150], '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([ix[1]]*2, [0, 150], '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.axis('off')

    ax = axs[1]
    ax.plot(seq[0, row, 400:600], color='k')
    ax.fill_betweenx(lim := ax.get_ylim(), *ix, color=RWTH_COLORS_50['black'], alpha=0.3,
                     linewidth=0.0)
    ax.plot([ix[0]]*2, lim, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([ix[1]]*2, lim, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.set_ylim(lim)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2]
    ax.plot(np.gradient(seq[0, row, 400:600], 1), color='k')
    ax.fill_betweenx(lim := ax.get_ylim(), *ix, color=RWTH_COLORS_50['black'], alpha=0.3,
                     linewidth=0.0)
    ax.plot([ix[0]]*2, lim, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([ix[1]]*2, lim, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.set_ylim(lim)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(SAVE_PATH / 'knife_edge.pdf')

# %%%% Plot linear fit
erroralpha = 0.5
errorcolor = RWTH_COLORS['blue']
ix = (np.where(~np.isnan(seqnan[0, row, col]))[0][[0, -1]] + (col[0] - 400))
xx = pos_vs_vdc(x, *popt_posvdc)
yy = y1.mean('counter_time_axis')
xxerr = [xx - pos_vs_vdc(x, *(popt_posvdc - np.sqrt(np.diag(pcov_posvdc)))),
         pos_vs_vdc(x, *(popt_posvdc + np.sqrt(np.diag(pcov_posvdc)))) - xx]
yyerr = y1.std('counter_time_axis') / np.sqrt(count_rate.sizes['counter_time_axis'])

with mpl.style.context(MARGINSTYLE, after_reset=True), changed_plotting_backend('pgf'):
    fig, axs = plt.subplots(nrows=2, figsize=(MARGINWIDTH, MARGINWIDTH / const.golden * 2),
                            gridspec_kw=dict(height_ratios=(2, 3)),
                            layout='constrained')
    fig.get_layout_engine().set(h_pad=0)

    # pos vs vdc
    ax = axs[0]
    ax.errorbar(vdc_calib, unp.nominal_values(position)*1e6, unp.std_devs(position)*1e6,
                ecolor=mpl.colors.to_rgb(errorcolor) + (erroralpha,),
                **markerprops(errorcolor))
    ax.plot(vdc_calib, np.polyval(popt_posvdc, vdc_calib)*1e6, zorder=5)
    ax.margins(x=0.05)
    ax.grid()
    ax.set_xlabel(r'$V_\mathrm{DC}$ (V)')
    ax.set_ylabel(r'$y - \langle y\rangle$ (μm)')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(which='both', labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position('top')
    xlim = ax.get_xlim()

    # cps vs pos
    ax = axs[1]
    ax.errorbar(xx, yy, yerr=yyerr, xerr=xxerr, label='Data',
                ecolor=mpl.colors.to_rgb(errorcolor) + (erroralpha,),
                **markerprops(errorcolor, marker='.'))
    ax.plot(pos, linear_model.fcn(linear_output.beta, pos), zorder=5, label='Fit')

    ax2 = ax.secondary_xaxis(
        'top',
        functions=(functools.partial(vdc_vs_pos, a=popt_posvdc[0], b=popt_posvdc[1]),
                   functools.partial(pos_vs_vdc, a=popt_posvdc[0], b=popt_posvdc[1]))
    )
    ax2.sharex(axs[0])
    ax.set_ylabel(r'$\Phi_{\mathrm{r}}$ (Mcps)')
    ax.set_xlabel(r'$y - \langle y\rangle$ (μm)')
    ax.grid()
    ax.set_yticks([2, 3, 4])
    ax.set_xlim(pos_vs_vdc(np.array(xlim), *popt_posvdc))
    ax.set_ylim(2, 4)

    fig.savefig(SAVE_PATH / 'knife_edge_fits.pdf')

# %%%% Theory plot
x = np.linspace(-1.5, 1.5, 1001)
I0 = 2
w0 = .3
x0 = 0
R = 0.2
N = I0*w0*np.sqrt(0.5*np.pi)

with mpl.style.context([MARGINSTYLE]), changed_plotting_backend('pgf'):
    fig, ax = plt.subplots(layout='constrained')

    ax.plot(x, erf_theory(x*w0, I0, w0, x0, R) / N)

    ylim = ax.get_ylim()
    ax.plot(x, (-I0*(1-R)*x*w0 + 0.5*N) / N, ls='--', color=RWTH_COLORS_75['black'])
    ax.set_ylim(ylim)

    ax.grid()
    ax.margins(x=0)
    ax.set_yticks([R/2, 0.5, 1-R/2],
                  [r'$\frac{\mathrm{r}_0}{2}$', r'$\frac{1}{2}$', r'$1-\frac{\mathrm{r}_0}{2}$'],
                  va='center')
    ax.set_ylabel(r'$P_{\mathrm{r}}(y) / (I_0 w_0 \sqrt{\pi/2})$')

    match mpl.get_backend():
        case 'pgf':
            ax.set_xlabel(r'$\flatfrac{y}{w_0}$')
        case 'qtagg':
            ax.set_xlabel(r'$y/w_0$')

    fig.savefig(SAVE_PATH / 'knife_edge_theory.pdf')

# %%% Full knife-edge fit
x = count_rate['positioners_y_axis_voltage']
xx = pos_vs_vdc(x, *popt_posvdc)
yy = y1.mean('counter_time_axis')
sxx = [xx - pos_vs_vdc(x, *(popt_posvdc - np.sqrt(np.diag(pcov_posvdc)))),
       pos_vs_vdc(x, *(popt_posvdc + np.sqrt(np.diag(pcov_posvdc)))) - xx]
syy = y1.std('counter_time_axis') / np.sqrt(count_rate.sizes['counter_time_axis'])

# GaAs @ 800 nm
n = n_GaAs(30e-3)
R = abs((n - 1) / (n + 1))**2  # at 30 mK

knife_edge_data = odr.RealData(xx, yy, sx=np.average(sxx, axis=0), sy=syy)
knife_edge_model = odr.Model(lambda beta, x: erf_theory(-x, *beta))

knife_edge_fit = odr.ODR(knife_edge_data, knife_edge_model, beta0=[5, 1, 0, R], ifixb=[1, 1, 1, 1])
knife_edge_output = knife_edge_fit.run()
if 'Sum of squares convergence' not in knife_edge_output.stopreason:
    knife_edge_output = knife_edge_fit.restart(100)

fitpar = unp.uarray(knife_edge_output.beta, knife_edge_output.sd_beta)

print('Knife edge fit results:')
print(f'w_0 = {fitpar[1]:.3g} μm')
print(f'R = {fitpar[3]:.3g}')

knife_edge_fit_rfix = odr.ODR(knife_edge_data, knife_edge_model, beta0=[5, 1, 0, R],
                              ifixb=[1, 1, 1, 0])
knife_edge_output_rfix = knife_edge_fit_rfix.run()
if 'Sum of squares convergence' not in knife_edge_output_rfix.stopreason:
    knife_edge_output_rfix = knife_edge_fit_rfix.restart(100)

fitpar_rfix = unp.uarray(knife_edge_output_rfix.beta, knife_edge_output_rfix.sd_beta)

# %%%% Plot
erroralpha = 0.5
errorcolor = RWTH_COLORS['blue']

with mpl.style.context(MARGINSTYLE, after_reset=True), changed_plotting_backend('pgf'):
    fig, ax = plt.subplots(layout='constrained')

    ax.errorbar(xx, yy, yerr=syy, xerr=sxx, alpha=0.75,
                ecolor=mpl.colors.to_rgb(errorcolor) + (erroralpha,),
                **markerprops(errorcolor, marker='.', markersize=2.5))

    ax.plot(xtmp := np.linspace(-1, 1, 1001), erf_theory(-xtmp, *unp.nominal_values(fitpar)),
            zorder=5)

    ylim = ax.get_ylim()
    ax.plot(xtmp, erf_theory(-xtmp, *unp.nominal_values(fitpar_rfix)), ls='--',
            color=RWTH_COLORS_50['magenta'], zorder=5)

    ax.set_ylabel(r'$\Phi_{\mathrm{r}}$ (Mcps)')
    ax.set_xlabel(r'$y - \langle y\rangle$ (μm)')
    ax.grid()
    ax.set_yticks([2, 3, 4])
    ax.set_ylim(1.6)

    fig.savefig(SAVE_PATH / 'knife_edge_erf.pdf')

# %% Load spects
with changed_plotting_backend('qtagg'):
    spect_accel = Spectrometer.recall_from_disk(
        DATA_PATH / 'spectrometer_odin_puck', savepath=DATA_PATH
    )
    spect_optic = Spectrometer.recall_from_disk(
        DATA_PATH / 'spectrometer_photon_counting_23-09-06', savepath=DATA_PATH
    )

spects = [spect_accel, spect_optic]

# %%% Apply settings
spect_accel.procfn = functools.chain(comp_gain, functools.scaled(1e6))
spect_accel.psd_estimator = functools.partial(
    real_space.welch, fourier_procfn=(sensitivity, fourier_space.derivative)
)
spect_accel.processed_unit = 'μm'
spect_accel.reprocess_data(*spect_accel.keys(), detrend='constant')

spect_optic.procfn = cps_calib
spect_optic.reprocess_data(*spect_optic.keys(), pos_vs_cps_calibration=linear_output.beta,
                           detrend='constant')

figure_kw = dict(figsize=(TEXTWIDTH, TEXTWIDTH / const.golden * 1.25))
legend_kw = dict(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                 ncols=2, mode="expand", borderaxespad=0., frameon=False)
settings = dict(
    plot_style=MAINSTYLE,
    prop_cycle=get_rwth_color_cycle(100),
    plot_timetrace=False,
    plot_cumulative=True
)


def apply_settings(spect, settings, figure_kw, legend_kw):
    pm = spect._plot_manager
    shown = pm.shown

    spect.hide(*shown)
    plt.close(spect.fig)

    pm.legend_kw.update(legend_kw)
    pm.figure_kw.update(figure_kw)
    for key, val in settings.items():
        setattr(spect, key, val)

    spect.show(*shown)
    pm._leg = spect.ax[0].legend(labels=[com for _, com in spect.keys()], **pm.legend_kw)


for typ, spect in zip(['spect_accel', 'spect_optic'], spects):
    apply_settings(spect, settings, figure_kw, legend_kw)

# spect_accel.ax[1].set_yscale('asinh', linear_width=1.5e-2)
# spect_optic.ax[1].set_yscale('asinh', linear_width=1.65e-3)
# spect_accel.ax[1].set_ylim(0)
# spect_optic.ax[1].set_ylim(0)
spect_optic.ax[1].set_yticks([0.0, 0.1, 0.2])
spect_accel.ax[1].set_yticks([0, 5, 10])

# %%% Resave
# to_relative_paths(spect_accel, 'spectrometer_odin_puck', 2, 3, 4, 5)
# to_relative_paths(spect_optic, 'spectrometer_photon_counting_23-09-06', *spect_optic.keys())

# %% Plot
data = spect_optic[0]
cts = data['timetrace_raw']
fs = data['settings']['fs']
conversion_factor = fs / (s * 1e6)  # a has units Mcps/μm, so convert to cps/μm
shot_noise_floor = 2 * cts.mean() / fs * conversion_factor ** 2  # factor two for one-sided
# print(f'shot noise floor for {key} is {unp.sqrt(shot_noise_floor)}')

spect_optic.ax[0].axhline(np.sqrt(shot_noise_floor.nominal_value), ls='--',
                          color=RWTH_COLORS_50['black'], zorder=5)

with changed_plotting_backend('pgf'):
    for typ, spect in zip(['spect_accel', 'spect_optic'], spects):
        spect.fig.savefig(SAVE_PATH / f'{typ}.pdf')

# %% Vibration criterion individual (unused)
with mpl.style.context([MAINSTYLE]), changed_plotting_backend('pgf'):
    for typ, spect in zip(['spect_accel', 'spect_optic'], spects):
        fig, ax = plt.subplots(layout='constrained')

        for key, sty in zip(spect.keys(), get_rwth_color_cycle(100)):
            S = np.sqrt(spect[key]['S_processed'].mean(0))
            f = spect[key]['f_processed']

            vc, vc_f = fourier_space.octave_band_rms(*fourier_space.derivative(S, f, order=1),
                                                     fraction=3)

            ax.loglog(vc_f, vc, label=key[1], zorder=5, color=sty['color'],
                      **(markerprops(sty['color'], markersize=4) | dict(ls='--')))

        xlim = spect[key]['settings']['f_min'], spect[key]['settings']['f_max']
        ax.plot([8, xlim[1]], [25, 25], marker='', ls='-', color=RWTH_COLORS_50['black'])
        ax.plot([xlim[0], 8], [200/xlim[0], 25], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='-')
        ax.plot([8, xlim[1]], [50, 50], marker='', ls='--', color=RWTH_COLORS_50['black'])
        ax.plot([xlim[0], 8], [400/xlim[0], 50], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='--')
        ax.plot([8, xlim[1]], [100, 100], marker='', ls='-.', color=RWTH_COLORS_50['black'])
        ax.plot([xlim[0], 8], [800/xlim[0], 100], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='-.')
        ax.plot([8, xlim[1]], [200, 200], marker='', ls=':', color=RWTH_COLORS_50['black'])
        ax.plot([xlim[0], 8], [1600/xlim[0], 200], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls=':')

        ax.grid()
        ax.set_xlim(xlim)
        match typ:
            case 'spect_accel':
                ax.set_ylim(1e-3, 1e3)
            case 'spect_optic':
                ax.set_ylim(top=5e2)

        ax.set_xlabel(r'$f_\mathrm{m}$ (Hz)')
        ax.set_ylabel(r'$1/3$ octave band $\mathrm{RMS}$ (μm/s)')
        ax.legend(**legend_kw)

        # fig.savefig(SAVE_PATH / f'{typ}_vc.pdf')

# %% Vibration criterion together
cyclers = [
    get_rwth_color_cycle(50)[1:] + cycler(marker=['o']*12, ls=['--']*12),
    get_rwth_color_cycle(100)[1:] + cycler(marker=['D']*12, markersize=[4]*12, ls=['-.']*12),
]

with mpl.style.context([MAINSTYLE]), changed_plotting_backend('pgf'):
    fig, ax = plt.subplots(layout='constrained')
    lines = []
    for typ, spect, cycle in zip(['spect_accel', 'spect_optic'], spects, cyclers):

        for key, sty in zip(spect.keys(), cycle):
            if 'PTR off' in key[1]:
                continue
            S = np.sqrt(spect[key]['S_processed'].mean(0))
            f = spect[key]['f_processed']

            vc, vc_f = fourier_space.octave_band_rms(*fourier_space.derivative(S, f, order=1),
                                                     fraction=3)

            ln, = ax.loglog(vc_f, vc, zorder=5, **(markerprops(sty['color']) | sty))
            lines.append(ln)

        vc_sn, vc_sn_f = fourier_space.octave_band_rms(
            *fourier_space.derivative(np.full_like(f, np.sqrt(shot_noise_floor.nominal_value)),
                                      f, order=1),
            fraction=3
        )

        ax.loglog(vc_sn_f, vc_sn, ls=(0, (5, 10)), color=RWTH_COLORS_75['black'])

        xlim = spect[key]['settings']['f_min'], spect[key]['settings']['f_max']
        ax.plot([8, xlim[1]], [25, 25], marker='', ls='-', color=RWTH_COLORS_50['black'])
        ax.plot([xlim[0], 8], [200/xlim[0], 25], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='-')
        ax.plot([8, xlim[1]], [50, 50], marker='', ls='--', color=RWTH_COLORS_50['black'])
        ax.plot([xlim[0], 8], [400/xlim[0], 50], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='--')
        ax.plot([8, xlim[1]], [100, 100], marker='', ls='-.', color=RWTH_COLORS_50['black'])
        ax.plot([xlim[0], 8], [800/xlim[0], 100], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='-.')
        ax.plot([8, xlim[1]], [200, 200], marker='', ls=':', color=RWTH_COLORS_50['black'])
        ax.plot([xlim[0], 8], [1600/xlim[0], 200], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls=':')

    ax.set_yticks([1e-3, 1e-1, 1e1, 1e3])
    ax.set_xlim(xlim)
    ax.set_ylim(1e-3, 1e3)

    ax.set_xlabel(r'$f_\mathrm{m}$ (Hz)')
    ax.set_ylabel(r'$1/3$ octave band $\mathrm{RMS}$ (μm/s)')
    ax.legend(
        handles=lines,
        labels=['Accel. susp. off', 'Accel. susp. on', 'Optical susp. off', 'Optical susp. on'],
        **legend_kw
    )

    fig.savefig(SAVE_PATH / 'vc.pdf')

# %% Relative dB
settings['plot_dB_scale'] = True
settings['plot_amplitude'] = False

with mpl.style.context([MARGINSTYLE], after_reset=True):
    fig, ax = plt.subplots(nrows=2, sharex=True, layout='constrained',
                           gridspec_kw=dict(height_ratios=[3, 2]),
                           figsize=(MARGINWIDTH, MARGINWIDTH / const.golden * 2))
    ax[0].set_prop_cycle(color=mpl.color_sequences['rwth'][1:])
    ax[1].set_prop_cycle(color=mpl.color_sequences['rwth'][1:])

    for typ, spect in zip(['spect_accel', 'spect_optic'], spects):
        spect.hide('PTR off, susp. off', 'PTR off, susp. on')
        spect.set_reference_spectrum('PTR on, susp. off')
        apply_settings(spect, settings, figure_kw, legend_kw)

        ln = spect._plot_manager.lines[1, 'PTR on, susp. on']['main']['processed']['line']
        ax[0].semilogx(*ln.get_data(),
                       label='Accelerometer' if typ == 'spect_accel' else 'Optical')
        ln = spect._plot_manager.lines[1, 'PTR on, susp. on']['cumulative']['processed']['line']
        ax[1].semilogx(*ln.get_data())

    ax[0].axhline(color=RWTH_COLORS_75['black'])
    ax[1].axhline(color=RWTH_COLORS_75['black'])
    ax[0].set_yticks([20, 0, -20, -40, -60])
    ax[1].set_yticks([0, -20])
    ax[0].set_xlim(spect.ax[-1].get_xlim())
    ax[1].set_xlabel(spect.ax[-1].get_xlabel())
    ax[0].set_ylabel('Instant. (dB)')
    ax[1].set_ylabel('Integrated (dB)')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend(**(legend_kw | dict(ncols=1)))

    with changed_plotting_backend('pgf'):
        fig.savefig(SAVE_PATH / 'spect_dB.pdf')
