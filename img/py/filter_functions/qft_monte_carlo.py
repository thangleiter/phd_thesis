# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 08:35:32 2021

@author: Hangleiter
"""
import copy
import logging
import pathlib
import json
import shutil
import sys
import time
from datetime import datetime

import git
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import filter_functions as ff
import lindblad_mc_tools as lmt

from load_save_ff_pulse import load_pulse, save_pulse

# %% Flags
_show_progressbar = True
_threads = None
_chunk_noise = 8

_compute_mc = True
_compute_ff = False
_compute_lb = False
_compute_ff_process = False
# %% Paths
base_path = pathlib.Path.home()
base_path /= 'Documents/Uni/Physik/Publication/efficient_calculation_of_generalized_filter_functions'

# git_path = pathlib.Path.home() / 'git'
git_path = pathlib.Path.home() / 'Code'

data_path = base_path / 'data' / 'qft_monte_carlo'
save_path = data_path / str(datetime.now()).partition('.')[0].replace(' ', '_').replace(':', '-')
save_path.mkdir(exist_ok=True, parents=True)

# thisfile = base_path / 'code' / 'python' / 'qft_monte_carlo.py'
thisfile = base_path / 'code' / 'qft_monte_carlo.py'

# %% Load pulses, define parameters
qft_pulse = load_pulse(data_path / 'qft_pulse')
qft_pulse_echo = load_pulse(data_path / 'qft_pulse_echo')
logging.info('Loaded pulses.')

seed = 210302385
rng = np.random.default_rng(seed)

S_0 = {'pink': 1e-9, 'white': 2e-6}
n_MC = 1000
n_omega = 1000
omega = np.geomspace(1e-4, 1e2, n_omega)
# omega = np.geomspace(1e-6, 3e1, n_omega)
omega_dc = np.linspace(0, omega[0], 11)[:-1]
omega_all = np.concatenate([omega_dc, omega])
f = omega/2/np.pi
f_all = omega_all/2/np.pi
max_frequency_sample_spacing = f[0]/10

identifiers = [r'$\sigma_y^{(3)}$']
idx = ff.util.get_indices_from_identifiers(qft_pulse.n_oper_identifiers, identifiers)

# Strip noise operators we're not interested in
qft_pulse.n_opers = qft_pulse.n_opers[idx]
qft_pulse.n_coeffs = qft_pulse.n_coeffs[idx]
qft_pulse.n_oper_identifiers = qft_pulse.n_oper_identifiers[idx]
qft_pulse._control_matrix = qft_pulse._control_matrix[idx]
qft_pulse._filter_function = qft_pulse._filter_function[idx][:, idx]
qft_pulse_echo.n_opers = qft_pulse_echo.n_opers[idx]
qft_pulse_echo.n_coeffs = qft_pulse_echo.n_coeffs[idx]
qft_pulse_echo.n_oper_identifiers = qft_pulse_echo.n_oper_identifiers[idx]
qft_pulse_echo._control_matrix  = qft_pulse_echo._control_matrix[idx]
qft_pulse_echo._filter_function  = qft_pulse_echo._filter_function[idx][:, idx]

pulses = {'echo': qft_pulse_echo, 'noecho': qft_pulse}
pulses_dc = {key: copy.deepcopy(pulse) for key, pulse in pulses.items()}
pulses_all = {key: copy.deepcopy(pulse) for key, pulse in pulses.items()}
# %% Function definitions


def increase_file_counter(path, file_name, sep):
    trunk, sep, ext = file_name.partition(sep)
    _, dot, ext = ext.partition('.')
    count = max((int(parts[2].partition('.')[0]) for p in path.iterdir()
                 if (parts := p.name.partition(sep))[0] == trunk and p.is_file()),
                default=-1)
    return trunk + sep + str(count + 1) + dot + ext


def spectrum(omega, A, omega_A, alpha):
    r"""
    Spectrum :math:`S(\omega) = A\left(\frac{\omega}{\omega_A}\right)^\alpha`
    """
    return A * (omega/omega_A)**alpha


def run_mc(pulse, spectrum, f, n_MC, identifiers, max_frequency_sample_spacing, chunk_noise,
           show_progressbar, threads, spec, key):

    mc_logger.info(f'Solving {spec} {key} with Δf = {max_frequency_sample_spacing:.1e}.')

    this_seed = rng.integers(seed)
    # this_seed = seed
    mc_logger.info(f'Using seed {this_seed}.')

    tic = time.perf_counter()
    mc_infids, propagators, sampler = lmt.infidelity_monte_carlo(
        pulse,
        spectrum,
        f_nyquist=f[-1],
        f_min=f[0],
        n_MC=n_MC,
        n_oper_identifiers=identifiers,
        max_frequency_sample_spacing=max_frequency_sample_spacing,
        chunk_noise=chunk_noise,
        show_progressbar=show_progressbar,
        threads=threads,
        seed=this_seed,
        return_sampler=True,
        return_propagators=True
    )
    toc = time.perf_counter()

    sys.stdout.flush()
    mc_logger.info(f'Took {toc-tic:.1e} s.')
    mc_logger.info(f'Infidelity is {mc_infids.mean():.4e} ± ' +
                 f'{mc_infids.std()/np.sqrt(mc_infids.size):.4e}.')

    file_name = f'monte_carlo_propagators_{key}_{spec}_run_1.npy'
    while (save_path / file_name).exists():
        file_name = increase_file_counter(save_path, file_name, '_run_')

    np.save(save_path / file_name, propagators)
    mc_logger.info(f'Saved propagators at {save_path / file_name}.')

    return propagators, sampler


def run_ff(pulse, spectrum, omega, identifiers, show_progressbar, compute_process, spec, key):

    ff_logger.info(f'Solving {spec} {key} with n_ω = {omega.size}.')

    tic = time.perf_counter()
    ff_infid = ff.infidelity(pulse, spectrum(omega), omega, n_oper_identifiers=identifiers,
                             show_progressbar=show_progressbar,
                             cache_intermediates=compute_process).sum()
    toc = time.perf_counter()

    sys.stdout.flush()
    ff_logger.info(f'Took {toc-tic:.1e} s.')
    ff_logger.info(f'Infidelity is {ff_infid.mean():.4e}.')

    return ff_infid


def run_ff_process(pulse, spectrum, omega, identifiers, show_progressbar, spec, key, DC=False):

    dcstr = '_with_DC' if DC else ''

    ff_logger.info(f'Computing quantum process for {spec} {key} with n_ω = {omega.size}' +
                 f"{dcstr.replace('_', ' ')}.")

    tic = time.perf_counter()
    decayamps = ff.numeric.calculate_decay_amplitudes(pulse, spectrum(omega), omega,
                                                      cache_intermediates=True,
                                                      memory_parsimonious=True,
                                                      show_progressbar=show_progressbar)
    freqshifts = ff.numeric.calculate_frequency_shifts(pulse, spectrum(omega), omega,
                                                       show_progressbar=show_progressbar)
    cumulantfun = ff.numeric.calculate_cumulant_function(pulse,
                                                         decay_amplitudes=decayamps,
                                                         frequency_shifts=freqshifts,
                                                         second_order=True)
    error_transfer_matrix = ff.error_transfer_matrix(pulse, cumulant_function=cumulantfun)
    toc = time.perf_counter()

    sys.stdout.flush()
    ff_logger.info(f'Took {toc-tic:.1e} s.')

    file_name = f'process_{key}_{spec}' + dcstr + '_run_1.npz'
    while (save_path / file_name).exists():
        file_name = increase_file_counter(save_path, file_name, '_run_')

    np.savez(save_path / file_name,
             decay_amplitudes=decayamps,
             frequency_shifts=freqshifts,
             cumulant_function=cumulantfun,
             error_transfer_matrix=error_transfer_matrix)

    ff_logger.info(f'Saved quantum process at {save_path / file_name}.')

    return decayamps, freqshifts, cumulantfun, error_transfer_matrix


def run_ff_dc(pulses, spectrum, omegas, identifiers, show_progressbar, compute_process, spec, key):

    pulse_dc, pulse, pulse_all = pulses
    omega_dc, omega, omega_all = omegas

    ff_logger.info(f'Computing for DC {spec} {key}.')
    ff_logger.debug(f"Control matrix cached: {pulse_dc.is_cached('control matrix')}")
    ff_logger.debug(f"Second order cached: {pulse_dc.is_cached('second order filter function')}")

    tic = time.perf_counter()
    pulse_dc.cache_control_matrix(omega_dc, show_progressbar=show_progressbar,
                                  cache_intermediates=compute_process)
    if compute_process:
        pulse_dc.cache_filter_function(omega_dc, show_progressbar=show_progressbar, order=2)
    toc = time.perf_counter()

    sys.stdout.flush()
    ff_logger.info(f'Took {toc-tic:.1e} s.')

    ff_logger.info('Combining filter functions (DC and fast).')
    ff_logger.debug(f"Control matrix cached: {pulse.is_cached('control matrix')}")
    ff_logger.debug(f"Second order cached: {pulse.is_cached('second order filter function')}")

    tic = time.perf_counter()
    pulse_all.cache_filter_function(
        omega_all,
        control_matrix=np.concatenate(
            [pulse_dc.get_control_matrix(omega_dc, show_progressbar=show_progressbar),
             pulse.get_control_matrix(omega, show_progressbar=show_progressbar)],
            axis=-1
        ),
        show_progressbar=show_progressbar,
        which='generalized'
    )
    if compute_process:
        pulse_all.cache_filter_function(
            omega_all,
            filter_function=np.concatenate(
                [pulse_dc.get_filter_function(omega_dc, order=2, which='generalized',
                                              show_progressbar=show_progressbar),
                 pulse.get_filter_function(omega, order=2, which='generalized',
                                           show_progressbar=show_progressbar)],
                axis=-1
            ),
            order=2,
            show_progressbar=show_progressbar
        )
    toc = time.perf_counter()

    ff_logger.debug(f"Filter function cached: {pulse_all.is_cached('filter function')}")
    ff_infid_all = ff.infidelity(pulse_all, spectrum(omega_all), omega_all,
                                 n_oper_identifiers=identifiers).sum()

    sys.stdout.flush()
    ff_logger.info(f'Took {toc-tic:.1e} s.')
    ff_logger.info(f'Infidelity is {ff_infid_all.mean():.4e}.')

    return ff_infid_all


def run_lb(pulse, S_0, identifiers, show_progressbar, spec, key):

    lb_logger.info(f'Solving {spec} {key}.')

    tic = time.perf_counter()
    lb_infid, lb_propagator = lmt.infidelity_lindblad(pulse, S_0, identifiers,
                                                      show_progressbar=True,
                                                      return_propagator=True)
    toc = time.perf_counter()

    sys.stdout.flush()
    lb_logger.info(f'Took {toc-tic:.1e} s.')
    lb_logger.info(f'Infidelity is {lb_infid:.4e}.')

    propagators_lindblad[spec][key] = lb_propagator

    file_name = f'lindblad_propagator_{key}_run_1.npy'
    while (save_path / file_name).exists():
        file_name = increase_file_counter(save_path, file_name, '_run_')

    np.save(save_path / file_name, lb_propagator)
    lb_logger.info(f'Saved propagators at {save_path / file_name}.')

    return lb_infid, lb_propagator


# %% Set up logging
log_file = "run_1.log"
while (save_path / log_file).exists():
    log_file = increase_file_counter(save_path, log_file, 'run_')

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] in %(filename)s:%(funcName)s: %(message)s",
    handlers=[
        logging.FileHandler(save_path / log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

mc_logger = logging.getLogger('MC')
lb_logger = logging.getLogger('LB')
ff_logger = logging.getLogger('FF')
# %% Git hashes
repos = {package: git.Repo(git_path / package) for package in
         ('filter_functions', 'qopt', 'qutil', 'lindblad_mc_tools')}
sha = {package: repo.head.object.hexsha for package, repo in repos.items()}
with open(save_path / 'versions.json', 'w', encoding='utf-8') as file:
    json.dump(sha, file)
    logging.info(f"Dumped sha's to file {file.name}.")
shutil.copyfile(thisfile, save_path / thisfile.name)
logging.info(f"Copied this file to {save_path / thisfile.name}.")

# %% Run all calculations
propagators_mc = dict(white=dict(), pink=dict())
propagators_lindblad = dict(white=dict())
infids_ff = dict(white=dict(), pink=dict())
decay_amplitudes = dict(white=dict(), pink=dict())
frequency_shifts = dict(white=dict(), pink=dict())
cumulant_functions = dict(white=dict(), pink=dict())
error_transfer_matrices = dict(white=dict(), pink=dict())
for key in ['noecho', 'echo']:
    for spec in ['pink', 'white']:
        alpha = 0.0 if spec == 'white' else -1.0

        def spectrum_angular(omega):
            return spectrum(omega, S_0[spec], omega_A=1, alpha=alpha)

        def spectrum_real(f):
            return spectrum(2*np.pi*f, S_0[spec], omega_A=1, alpha=alpha)

        if spec == 'pink':
            if _compute_mc:
                propagators_mc[spec][key], sampler = run_mc(pulses[key], spectrum_real, f, n_MC,
                                                            identifiers,
                                                            max_frequency_sample_spacing,
                                                            chunk_noise=_chunk_noise,
                                                            show_progressbar=_show_progressbar,
                                                            threads=_threads, spec=spec, key=key)
            else:
                mc_logger.info(f'Skipped {spec} {key}.')

            if _compute_ff:
                ff_logger.debug("Control matrix cached: " +
                                f"{pulses[key].is_cached('control matrix')}")
                infids_ff[spec][key] = run_ff(pulses[key], spectrum_angular, omega, identifiers,
                                              show_progressbar=_show_progressbar,
                                              compute_process=_compute_ff_process,
                                              spec=spec, key=key)
            else:
                ff_logger.info(f'Skipped {spec} {key}.')

            if _compute_ff_process:
                ff_logger.debug("Control matrix cached: " +
                                f"{pulses[key].is_cached('control matrix')}")
                ff_logger.debug("Second order cached: " +
                                f"{pulses[key].is_cached('second order filter function')}")
                results = run_ff_process(pulses[key], spectrum_angular, omega, identifiers,
                                         show_progressbar=_show_progressbar, spec=spec, key=key,
                                         DC=False)
                decay_amplitudes[spec][key] = results[0]
                frequency_shifts[spec][key] = results[1]
                cumulant_functions[spec][key] = results[2]
                error_transfer_matrices[spec][key] = results[3]
            else:
                ff_logger.info(f'Skipped process {spec} {key}.')

        if spec == 'white':
            if _compute_mc:
                propagators_mc[spec][key], sampler = run_mc(pulses[key], spectrum_real, f_all,
                                                            n_MC, identifiers,
                                                            max_frequency_sample_spacing,
                                                            chunk_noise=_chunk_noise,
                                                            show_progressbar=_show_progressbar,
                                                            threads=_threads, spec=spec, key=key)
            else:
                mc_logger.info(f'Skipped {spec} {key}.')

            if _compute_ff:
                infids_ff[spec][key] = run_ff_dc([x[key] for x in (pulses_dc, pulses, pulses_all)],
                                                 spectrum_angular, (omega_dc, omega, omega_all),
                                                 identifiers, show_progressbar=_show_progressbar,
                                                 compute_process=_compute_ff_process,
                                                 spec=spec, key=key)
            else:
                ff_logger.info(f'Skipped DC {spec} {key}.')

            if _compute_ff_process:
                ff_logger.debug("Control matrix cached: " +
                                f"{pulses_all[key].is_cached('control matrix')}")
                ff_logger.debug("Second order cached: " +
                                f"{pulses_all[key].is_cached('second order filter function')}")
                _ = run_ff_process(pulses_all[key], spectrum_angular, omega_all, identifiers,
                                   show_progressbar=_show_progressbar, spec=spec, key=key, DC=True)
            else:
                ff_logger.info(f'Skipped process DC {spec} {key}.')

            if _compute_lb:
                lb_infid, lb_propagator = run_lb(pulses[key], S_0[spec], identifiers,
                                                 show_progressbar=_show_progressbar,
                                                 spec=spec, key=key)
            else:
                lb_logger.info(f'Skipped {spec} {key}.')

    save_pulse(pulses_dc[key], pathlib.Path(save_path / rf'qft_pulse_dc_{key}_with_FF'))
    save_pulse(pulses[key], pathlib.Path(save_path / rf'qft_pulse_{key}_with_FF'))
    save_pulse(pulses_all[key], pathlib.Path(save_path / rf'qft_pulse_all_{key}_with_FF'))

# %% Evaluation

if _compute_mc:
    infids = dict(white={}, pink={})
    shape = dict(white={}, pink={})
    loc = dict(white={}, pink={})
    scale = dict(white={}, pink={})

    fig, ax = plt.subplots(2, 2, sharex=False, sharey=True, constrained_layout=True)
    for i, (n, tmp) in enumerate(propagators_mc.items()):
        for j, ((p, props), pulse) in enumerate(zip(tmp.items(), (qft_pulse, qft_pulse_echo))):
            infids[n][p] = 1 - abs((pulse.total_propagator.conj().T @ props
                                    ).trace(0, 1, 2)/pulse.d)**2
            shape[n][p], loc[n][p], scale[n][p] = stats.gamma.fit(infids[n][p], floc=0)
            if shape[n][p] < 1:
                shape[n][p], loc[n][p], scale[n][p] = stats.gamma.fit(infids[n][p], floc=0, f0=1.0)

            sp = (stats.gamma.ppf(stats.norm.cdf(2), shape[n][p], loc[n][p], scale[n][p]))
            sm = (stats.gamma.ppf(stats.norm.cdf(-2), shape[n][p], loc[n][p], scale[n][p]))
            se = stats.gamma.std(shape[n][p], loc[n][p], scale[n][p])/np.sqrt(infids[n][p].size)
            se = infids[n][p].std()/np.sqrt(infids[n][p].size)
            # print(f'{infids[n][p].mean():.2e} - {sm:.2e} + {sp:.2e}')
            print(f'{n} {p}\t{infids[n][p].mean():.2e} ± {se:.2e}')

            x = np.linspace(0, stats.gamma.ppf(.999, shape[n][p], loc[n][p], scale[n][p]), 500)
            y = stats.gamma.pdf(x, shape[n][p], loc[n][p], scale[n][p])
            ax[i, j].hist(infids[n][p], bins='auto', density=True, log=True, label='data',
                          histtype='step', range=(0, infids[n][p].max()))
            ax[i, j].plot(x, y, color='tab:red', label='fit')
            # ax[i, j].axvline(infids[n][p].mean(), color='black')
            ax[i, j].axvline(sm, color='grey', ls='--', zorder=1)
            ax[i, j].axvline(sp, color='grey', ls='--', zorder=1, label=r'2σ')
            # ax[i, j].set_xlim(x[0], x[-1])

            ylim = [1, 1e3]
            ax[i, j].set_ylim(ylim)
            ax[i, j].fill_betweenx(ylim, infids[n][p].mean() - se, infids[n][p].mean() + se,
                                   alpha=0.2, color='black', zorder=10, label='mean')
            ax[i, j].set_title(' '.join([n, p]))

            ax[i, j].set_xscale('log')

    ax[i, j].legend()
