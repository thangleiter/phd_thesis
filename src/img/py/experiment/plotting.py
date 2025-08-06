import json
import pprint

import matplotlib.pyplot as plt
from mjolnir import plotting
from qcodes.dataset import DataSetProtocol, initialise_or_create_database_at, load_by_run_spec
from qutil import functools, misc
from qutil.plotting import BlitManager


def _cleanup_figs(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        nums = set(plt.get_fignums())
        try:
            return func(*args, **kwargs)
        except Exception:
            if new := nums.symmetric_difference(plt.get_fignums()):
                for num in new:
                    plt.close(num)
            raise

    return wrapped


@functools.wraps(plotting.plot_nd)
def plot_nd(ds_or_id, **kwargs):
    if isinstance(ds_or_id, int):
        ds = load_by_run_spec(captured_run_id=ds_or_id)
    else:
        ds = ds_or_id
    if isinstance(ds, DataSetProtocol):
        with misc.filter_warnings(action='ignore', category=UserWarning):
            ds = ds.to_xarray_dataset()

    if kwargs.pop('verbose', True):
        cm = json.loads(ds.attrs['custom_metadata'])
        print(s := f'Run {ds.run_id}: {ds.attrs["ds_name"]}')
        print('='*len(s))
        if mis := cm.get('measurement_initialization_settings', False):
            print('\nMeasurement initialization settings')
            print('-'*len(s))
            pprint.pprint(mis)
        if mpc := cm.get('measurement_parameter_contexts', False):
            print('\nMeasurement parameter contexts')
            print('-'*len(s))
            pprint.pprint(mpc)

    return plotting.plot_nd(ds, **kwargs), ds


def browse_db(initial_run_id: int = 1, db: str | None = None, **plot_nd_kwargs):
    if db is not None:
        initialise_or_create_database_at(db)

    fig_md, ax_md = plt.subplots(layout='constrained')
    ax_md.axis('off')
    txt = ax_md.text(0.05, 0.5, '', fontsize=12, ha='left', va='center', transform=ax_md.transAxes,
                     family='monospace')
    bm = BlitManager(fig_md.canvas, [txt])

    while True:
        try:
            fig, ax, sliders = _open_figure(initial_run_id, bm, txt,
                                            **(plot_nd_kwargs | {'verbose': False}))
        except (ValueError, IndexError, TypeError):
            initial_run_id += 1
        else:
            break

    print('DB browser:')
    print('===========')
    print('Navigate runs using left and right arrow keys, or type a run ID with the figure in '
          'focus and confirm with enter.')
    print('Pass keyword arguments through to mjolnir.plotting.plot_nd() to configure the plot.')
    print('A secondary window opens up that displays some metadata of the currently selected run.')
    print()


def print_params(ds, voltages=True, wavelength=True, power=True, tex=False):
    print('Measurement:', ds.ds_name)
    s = json.loads(ds.attrs['snapshot'])
    try:
        ep = s['station']['instruments']['excitation_path']['parameters']
        sample = s['station']['instruments'][ds.sample_name]
    except KeyError:
        print('No snapshot.')
        return

    active_trap = sample['parameters']['active_trap']['value']
    if isinstance(active_trap, int):
        active_trap = sample['name'] + f'_trap_{active_trap}'
    elif len(parts := active_trap.split(',')) > 1:
        active_trap = parts[0].lstrip('Trap(name=')
    else:
        active_trap = active_trap.split()[1]
    gates = sample['submodules']['traps']['channels'][active_trap]['parameters']
    if all(gate.startswith(active_trap) for gate in gates):
        prefix = f'{active_trap}_'
    else:
        prefix = ''

    if voltages:
        for typ in ('guard', 'central'):
            for mode in ('difference_mode', 'common_mode'):
                if f'{prefix}{typ}_{mode}' in gates:
                    if tex:
                        print(r'\thevoltage{', end='')
                    else:
                        print(f'Trap {active_trap} {typ} {mode.replace("_", " ")}: ', end='')
                    print(f"{gates[f'{prefix}{typ}_{mode}']['value']:.2f}",
                          end='' if tex else '\n')
                    if tex:
                        print('}{' + typ[0] + ''.join(m[0] for m in mode.split('_')) + '}')
    if wavelength:
        if tex:
            print(r'\thewavelength{', end='')
        else:
            print('Excitation wavelength: ', end='')
        print(f"{ep['wavelength']['value']:.1f}", end='' if tex else '\n')
        if tex:
            print('}')
    if power:
        if tex:
            print(r'\thepower{', end='')
        else:
            print('Excitation power at sample: ', end='')
        print(f"{ep['power_at_sample']['value']*1e6:.2g}", end='' if tex else '\n')
        if tex:
            print(r'}{\micro}')


@_cleanup_figs
def _open_figure(run_id, bm, txt, **plot_nd_kwargs):
    input_str = ""

    def on_key(event):
        nonlocal input_str, run_id, fig, ax, sliders
        if event.key == 'left':
            run_id -= 1
        elif event.key == 'right':
            run_id += 1
        elif event.key == 'enter':
            run_id = int(input_str)
            input_str = ""
        elif event.key == 'escape':
            input_str = ""
            return
        elif event.key.isdigit():
            input_str += event.key
            return
        else:
            return

        fig2 = fig
        try:
            print(f'Run #{run_id}: Plotting... ', end='')
            fig, ax, sliders = _open_figure(run_id, bm, txt, **plot_nd_kwargs)
        except (ValueError, IndexError, TypeError):
            print('Failed.')
        except NameError:
            print('End of db reached.')
            run_id = run_id - 1 if event.key == 'right' else run_id + 1
        except Exception as err:
            raise RuntimeError("Unexpected exception") from err
        else:
            print()
            fig2.canvas.mpl_disconnect(cid)
            plt.close(fig2)

    (fig, ax, sliders), ds = plot_nd(run_id, **plot_nd_kwargs)
    fig.canvas.mpl_connect('key_press_event', on_key)
    _update_text(ds, txt, bm)
    cid = fig.canvas.mpl_connect('close_event', lambda event: plt.close(bm.canvas.figure))
    return fig, ax, sliders


def _update_text(ds, txt, bm):
    s = (t := f'Run {ds.run_id}: {ds.attrs["ds_name"]}')
    s += '\n' + '='*len(t)
    s += f'\nRun timestamp: {ds.attrs["run_timestamp"]}\n'

    if cm := json.loads(ds.attrs.get('custom_metadata', '')):
        if mis := cm.get('measurement_initialization_settings', False):
            s += ('\nMeasurement initialization settings'
                  + '\n' + '-'*len(t) + '\n'
                  + pprint.pformat(mis))
        if mpc := cm.get('measurement_parameter_contexts', False):
            s += ('\n\nMeasurement parameter contexts'
                  + '\n' + '-'*len(t) + '\n'
                  + pprint.pformat(mpc))
    if c := json.loads(ds.attrs.get('comment', '""')):
        s += ('\n\nComment'
              + '\n' + '-'*len(t) + '\n'
              + pprint.pformat(c))

    txt.set_text(s)
    bm.update()
