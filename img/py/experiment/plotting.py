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
        print('\nMeasurement initialization settings')
        print('-'*len(s))
        pprint.pprint(cm['measurement_initialization_settings'])
        print('\nMeasurement parameter contexts')
        print('-'*len(s))
        pprint.pprint(cm['measurement_parameter_contexts'])

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


@_cleanup_figs
def _open_figure(run_id, bm, txt, **plot_nd_kwargs):

    def on_key(event):
        nonlocal run_id, fig, ax, sliders
        if event.key == 'left':
            run_id -= 1
        elif event.key == 'right':
            run_id += 1
        else:
            return

        fig2 = fig
        try:
            print(f'Run #{run_id}: plotting.')
            fig, ax, sliders = _open_figure(run_id, bm, txt, **plot_nd_kwargs)
        except (ValueError, IndexError, TypeError):
            print(' '*(7 + len(str(run_id))) + 'failed.')
        except Exception as err:
            raise RuntimeError("Unexpected exception") from err
        else:
            fig2.canvas.mpl_disconnect(cid)
            plt.close(fig2)

    (fig, ax, sliders), ds = plot_nd(run_id, **plot_nd_kwargs)
    fig.canvas.mpl_connect('key_press_event', on_key)
    _update_text(ds, txt, bm)
    cid = fig.canvas.mpl_connect('close_event', lambda event: plt.close(bm.canvas.figure))
    return fig, ax, sliders


def _update_text(ds, txt, bm):
    s = (t := f'Run {ds.run_id}: {ds.attrs["ds_name"]}')
    if cm := json.loads(ds.attrs.get('custom_metadata', '')):
        if mis := cm.get('measurement_initialization_settings', False):
            s += ('\n' + '='*len(t) + '\n'
                  + '\nMeasurement initialization settings'
                  + '\n' + '-'*len(t) + '\n'
                  + pprint.pformat(mis))
        if mpc := cm.get('measurement_parameter_contexts', False):
            s += ('\n\nMeasurement parameter contexts'
                  + '\n' + '-'*len(t) + '\n'
                  + pprint.pformat(mpc))

    txt.set_text(s)
    bm.update()
