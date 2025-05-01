import pathlib

TEXTWIDTH = 4.2134
MARGINWIDTH = 1.87831
TOTALWIDTH = TEXTWIDTH + TEXTWIDTH + 0.24414
PATH = pathlib.Path(__file__).parents[1]
MAINSTYLE = PATH / 'py/main.mplstyle'
MARGINSTYLE = PATH / 'py/margin.mplstyle'
MARKERSTYLE = PATH / 'py/markers.mplstyle'


def apply_sketch_style(ax):
    ax.spines.left.set_position(("data", 0))
    ax.spines.bottom.set_position(("data", 0))
    ax.spines.left.set_zorder(0.5)
    ax.spines.bottom.set_zorder(0.5)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    ax.tick_params(direction='inout')
    ax.set_xlabel(ax.get_xlabel(), rotation='horizontal')
    ax.set_ylabel(ax.get_ylabel(), rotation='horizontal')
