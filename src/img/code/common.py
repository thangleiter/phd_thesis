
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
