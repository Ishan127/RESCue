import os
import matplotlib.pyplot as plt


def show_or_save(fig, out_dir: str, filename: str, mode: str = "save"):
    """
    Display or save a matplotlib figure.
    mode:
      - 'display': try IPython display; fallback to save
      - 'save': save to out_dir/filename
      - 'off': do nothing
    Always closes the figure to avoid printing repr like 'Figure(1200x400)'.
    """
    try:
        if mode == "off":
            return

        if mode == "display":
            try:
                from IPython.display import display
                display(fig)
                return
            except Exception:
                # fall back to save if display fails (e.g., subprocess)
                mode = "save"

        if mode == "save":
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, filename)
            fig.savefig(path, bbox_inches='tight')
            print(f"Saved visualization: {path}")
    finally:
        plt.close(fig)
