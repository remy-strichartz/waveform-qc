"""Shared matplotlib helpers for the preprocessing diagnostics scripts
(hodoscope_efficiency, pulse_window, waveform_triage, channel_diagnostics)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("preprocessing.plotting")


def setup_mpl(show: bool):
    """Import and return pyplot, forcing the non-interactive Agg backend when the
    figures will only be saved (no plot windows are wanted)."""
    import matplotlib
    if not show:
        try:
            matplotlib.use("Agg", force=True)   # works even if pyplot is already imported
        except Exception:
            pass
    import matplotlib.pyplot as plt
    return plt


def finish_figure(fig, out_path=None, show: bool = False, dpi: int = 200) -> None:
    """Save the figure (when `out_path` is given), show it (when interactive),
    then close it so figures never accumulate across a multi-plot run."""
    import matplotlib.pyplot as plt
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved %s", out_path)
    if show:
        plt.show()
    plt.close(fig)


def paged_figure(n_pages: int, draw, figsize=(13, 9), extra_keys: dict | None = None) -> None:
    """Show an interactive figure whose pages are rendered by ``draw(fig, page)``.

    LEFT / RIGHT (or n / p) page through; Esc / Q closes.  `extra_keys` maps a
    key to a zero-argument callback (the page is redrawn after it runs) for view
    toggles such as the galleries' raw-ADC 'r' key.  This is the project's ONE
    interactive browser: the triage / pulse_window galleries and every
    channel_diagnostics paged plot share these controls, so no plot ever crams
    everything onto one page."""
    import matplotlib.pyplot as plt
    state = {"page": 0}
    fig = plt.figure(figsize=figsize)

    def render():
        fig.clf(); draw(fig, state["page"]); fig.canvas.draw_idle()

    def on_key(e):
        if e.key in ("right", "n"):
            state["page"] = (state["page"] + 1) % n_pages; render()
        elif e.key in ("left", "p"):
            state["page"] = (state["page"] - 1) % n_pages; render()
        elif extra_keys and e.key in extra_keys:
            extra_keys[e.key](); render()
        elif e.key in ("escape", "q"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render(); plt.show(); plt.close(fig)
