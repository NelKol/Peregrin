# app.py
# pip install shiny plotnine pandas numpy

import numpy as np
import pandas as pd
from shiny import App, ui, render
from plotnine import ggplot, aes, geom_point, theme_minimal, labs

rng = np.random.default_rng(0)

# --- Build consistent demo data (variable number of spots per track) ---
track_ids = np.arange(1001, 1021)                     # 20 tracks
counts = rng.integers(8, 18, size=track_ids.size)      # 8..17 spots per track
index = np.repeat(track_ids, counts)                   # length N = counts.sum()
N = index.size

x = rng.random(N)                                      # length N
y = rng.random(N)                                      # length N
assert len(index) == len(x) == len(y), "Length mismatch!"

df = pd.DataFrame({"TrackIndex": index, "x": x, "y": y}).set_index("TrackIndex")

# --- UI ---
app_ui = ui.page_fluid(
    ui.h3("Array plot — brush to select"),
    ui.output_plot(
        "plot",
        height="420px",
        brush=ui.brush_opts(direction="xy")
    ),
    ui.hr(),
    ui.h4("Brushed rows"),
    ui.output_table("table")
)

# --- Server ---
def server(input, output, session):
    @output
    @render.plot
    def plot():
        # plotnine wants columns, so reset_index for plotting
        return (
            ggplot(df.reset_index(), aes("x", "y"))
            + geom_point(alpha=0.7, size=2)
            + theme_minimal()
            + labs(x="x metric", y="y metric", title="Brush to select points")
        )

    @output
    @render.table
    def table():
        b = input.plot_brush()
        if not b:
            return pd.DataFrame(columns=["TrackIndex", "x", "y"])
        xmin, xmax = (b["xmin"], b["xmax"]) if isinstance(b, dict) else (b.xmin, b.xmax)
        ymin, ymax = (b["ymin"], b["ymax"]) if isinstance(b, dict) else (b.ymin, b.ymax)

        sel = df[(df["x"] >= xmin) & (df["x"] <= xmax) & (df["y"] >= ymin) & (df["y"] <= ymax)]
        return sel.reset_index()

app = App(app_ui, server)
