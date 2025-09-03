from shiny import App, Inputs, Outputs, Session, render, reactive, req, ui
from shiny.types import FileInfo
from shinywidgets import render_plotly, render_altair, output_widget, render_widget, register_widget
from shiny.plotutils import brushed_points, near_points

from utils.Select import Metrics, Styles, Markers, Modes
from utils.Function import DataLoader, Process, Calc, Threshold
from utils.ratelimit import debounce, throttle
from utils.Customize import Format, Brush

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objs as go
from plotnine import ggplot, aes, geom_point, theme_minimal, theme, element_blank, element_line, scale_x_continuous, scale_y_continuous, coord_equal, labs


# --- UI definition ---
app_ui = ui.page_sidebar(

    # ========== SIDEBAR - DATA FILTERING ==========
    ui.sidebar(
        ui.tags.style(Format.Accordion),
        ui.markdown("""  <p>  """),
        ui.output_ui("sidebar_label"),
        ui.input_action_button("add_threshold", "Add threshold", class_="btn-primary"),
        ui.input_action_button("remove_threshold", "Remove threshold", class_="btn-primary", disabled=True),
        ui.output_ui("sidebar_accordion"),
        ui.input_task_button("apply_threshold", label="Apply thresholding", label_busy="Applying...", type="secondary"),
        id="sidebar", open="closed", position="right", bg="f8f8f8",
    ),

    # ========== MAIN NAVIGATION BAR ==========
    ui.navset_bar(
        # ========== RAW DATA INPUT - ANALYSIS INITIALIZATION ==========    
        ui.nav_panel(
            "Input",
            ui.div(
                {"id": "data-inputs"},
                # Action buttons
                ui.input_action_button("add_input", "Add data input", class_="btn-primary"),
                ui.input_action_button("remove_input", "Remove data input", class_="btn-primary", disabled=True),
                ui.input_action_button("run", label="Run", class_="btn-secondary", disabled=True),
                ui.input_action_button("reset", "Reset", class_="btn-danger"),
                ui.input_action_button("input_help", "Show help"),
                ui.output_ui("initialize"),
                ui.markdown("""___"""),
                # File inputs
                ui.row(
                    ui.column(4, ui.output_ui("input_file_pairs")),
                ),
                # Assigning selected columns - draggable window
                ui.panel_absolute(
                    ui.panel_well(
                        ui.markdown("<h5>Select columns:</h5>"),
                        ui.input_selectize("select_id", "Track identifier:", ["e.g. TRACK_ID"]),
                        ui.input_selectize("select_time", "Time point:", ["e.g. POSITION_T"]),
                        ui.input_selectize("select_x", "X coordinate:", ["e.g. POSITION_X"]),
                        ui.input_selectize("select_y", "Y coordinate:", ["e.g. POSITION_Y"]),
                        ui.markdown("<span style='color:darkgrey; font-style:italic;'>You can drag me around!</span>")
                    ),
                    width="350px", right="315px", top="220px", draggable=True
                ),
            ),
        ),

        # ========== PROCESSED DATA DISPLAY ==========
        ui.nav_panel(
            "Data frames",
            ui.markdown(""" <p> """),

            # Input for already processed data
            ui.input_file("already_proccesed_input", "Got previously processed data?", placeholder="Drag and drop here!", accept=[".csv"], multiple=False),
            ui.markdown(""" ___ """),

            ui.column(6, ui.tags.b("Points in brush"), ui.output_table("in_brush")),

            # Data frames display
            ui.layout_columns(
                ui.card( # Spot stats
                    ui.card_header("Spot stats"),
                    ui.output_data_frame("render_spot_stats"),
                    ui.download_button("download_spot_stats", "Download CSV"),
                ),
                ui.card( # Track stats
                    ui.card_header("Track stats"),
                    ui.output_data_frame("render_track_stats"),
                    ui.download_button("download_track_stats", "Download CSV"),
                ),
                ui.card( # Time stats
                    ui.card_header("Time stats"),
                    ui.output_data_frame("render_time_stats"),
                    ui.download_button("download_time_stats", "Download CSV"),
                ),
            ),
        ),

        # ========== VISUALIZATION PANEL ==========
        ui.nav_panel(
            "Visualisation",

            ui.navset_pill_list(
            
                ui.nav_panel(
                    "Tracks",
                    # Interactive settings
                    ui.panel_well(
                        ui.markdown(
                            """
                            #### **Track visualization**
                            *made with*  `plotly`
                            <hr style="height: 4px; background-color: black; border: none" />
                            """
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                "Dataset",
                                ui.input_selectize("condition_tracks", "Condition:", ["all", "not all"]),
                                ui.panel_conditional(
                                    "input.condition_tracks != 'all'",
                                    ui.input_selectize("replicate_tracks", "Replicate:", []),
                                ),
                            ),
                            ui.accordion_panel(
                                "Track lines",
                                ui.input_checkbox("show_tracks", "Show tracks", True),
                                ui.panel_conditional(
                                    "input.show_tracks",
                                    ui.input_numeric("smoothing", "Smoothing index:", 0),
                                    ui.input_numeric('track_line_width', 'Line width:', 0.85),
                                ),
                            ),
                            ui.accordion_panel(
                                "Markers",
                                ui.input_checkbox("show_markers", "Show end track markers", True),
                                ui.panel_conditional(
                                    "input.show_markers",
                                    ui.panel_conditional(
                                        "input.basic",
                                        ui.input_selectize("basic_markers", "Markers:", Markers.PlotlyOpen),
                                        ui.input_numeric("basic_marker_size", "Marker size:", 5),
                                    ),
                                    ui.panel_conditional(
                                        "input.basic == false",
                                        ui.input_selectize("not_basic_markers", "Markers:", Markers.Emoji),
                                        ui.input_numeric("not_basic_marker_size", "Marker size:", 5),
                                    ),
                                    ui.input_switch("basic", "Basic", True),
                                ),
                            ),
                            ui.accordion_panel(
                                "Coloring",
                                ui.input_selectize("color_mode", "Color mode:", Styles.ColorMode),
                                ui.panel_conditional(
                                    "input.color_mode != 'random greys' && input.color_mode != 'random colors' && input.color_mode != 'only-one-color' && input.color_mode != 'differentiate conditions/replicates'",
                                    ui.input_selectize('lut_scaling', 'LUT scaling metric:', []),
                                ),
                                ui.panel_conditional(
                                    "input.color_mode == 'only-one-color'",
                                    ui.input_selectize('only_one_color', 'Color:', Styles.Color),
                                ),
                                ui.input_selectize('background', 'Background:', Styles.Background),
                                ui.input_checkbox("show_gridlines", "Gridlines", True),
                            ),
                            ui.accordion_panel(
                                "Hover info",
                                ui.input_selectize("let_me_look_at_these", "Let me look at these:", ["Condition", "Track length", "Net distance", "Speed mean"], multiple=True),
                                ui.input_action_button("hover_info", "See info"),
                            ),
                        ),
                    ),
                    ui.markdown(""" <p> """),
                    # Plotly outputs
                    ui.card(
                        ui.output_plot("plotly_true_visualization"),
                        ui.download_button("download_plotly_true_visualization_html", "Download HTML"),
                        ui.download_button("download_plotly_true_visualization_svg", "Download SVG"),
                    ),
                    ui.card(
                        ui.output_plot("plotly_spiderplot"),
                        ui.download_button("download_plotly_spiderplot_html", "Download HTML"),
                        ui.download_button("download_plotly_spiderplot_svg", "Download SVG"),
                    ),
                    ui.card(
                        ui.download_button("download_lut_map_svg", "Download LUT Map SVG"),
                    ),
                ),


                ui.nav_panel(
                    "Time charts",
                    ui.panel_well(
                        ui.markdown(
                            """
                            #### **Time series charts**
                            *made with*  `altair`
                            <hr style="height: 4px; background-color: black; border: none" />
                            """
                        ),
                        
                        ui.input_select("tch_plot", "Plot:", choices=["Scatter", "Line", "Error band"]),

                        ui.accordion(
                            ui.accordion_panel(
                                "Dataset",
                                ui.input_selectize("tch_condition", "Condition:", ["all", "not all"]),
                                ui.panel_conditional(
                                    "input.tch_condition != 'all'",
                                    ui.input_selectize("tch_replicate", "Replicate:", ["all", "not all"]),
                                    ui.panel_conditional(
                                        "input.tch_replicate == 'all'",
                                        ui.input_checkbox("time_separate_replicates", "Show replicates separately", False),
                                    ),
                                ),
                            ),

                            ui.accordion_panel(
                                "Metric",
                                ui.input_selectize("tch_metric", label=None, choices=Metrics.Time, selected="Confinement ratio mean"),
                                ui.input_radio_buttons("y_axis", "Y axis with", ["Absolute values", "Relative values"]),
                            ),

                            ui.accordion_panel(
                                "Plot settings",

                                ui.panel_conditional(
                                    "input.tch_plot == 'Scatter'",

                                    ui.accordion(
                                        ui.accordion_panel(
                                            "Central tendency",
                                            ui.input_selectize("tch_scatter_central_tendency", label=None, choices=["Mean", "Median"], selected=["Median"]),
                                        ),
                                        ui.accordion_panel(
                                            "Polynomial fit",
                                            ui.row(
                                                ui.column(3, ui.input_checkbox("tch_polynomial_fit", "Fit", True)),
                                                ui.panel_conditional(
                                                    "input.tch_polynomial_fit == true",
                                                    ui.input_switch("tch_fit_best", "Automatic fit", True),
                                                ),
                                            ),
                                            ui.panel_conditional(
                                                "input.tch_polynomial_fit == true",
                                                ui.panel_conditional(
                                                    "input.tch_fit_best == false",
                                                    ui.input_selectize("tch_force_fit", "Fit:", list(Modes.FitModel.keys())),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),

                                ui.panel_conditional(
                                    "input.tch_plot == 'Line'",
                                    ui.input_selectize("tch_line_interpolation", "Interpolation:", choices=Modes.Interpolate),
                                ),

                                ui.panel_conditional(
                                    "input.tch_plot == 'Error band'",
                                    ui.input_selectize("tch_errorband_error", "Error:", Modes.ExtentError),
                                    ui.input_selectize("tch_errorband_interpolation", "Interpolation:", Modes.Interpolate),
                                ),
                            ),

                            ui.accordion_panel(
                                "Aesthetics",

                                ui.panel_conditional(
                                    "input.tch_plot == 'Scatter'",

                                    ui.accordion(
                                        ui.accordion_panel(
                                            "Coloring",
                                            ui.input_selectize("tch_scatter_background", "Background:", Styles.Background),
                                            ui.input_selectize("tch_scatter_color_palette", "Color palette:", Styles.PaletteQualitative),
                                        ),

                                        ui.accordion_panel(
                                            "Elements",
                                            # Bullet settings
                                            ui.accordion(
                                                ui.accordion_panel(
                                                    "Bullets",
                                                    ui.input_checkbox("tch_show_scatter", "Show scatter", True),
                                                    ui.panel_conditional(
                                                        "input.tch_show_scatter == true",
                                                        ui.input_numeric("tch_bullet_opacity", "Opacity:", 0.5, min=0, max=1, step=0.1),
                                                        ui.input_checkbox("tch_fill_bullets", "Fill bullets", True),
                                                        ui.input_numeric("tch_bullet_size", "Bullet size:", 5, min=0, step=0.5),
                                                        ui.panel_conditional(
                                                            "input.tch_fill_bullets == true",
                                                            ui.input_checkbox("tch_outline_bullets", "Outline bullets", False),
                                                            ui.panel_conditional(
                                                                "input.tch_outline_bullets == true",
                                                                ui.input_selectize("tch_bullet_outline_color", "Outline color:", Styles.Color + ["match"], selected="match"),
                                                                ui.input_numeric("tch_bullet_outline_width", "Outline width:", 1, min=0, step=0.1),
                                                            ),
                                                        ),
                                                        ui.panel_conditional(
                                                            "input.tch_fill_bullets == false",
                                                            ui.input_numeric("tch_bullet_stroke_width", "Stroke width:", 1, min=0, step=0.1),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                            # Line settings
                                            ui.panel_conditional(
                                                "input.tch_polynomial_fit == true",
                                                ui.markdown("""  <br>  """),
                                                ui.input_numeric("tch_scatter_line_width", "Line width:", 1, min=0, step=0.1),
                                            ),
                                        ),
                                    ),
                                ),

                                ui.panel_conditional(
                                    "input.tch_plot == 'Line'",
                                    ui.accordion(
                                        ui.accordion_panel(
                                            "Coloring",
                                            ui.input_selectize("tch_line_background", "Background:", Styles.Background),
                                            ui.input_selectize("tch_line_color_palette", "Color palette:", Styles.PaletteQualitative),
                                        ),
                                        ui.accordion_panel(
                                            "Elements",
                                            # Line settings
                                            ui.markdown("""  <p>  """),
                                            ui.input_numeric("tch_line_line_width", "Line width:", 1, min=0, step=0.1),
                                            # Bullets settings
                                            ui.markdown("""  <br>  """),
                                            ui.input_checkbox("tch_line_show_bullets", "Show bullets", False),
                                            ui.panel_conditional(
                                                "input.tch_line_show_bullets == true",
                                                ui.input_numeric("tch_line_bullet_size", "Bullet size:", 1, min=0, step=0.5),
                                            ),
                                        ),
                                    ),
                                ),

                                ui.panel_conditional(
                                    "input.tch_plot == 'Error band'",
                                    ui.accordion(
                                        ui.accordion_panel(
                                            "Coloring",
                                            ui.input_selectize("tch_errorband_background", "Background:", Styles.Background),
                                            ui.input_selectize("tch_errorband_color_palette", "Color palette:", Styles.PaletteQualitative),
                                        ),
                                    
                                        ui.accordion_panel(
                                            "Bands",
                                            # Error band settings
                                            ui.input_checkbox("tch_errorband_fill", "Fill area", True),
                                            ui.panel_conditional(
                                                "input.tch_errorband_fill == true",
                                                ui.input_numeric("tch_errorband_fill_opacity", "Fill opacity:", 0.5, min=0, max=1, step=0.1),
                                                ui.input_checkbox("tch_errorband_outline", "Outline area", False),
                                            ),
                                            ui.panel_conditional(
                                                "input.tch_errorband_fill == true && input.tch_errorband_outline == true || input.tch_errorband_fill == false",
                                                ui.input_numeric("tch_errorband_outline_width", "Outline width:", 1, min=0, step=0.1),
                                                ui.input_selectize("tch_errorband_outline_color", "Outline color:", Styles.Color + ["match"], selected="match"),
                                            ),
                                        ),
                                        ui.accordion_panel(
                                            "Lines",
                                            ui.accordion(
                                                ui.accordion_panel(
                                                    "Mean",
                                                    ui.input_checkbox("tch_errorband_show_mean", "Show mean", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_mean == true",
                                                        ui.input_selectize("tch_errorband_mean_line_color", "Line color:", Styles.Color + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_mean_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_mean_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.accordion_panel(
                                                    "Median",
                                                    ui.input_checkbox("tch_errorband_show_median", "Show median", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_median == true",
                                                        ui.input_selectize("tch_errorband_median_line_color", "Line color:", Styles.Color + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_median_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_median_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.accordion_panel(
                                                    "Min",
                                                    ui.input_checkbox("tch_errorband_show_min", "Show min", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_min == true",
                                                        ui.input_selectize("tch_errorband_min_line_color", "Line color:", Styles.Color + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_min_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_min_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.accordion_panel(
                                                    "Max",
                                                    ui.input_checkbox("tch_errorband_show_max", "Show max", False),
                                                    ui.panel_conditional(
                                                        "input.tch_errorband_show_max == true",
                                                        ui.input_selectize("tch_errorband_max_line_color", "Line color:", Styles.Color + ["match"], selected="match"),
                                                        ui.input_selectize("tch_errorband_max_line_style", "Line style:", Styles.LineStyle),
                                                        ui.input_numeric("tch_errorband_max_line_width", "Line width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    ui.markdown(""" <p> """),
                    ui.card(
                        ui.output_plot("time_series_poly_fit_chart"),
                        ui.download_button("download_time_series_poly_fit_chart__html", "Download Time Series Poly Fit Chart HTML"),
                        ui.download_button("download_time_series_poly_fit_chart_svg", "Download Time Series Poly Fit Chart SVG"),
                    ),
                    # ... more cards for line chart and errorband
                ),
                ui.nav_panel(
                    "Superplots",
                    ui.panel_well(
                        ui.input_selectize("testing_metric", "Test for metric:", []),
                        ui.input_selectize('palette', 'Color palette:', []),
                        # ... etc
                    ),
                    ui.card(
                        ui.output_plot("seaborn_superplot"),
                        ui.download_button("download_seaborn_superplot_svg", "Download Seaborn Superplot SVG"),
                    ),
                ),
            ),
        ),
        ui.nav_panel("Task list"),
        ui.nav_spacer(),
        ui.nav_control(ui.input_dark_mode(mode="light")),
        title="Peregrin"
    ),
)



# --- Server logic skeleton ---

def server(input: Inputs, output: Outputs, session: Session):
    """
    Main server logic for the Peregrin UI application.
    Handles dynamic UI components and user interactions for threshold filtering and file input management.
    Args:
        input (Inputs): Reactive input object for user actions and UI events.
        output (Outputs): Reactive output object for rendering UI components.
        session (Session): Session object for sending messages and managing UI state.
    Features:
        - Dynamic creation and removal of threshold filter panels (1D and 2D modes).
        - Toggle between 1D and 2D thresholding modes, updating UI accordingly.
        - Dynamic enabling/disabling of remove threshold/input buttons based on current state.
        - Dynamic creation and removal of file input/label pairs for user data upload.
        - Renders sidebar accordion UI with threshold panels and filter settings.
        - Renders sidebar label indicating current thresholding mode.
        - Sends messages to the session to update UI controls as needed.
    Notes:
        - Uses reactive.Value for state management and reactive.effect for event-driven updates.
        - UI components are rendered using the `ui` and `render` modules.
        - Assumes existence of Metrics.SpotAndTrack and Modes.Thresholding for selectize choices.
    """


    # - - - - LInput IDs for file inputs - - - -
    input_list = reactive.Value([1])                # List of input IDs for file inputs

    # - - - - Dynamic Thresholds - - - -
    # threshold_dimension = reactive.Value("1D") <- This will be default
    threshold_dimension = reactive.Value("2D")
    # dimension_button_label = reactive.Value("2D") <- This will be default
    dimension_button_label = reactive.Value("1D")
    threshold_list = reactive.Value([0])  # Start with one threshold
    property_selections = reactive.Value({})
    filter_type_selections = reactive.Value({})
    quantile_selections = reactive.Value({})
    reference_selections = reactive.Value({})
    metric_x_selections = reactive.Value({})
    metric_y_selections = reactive.Value({})
    threshold_slider_outputs = {}
    thresholding_histogram_outputs = {}
    
    threshold2d_outputs_registered: set[int] = set()       # guards plotly widget outputs
    threshold2d_placeholders_registered: set[int] = set()  # guards UI placeholders
    threshold2d_store = reactive.Value({})



    thresholding_2D_memory = reactive.Value({})
    thresholding_1D_memory = reactive.Value({})  # initialize empty first

    thresholding_memory_2d_selection = reactive.Value({})

    @reactive.Effect
    @reactive.event(threshold_list)
    def initialize_thresholding_memory():

        if threshold_dimension.get() == "1D":

            memory = thresholding_1D_memory.get()
            ids = threshold_list.get()

            for _id in ids:
                if _id not in memory:
                    memory[_id] = {
                        _property: {
                            "Literal": {"values": None},
                            "Normalized 0-1": {"values": None},
                            "Quantile": {
                                _quantile: {"values": None} for _quantile in [200, 100, 50, 25, 20, 10, 5, 4, 2]
                            },
                            "Relative to...": {
                                _reference: {"values": None} for _reference in ["Mean", "Median", "My own value"]
                            },
                            "My own value": {"my_value": None},
                        }
                        for _property in Metrics.Thresholding.Properties
                    }

            # Remove deleted threshold entries from memory
            memory = {k: v for k, v in memory.items() if k in ids}
            thresholding_1D_memory.set(memory)

        else:
            pass



    # - - - - Data frame placeholders - - - -
    RAWDATA = reactive.Value(pd.DataFrame())         # Placeholder for raw data
    UNFILTERED_SPOTSTATS = reactive.Value(pd.DataFrame())    # Placeholder for spot statistics
    UNFILTERED_TRACKSTATS = reactive.Value(pd.DataFrame())   # Placeholder for track statistics
    UNFILTERED_TIMESTATS = reactive.Value(pd.DataFrame())    # Placeholder for time statistics
    SPOTSTATS = reactive.Value(pd.DataFrame())       # Placeholder for processed spot statistics
    TRACKSTATS = reactive.Value(pd.DataFrame())      # Placeholder for processed track statistics
    TIMESTATS = reactive.Value(pd.DataFrame())       # Placeholder for processed time statistics

    # Live, per-threshold inputs (what each threshold *receives* before it runs)
    # Shape: { tid: {"spot": df, "track": df} }
    THRESH_INPUTS = reactive.Value({})

    # Live preview of the *final* chained output (updates as you edit sliders)
    THRESH_PREVIEW_OUT = reactive.Value(pd.DataFrame())
    THRESHOLDED_DF = reactive.Value(pd.DataFrame())


    # - - - - File input management - - - -

    @reactive.Effect
    @reactive.event(input.add_input)
    def add_input():
        ids = input_list.get()
        new_id = max(ids) + 1 if ids else 1
        input_list.set(ids + [new_id])
        session.send_input_message("remove_input", {"disabled": len(ids) < 1})

    @reactive.Effect
    @reactive.event(input.remove_input)
    def remove_input():
        ids = input_list.get()
        if len(ids) > 1:
            input_list.set(ids[:-1])
        if len(input_list.get()) <= 1:

            session.send_input_message("remove_input", {"disabled": True})

    @output()
    @render.ui
    def input_file_pairs():
        ids = input_list.get()
        ui_blocks = []
        for idx in ids:
            ui_blocks.append([
                ui.input_text(f"condition_label{idx}", f"Condition {idx}" if len(ids) > 1 else "Condition", placeholder=f"Label me!"),
                ui.input_file(f"input_file{idx}", "Upload files:", placeholder="Drag and drop here!", multiple=True),
                ui.markdown(""" <hr style="border: none; border-top: 1px dotted" /> """),
            ])
        return ui_blocks
    
    # - - - - - - - - - - - - - - - - - - - -




    # - - - - Defined column specification - - - -

    @reactive.Effect
    def column_selection():
        ids = input_list.get()

        for idx in ids:
            files = input[f"input_file{idx}"]()
            if files and isinstance(files, list) and len(files) > 0:
                try:
                    columns = DataLoader.GetColumns(files[0]["datapath"])

                    for sel in Metrics.LookFor.keys():
                        choice = DataLoader.FindMatchingColumn(columns, Metrics.LookFor[sel])
                        if choice is not None:
                            ui.update_selectize(sel, choices=columns, selected=choice)
                        else:
                            ui.update_selectize(sel, choices=columns, selected=columns[0] if columns else None)
                    break  # Use the first available slot
                except Exception as e:
                    continue

    # - - - - - - - - - - - - - - - - - - - -




    # - - - - Running the analysis - - - -

    @reactive.Effect
    def enable_run_button():
        files_uploaded = [input[f"input_file{idx}"]() for idx in input_list.get()]
        def is_busy(val):
            return isinstance(val, list) and len(val) > 0
        all_busy = all(is_busy(f) for f in files_uploaded)
        session.send_input_message("run", {"disabled": not all_busy})

    @reactive.Effect
    @reactive.event(input.run)
    def parsed_files():
        ids = input_list.get()
        all_data = []

        for idx in ids:
            files = input[f"input_file{idx}"]()
            label = input[f"condition_label{idx}"]()

            if not files:
                continue

            for rep_idx, fileinfo in enumerate(files, start=1):
                try:
                    df = DataLoader.GetDataFrame(fileinfo["datapath"])
                    extracted = DataLoader.Extract(
                        df,
                        id_col=input.select_id(),
                        t_col=input.select_time(),
                        x_col=input.select_x(),
                        y_col=input.select_y(),
                        mirror_y=True,
                    )
                except Exception as e:
                    # Optionally log the error
                    continue

                # Assign condition label and replicate number
                extracted["Condition"] = label if label else str(idx)
                extracted["Replicate"] = rep_idx

                all_data.append(extracted)
            
        if all_data:
            
            # Concatenate all the data into one data frame
            df = pd.concat(all_data, axis=0, ignore_index=True)

            # Create a different index (1 - infinity) for each unique track
            df['INDEX'] = df.groupby(['Condition', 'Replicate', 'Track ID']).ngroup() + 1

            # Set the created index while dropping it as a column by modifying the DataFrame in place
            df.set_index('INDEX', drop=True, inplace=True)
        
            RAWDATA.set(df)
            UNFILTERED_SPOTSTATS.set(Calc.Spots(df))
            UNFILTERED_TRACKSTATS.set(Calc.Tracks(df))
            UNFILTERED_TIMESTATS.set(Calc.Time(df))
            THRESHOLDED_DF.set(UNFILTERED_SPOTSTATS.get())
            
            ui.update_sidebar(id="sidebar", show=True)

        else:
            pass

    @render.text
    @reactive.event(input.run)
    async def initialize():
        with ui.Progress(min=0, max=30) as p:
            p.set(message="Initialization in progress")

            for i in range(1, 12):
                p.set(i, message="Initializing Peregrin...")
                await asyncio.sleep(0.1)
        pass

    # - - - - - - - - - - - - - - - - - - - -




    # - - - - Adding and removing thresholds - - - -

    @reactive.Effect
    @reactive.event(input.add_threshold)
    def add_threshold():
        ids = threshold_list.get()
        threshold_list.set(ids + [max(ids)+1 if ids else 0])
        session.send_input_message("remove_threshold", {"disabled": False})

    @reactive.Effect
    @reactive.event(input.remove_threshold)
    def remove_threshold():
        ids = threshold_list.get()
        if len(ids) > 1:
            threshold_list.set(ids[:-1])
        if len(threshold_list.get()) <= 1:
            session.send_input_message("remove_threshold", {"disabled": True})

    @reactive.Effect
    @reactive.event(input.threshold_dimensional_toggle)
    def reset_threshold_list():
        threshold_list.set([0])


    # - - - - Sidebar accordion layout for thresholds - - - -

    @output()
    @render.ui
    def sidebar_accordion():
        ids = threshold_list.get()
        panels = []
        if threshold_dimension.get() == "1D":
            for i, threshold_id in enumerate(ids, 1):
                panels.append(
                    ui.accordion_panel(
                        f"Threshold {i}" if len(ids) >= 2 else "Threshold",
                        ui.panel_well(
                            ui.input_selectize(f"threshold_property_{threshold_id}", "Property", choices=Metrics.Thresholding.Properties),
                            ui.input_selectize(f"threshold_filter_{threshold_id}", "Filter type", choices=Modes.Thresholding),
                            ui.panel_conditional(
                                f"input.threshold_filter_{threshold_id} == 'Quantile'",
                                ui.input_selectize(f"threshold_quantile_{threshold_id}", "Quantile", choices=[200, 100, 50, 25, 20, 10, 5, 4, 2], selected=100),
                            ),
                            ui.panel_conditional(
                                f"input.threshold_filter_{threshold_id} == 'Relative to...'",
                                ui.input_selectize(f"reference_value_{threshold_id}", "Reference value", choices=["Mean", "Median", "My own value"]),
                                ui.panel_conditional(
                                    f"input.reference_value_{threshold_id} == 'My own value'",
                                    ui.input_numeric(f"my_own_value_{threshold_id}", "My own value", value=None, step=1)
                                ),
                            ),
                            ui.output_ui(f"manual_threshold_value_setting_{threshold_id}"),
                            ui.output_ui(f"threshold_slider_placeholder_{threshold_id}"),
                            ui.output_plot(f"thresholding_histogram_placeholder_{threshold_id}"),
                        ),
                    ),
                )
            panels.append(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_numeric("bins", "Number of bins", value=25, min=1, step=1),
                    ui.markdown(""" <p> """),
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                ),
            )
        elif threshold_dimension.get() == "2D":
            for i, threshold_id in enumerate(ids, 1):
                panels.append(
                    ui.accordion_panel(
                        f"Threshold {i}" if len(ids) >= 2 else "Threshold",
                        ui.panel_well(
                            ui.markdown(""" <h6>  Properties X;Y  </h6>"""),
                            ui.input_selectize(f"thresholding_metric_X_{threshold_id}", None, Metrics.Thresholding.Properties, selected="Track points"),
                            ui.input_selectize(f"thresholding_metric_Y_{threshold_id}", None, Metrics.Thresholding.Properties, selected="Confinement ratio"),
                            ui.div(
                                {
                                    "style": "position:relative; width:100%; height:175px;"
                                },
                                ui.output_plot(
                                    id=f"threshold2d_widget_{threshold_id}",
                                    # keep Shiny's built-in brush if you still want it
                                    brush=True,
                                    height="175px"
                                ),
                                Brush.OverlayBrush(f"threshold2d_widget_{threshold_id}")  # emits input[f"{out_id}_brush2"]
                            ),
                            # ui.output_plot(
                            #     f"threshold2d_widget_{threshold_id}",
                            #     # hover=ui.hover_opts(delay=60, delay_type="throttle"),
                            #     brush=ui.brush_opts(
                            #         stroke="#06519c",
                            #         opacity=0.175,
                            #         direction="xy",
                            #         delay=60,
                            #         delay_type="debounce"
                            #     ),
                            #     height="175px"
                            # ),
                            ui.input_action_button(id=f"pass_selected_{threshold_id}", label="Pass Selected", class_="space-x-2")
                        ),
                    ),
                )
            panels.append(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_numeric("threshold2d_dot_size", "Dot size", value=0.75, min=0, step=0.05),
                    ui.input_numeric("threshold2d_dot_opacity", "Dot opacity", value=0.9, min=0, max=1, step=0.05),
                    ui.markdown(""" <p> """),
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                ),
            ),


        # return ui.accordion(*panels, id="thresholds_accordion", open=["Threshold", f"Threshold {len(ids)}", "Filter settings"])
        return ui.accordion(*panels, id="thresholds_accordion", open=True)
    
    
    # - - - - Threshold dimension toggle - - - -

    @reactive.Effect
    @reactive.event(input.threshold_dimensional_toggle)
    def threshold_dimensional_toggle():
        if threshold_dimension.get() == "1D":
            threshold_dimension.set("2D")
            dimension_button_label.set("1D")
        else:
            threshold_dimension.set("1D")
            dimension_button_label.set("2D")

    @output()
    @render.text
    def sidebar_label():
        return ui.markdown(
            f""" <h5> <b>  {threshold_dimension.get()} Data filtering  </b> </h5> """
        )
    
   
    # - - - - Storing thresholding values - - - -

    def _get_threshold_memory(dict_memory, threshold_id, property_name, filter_type, default_values, quantile=None, reference=None, ref_val=None):
        """
        Returns a tuple: (values_pair, ref_val) where ref_val is only used for "My own value".
        If not found, returns (default_values, None).
        """
        def _are_valid(values):
            return (
                isinstance(values, (tuple, list))
                and len(values) == 2
                and all(isinstance(x, (int, float)) for x in values)
            )
        def _is_valid(x):
            return isinstance(x, (int, float))

        try:
            if quantile is None and reference is None:
                values = dict_memory[threshold_id][property_name][filter_type]["values"]
                if _are_valid(values):
                    return values, None

            elif quantile is not None:
                values = dict_memory[threshold_id][property_name][filter_type][quantile]["values"]
                if _are_valid(values):
                    return values, None

            elif reference is not None:
                if reference == "My own value":
                    values = dict_memory[threshold_id][property_name][filter_type][reference]["values"]
                    stored_ref = dict_memory[threshold_id][property_name]["My own value"]["my_value"]

                    # Prefer stored; otherwise fall back to provided ref_val (if numeric)
                    final_ref = stored_ref if isinstance(stored_ref, (int, float)) else (ref_val if isinstance(ref_val, (int, float)) else None)

                    if _are_valid(values) and isinstance(final_ref, (int, float)):
                        return values, final_ref
                    # If values missing but we at least have a numeric ref, still return (default_values, ref)
                    if isinstance(final_ref, (int, float)):
                        return default_values, final_ref
                else:
                    values = dict_memory[threshold_id][property_name][filter_type][reference]["values"]
                    if _are_valid(values):
                        return values, None

        except Exception:
            pass

        # Fallback – preserve expected 2-tuple result
        return default_values, None

    def _set_threshold_memory(dict_memory, threshold_id, property_name, filter_type, values, quantile=None, reference=None, ref_val=None):
        if threshold_id not in dict_memory:
            dict_memory[threshold_id] = {}
        if property_name not in dict_memory[threshold_id]:
            dict_memory[threshold_id][property_name] = {}
        if filter_type not in dict_memory[threshold_id][property_name]:
            dict_memory[threshold_id][property_name][filter_type] = {}

        if quantile is None and reference is None:
            dict_memory[threshold_id][property_name][filter_type]["values"] = tuple(values)
        elif quantile is not None:
            dict_memory[threshold_id][property_name][filter_type].setdefault(quantile, {})
            dict_memory[threshold_id][property_name][filter_type][quantile]["values"] = tuple(values)
        elif reference is not None:
            dict_memory[threshold_id][property_name][filter_type].setdefault(reference, {})
            dict_memory[threshold_id][property_name][filter_type][reference]["values"] = tuple(values)
            if reference == "My own value":
                dict_memory[threshold_id][property_name].setdefault(filter_type, {})
                dict_memory[threshold_id][property_name].setdefault("My own value", {})
                dict_memory[threshold_id][property_name]["My own value"]["my_value"] = ref_val

        return dict_memory



    EPS = 1e-12

    def _is_whole_number(x) -> bool:
        try:
            fx = float(x)
        except Exception:
            return False
        return abs(fx - round(fx)) < EPS

    def _int_if_whole(x):
        # Return an int if x is effectively whole, otherwise return float
        if x is None:
            return None
        try:
            fx = float(x)
        except Exception:
            return x
        if _is_whole_number(fx):
            return int(round(fx))
        return fx

    def _format_numeric_pair(values):
        """
        Normalize `values` into a (low, high) numeric pair.

        Accepts:
        - scalar numbers (including numpy.float64) -> returns (v, v)
        - 1-length iterables -> returns (v, v)
        - 2+-length iterables -> returns (first, second) but ensures low<=high where possible
        - None or empty -> (None, None)
        """
        import numpy as _np

        if values is None:
            return None, None

        # numpy scalar or python scalar
        if _np.isscalar(values):
            v = values.item() if hasattr(values, "item") else float(values)
            v = _int_if_whole(v)
            return v, v

        # Try to coerce to list/sequence
        try:
            seq = list(values)
        except Exception:
            # Fallback: treat as scalar
            try:
                v = float(values)
                v = _int_if_whole(v)
                return v, v
            except Exception:
                return None, None

        if len(seq) == 0:
            return None, None
        if len(seq) == 1:
            v = seq[0]
            try:
                fv = float(v)
                fv = _int_if_whole(fv)
                return fv, fv
            except Exception:
                return v, v

        # len >= 2 -> take first two and try to ensure lo <= hi
        a, b = seq[0], seq[1]
        try:
            fa = float(a)
            fb = float(b)
            if fa <= fb:
                return _int_if_whole(fa), _int_if_whole(fb)
            else:
                return _int_if_whole(fb), _int_if_whole(fa)
        except Exception:
            return _int_if_whole(a), _int_if_whole(b)

    def _compute_reference_and_span(values_series: pd.Series, reference: str, my_value: float | None):
        """
        Returns (reference_value, max_delta) for the 'Relative to...' mode.
        max_delta is the farthest absolute distance from reference to any data point.
        """
        vals = values_series.dropna()
        if vals.empty:
            return 0.0, 0.0

        if reference == "Mean":
            ref = float(vals.mean())
        elif reference == "Median":
            ref = float(vals.median())
        elif reference == "My own value":
            ref = float(my_value) if isinstance(my_value, (int, float)) else float(vals.mean())
        else:
            ref = float(vals.mean())

        max_delta = float(np.max(np.abs(vals - ref)))
        return ref, max_delta


    # - - - - Threshold slider generator - - - -

    def _get_steps(lowest, highest):
        """
        Returns the step size for the slider based on the range.
        """
        if (highest - lowest) < 0.01:
            steps = 0.00001
        elif 0.01 < (highest - lowest) < 0.1:
            steps = 0.0001
        elif 0.1 < (highest - lowest) < 1:
            steps = 0.001
        elif 1 < (highest - lowest) < 10:
            steps = 0.01
        elif 10 < (highest - lowest) < 100:
            steps = 0.1
        elif 100 < (highest - lowest) < 1000:
            steps = 1
        elif 1000 < (highest - lowest) < 10000:
            steps = 10
        else:
            steps = 1
        return steps


    def _get_filter_value_params(
        threshold_id: int,
        spot_data: pd.DataFrame, 
        track_data: pd.DataFrame, 
        property_name: str, 
        filter_type: str, 
        memory: dict, 
        quantile: int = None,
        reference: str = None,
        reference_value: float = None
    ):
        if filter_type == "Literal":
            if property_name in Metrics.Thresholding.SpotProperties:
                lowest = spot_data[property_name].min()
                highest = spot_data[property_name].max()
            elif property_name in Metrics.Thresholding.TrackProperties:
                lowest = track_data[property_name].min()
                highest = track_data[property_name].max()
            else:
                lowest, highest = 0, 100
            default = (lowest, highest)
            
            # Use memory if valid, otherwise use defaults
            steps = _get_steps(lowest, highest)
            values, ref_val = _get_threshold_memory(memory, threshold_id, property_name, filter_type, default)
            minimal = Process.Round(lowest, steps, "floor")
            maximal = Process.Round(highest, steps, "ceil")

        elif filter_type == "Normalized 0-1":
            lowest, highest = 0, 1
            default = (lowest, highest)

            # Use memory if valid, otherwise use defaults
            values, ref_val = _get_threshold_memory(memory, threshold_id, property_name, filter_type, default)
            steps = 0.01
            minimal = 0
            maximal = 1

        elif filter_type == "Quantile":
            lowest, highest = 0, 100
            default = (lowest, highest)

            # Use memory if valid, otherwise use defaults
            values, ref_val = _get_threshold_memory(memory, threshold_id, property_name, filter_type, default, quantile=quantile)
            steps = 100/float(quantile)
            minimal = 0
            maximal = 100

        elif filter_type == "Relative to...":
            # Pick the right table
            if property_name in Metrics.Thresholding.SpotProperties:
                series = spot_data[property_name]
            elif property_name in Metrics.Thresholding.TrackProperties:
                series = track_data[property_name]
            else:
                series = pd.Series(dtype=float)

            # Compute reference and span
            ref, max_delta = _compute_reference_and_span(series, reference, reference_value)

            lowest = 0.0
            highest = float(max_delta) if np.isfinite(max_delta) else 0.0
            default = (lowest, highest)

            steps = _get_steps(lowest, highest)
            values, ref_val = _get_threshold_memory(
                memory, threshold_id, property_name, filter_type, default,
                reference=reference, ref_val=reference_value
            )
            minimal = Process.Round(lowest, steps, "floor")
            maximal = Process.Round(highest, steps, "ceil")
            
        return steps, values, ref_val, minimal, maximal


    # Make threshold sliders dynamically based on the threshold ID
    def render_threshold_slider(threshold_id):
        @output(id=f"threshold_slider_placeholder_{threshold_id}")
        @render.ui
        def threshold_slider():

            state = THRESH_INPUTS.get().get(threshold_id, {})
            spot_data  = state.get("spot",  UNFILTERED_SPOTSTATS.get())
            track_data = state.get("track", UNFILTERED_TRACKSTATS.get())
            if spot_data is None or spot_data.empty or track_data is None or track_data.empty:
                return

            property_name = input[f"threshold_property_{threshold_id}"]()
            filter_type = input[f"threshold_filter_{threshold_id}"]()
            if not property_name or not filter_type:
                return

            steps, values, ref_val, minimal, maximal = _get_filter_value_params(
                threshold_id=threshold_id,
                spot_data=spot_data,
                track_data=track_data,
                property_name=property_name,
                filter_type=filter_type,
                memory=thresholding_1D_memory.get(),
                quantile=input[f"threshold_quantile_{threshold_id}"](),
                reference=input[f"reference_value_{threshold_id}"](),
                reference_value=input[f"my_own_value_{threshold_id}"]()
            )

            return ui.input_slider(
                f"threshold_slider_{threshold_id}",
                None,
                min=minimal,
                max=maximal,
                value=values,
                step=steps
            )


    def render_manual_threshold_values_setting(threshold_id):
        @output(id=f"manual_threshold_value_setting_{threshold_id}")
        @render.ui
        def manual_threshold_value_setting():

            state = THRESH_INPUTS.get().get(threshold_id, {})
            spot_data  = state.get("spot",  UNFILTERED_SPOTSTATS.get())
            track_data = state.get("track", UNFILTERED_TRACKSTATS.get())
            if spot_data is None or spot_data.empty or track_data is None or track_data.empty:
                return

            property_name = input[f"threshold_property_{threshold_id}"]()
            filter_type = input[f"threshold_filter_{threshold_id}"]()
            if not property_name or not filter_type:
                return
            
            steps, values, ref_val, minimal, maximal = _get_filter_value_params(
                threshold_id=threshold_id,
                spot_data=spot_data,
                track_data=track_data,
                property_name=property_name,
                filter_type=filter_type,
                memory=thresholding_1D_memory.get(),
                quantile=input[f"threshold_quantile_{threshold_id}"](),
                reference=input[f"reference_value_{threshold_id}"](),
                reference_value=input[f"my_own_value_{threshold_id}"]()
            )
            
            v_lo, v_hi = _format_numeric_pair(values)
            min_fmt, max_fmt = _int_if_whole(minimal), _int_if_whole(maximal)

            return ui.row(
                ui.column(6, ui.input_numeric(
                    f"bottom_threshold_value_{threshold_id}",
                    label="min",
                    value=v_lo,
                    min=min_fmt,
                    max=max_fmt,
                    step=steps
                )),
                ui.column(6, ui.input_numeric(
                    f"top_threshold_value_{threshold_id}",
                    label="max",
                    value=v_hi,
                    min=min_fmt,
                    max=max_fmt,
                    step=steps
                )),
            )

    # - - - - Threshold histogram generator - - - -

    def render_threshold_histogram(threshold_id):
        @output(id=f"thresholding_histogram_placeholder_{threshold_id}")
        @render.plot
        def threshold_histogram():
            state = THRESH_INPUTS.get().get(threshold_id, {})
            prop = input[f"threshold_property_{threshold_id}"]()
            if prop in Metrics.Thresholding.SpotProperties:
                data = state.get("spot", UNFILTERED_SPOTSTATS.get())
            elif prop in Metrics.Thresholding.TrackProperties:
                data = state.get("track", UNFILTERED_TRACKSTATS.get())
            else:
                return
            
            property = input[f"threshold_property_{threshold_id}"]()
            filter_type = input[f"threshold_filter_{threshold_id}"]()
            try:
                slider_low_pct, slider_high_pct = input[f"threshold_slider_{threshold_id}"]()
            except Exception:
                return

            if filter_type == "Literal":

                bins = input.bins() if input.bins() is not None else 25
                values = data[property].dropna()

                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(values, bins=bins, density=False)

                # Color threshold
                for i in range(len(patches)):
                    if bins[i] < slider_low_pct or bins[i+1] > slider_high_pct:
                        patches[i].set_facecolor('grey')
                    else:
                        patches[i].set_facecolor('#337ab7')

                # Add KDE curve (scaled to match histogram)
                kde = gaussian_kde(values)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                # Scale KDE to histogram
                y_kde_scaled = y_kde * (n.max() / y_kde.max())
                ax.plot(x_kde, y_kde_scaled, color='black', linewidth=1.5)

                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines[['top', 'left', 'right']].set_visible(False)

                return fig
            
            if filter_type == "Normalized 0-1":

                values = data[property].dropna()
                try:
                    normalized = (values - values.min()) / (values.max() - values.min())
                except ZeroDivisionError:
                    normalized = 0
                bins = input.bins() if input.bins() is not None else 25

                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(normalized, bins=bins, density=False)

                # Color threshold
                for i in range(len(patches)):
                    if bins[i] < slider_low_pct or bins[i+1] > slider_high_pct:
                        patches[i].set_facecolor('grey')
                    else:
                        patches[i].set_facecolor('#337ab7')

                # Add KDE curve (scaled to match histogram)
                kde = gaussian_kde(normalized)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                # Scale KDE to histogram
                y_kde_scaled = y_kde * (n.max() / y_kde.max())
                ax.plot(x_kde, y_kde_scaled, color='black', linewidth=1.5)

                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines[['top', 'left', 'right']].set_visible(False)

                return fig

            if filter_type == "Quantile":
                bins = input.bins() if input.bins() is not None else 25

                values = data[property].dropna()
                
                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(values, bins=bins, density=False)

                # Get slider quantile values, 0-100 scale
                slider_low, slider_high = slider_low_pct / 100, slider_high_pct / 100

                if not 0 <= slider_low <= 1 or not 0 <= slider_high <= 1:
                    slider_low, slider_high = 0, 1

                # Convert slider percentiles to actual values
                lower_bound = np.quantile(values, slider_low)
                upper_bound = np.quantile(values, slider_high)

                # Color histogram based on slider quantile bounds
                for i in range(len(patches)):
                    bin_start, bin_end = bins[i], bins[i + 1]
                    if bin_end < lower_bound or bin_start > upper_bound:
                        patches[i].set_facecolor('grey')
                    else:
                        patches[i].set_facecolor('#337ab7')

                # KDE curve
                kde = gaussian_kde(values)
                x_kde = np.linspace(values.min(), values.max(), 500)
                y_kde = kde(x_kde)
                y_kde_scaled = y_kde * (n.max() / y_kde.max()) if y_kde.max() != 0 else y_kde
                ax.plot(x_kde, y_kde_scaled, color='black', linewidth=1.5)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[['top', 'left', 'right']].set_visible(False)
                return fig

            if filter_type == "Relative to...":
                reference = input[f"reference_value_{threshold_id}"]()
                if reference == "Mean":
                    reference_value = float(data[property].dropna().mean())
                elif reference == "Median":
                    reference_value = float(data[property].dropna().median())
                elif reference == "My own value":
                    try:
                        mv = input[f"my_own_value_{threshold_id}"]()
                        reference_value = float(mv) if isinstance(mv, (int, float)) else float(data[property].dropna().mean())
                    except Exception:
                        reference_value = float(data[property].dropna().mean())
                else:
                    return

                # Build histogram in "shifted" space (centered at 0 = reference)
                shifted = data[property].dropna() - reference_value
                bins = input.bins() if input.bins() is not None else 25

                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(shifted, bins=bins, density=False)

                # Slider gives distances [low, high] away from the reference
                sel_low, sel_high = input[f"threshold_slider_{threshold_id}"]()
                # Normalize order just in case
                if sel_low > sel_high:
                    sel_low, sel_high = sel_high, sel_low

                # Utility: does [bin_start, bin_end] intersect either [-sel_high, -sel_low] or [sel_low, sel_high]?
                def _intersects_symmetric(b0, b1, a, b):
                    # interval A: [-b, -a], interval B: [a, b]
                    left_hit  = (b1 >= -b) and (b0 <= -a)
                    right_hit = (b1 >=  a) and (b0 <=  b)
                    return left_hit or right_hit

                # Color threshold bands: keep bars whose centers fall within the selected annulus
                for i in range(len(patches)):
                    bin_start, bin_end = bins[i], bins[i+1]
                    if _intersects_symmetric(bin_start, bin_end, sel_low, sel_high):
                        patches[i].set_facecolor('#337ab7')
                    else:
                        patches[i].set_facecolor('grey')

                # KDE on shifted values (optional but matches your style)
                kde = gaussian_kde(shifted)
                x_kde = np.linspace(bins[0], bins[-1], 500)
                y_kde = kde(x_kde)
                y_kde_scaled = y_kde * (n.max() / y_kde.max()) if y_kde.max() != 0 else y_kde
                ax.plot(x_kde, y_kde_scaled, color='black', linewidth=1.5)

                ax.axvline(0, linestyle='--', linewidth=1, color='black')


                ax.set_xticks([]); ax.set_yticks([])
                ax.spines[['top', 'left', 'right']].set_visible(False)
                return fig


    # - - - - Sync helpers & per-threshold synchronization - - - -

    @reactive.Effect
    def cache_2d_metric_selections() -> None:
        if threshold_dimension.get() == "2D":
            _x = {}
            _y = {}
            for threshold_id in threshold_list.get():
                xv = input[f"thresholding_metric_X_{threshold_id}"]()
                yv = input[f"thresholding_metric_Y_{threshold_id}"]()
                if isinstance(xv, str) and xv in Metrics.Thresholding.Properties:
                    _x[threshold_id] = xv
                if isinstance(yv, str) and yv in Metrics.Thresholding.Properties:
                    _y[threshold_id] = yv
            # keep only still-existing thresholds
            ids = set(threshold_list.get())
            metric_x_selections.set({tid: v for tid, v in _x.items() if tid in ids})
            metric_y_selections.set({tid: v for tid, v in _y.items() if tid in ids})

        else:
            pass

    @reactive.Effect
    def metric_x_selectize():
        saved = metric_x_selections.get()
        choices = Metrics.Thresholding.Properties
        default = choices[0] if choices else None
        for threshold_id in threshold_list.get():
            selected = saved.get(threshold_id, input[f"thresholding_metric_X_{threshold_id}"]() or default)
            if selected not in choices:
                selected = default
            ui.update_selectize(
                id=f"thresholding_metric_X_{threshold_id}",
                choices=choices,
                selected=selected
            )

    @reactive.Effect
    def metric_y_selectize():
        saved = metric_y_selections.get()
        choices = Metrics.Thresholding.Properties
        default = choices[0] if choices else None
        for threshold_id in threshold_list.get():
            selected = saved.get(threshold_id, input[f"thresholding_metric_Y_{threshold_id}"]() or default)
            if selected not in choices:
                selected = default
            ui.update_selectize(
                id=f"thresholding_metric_Y_{threshold_id}",
                choices=choices,
                selected=selected
            )


    def _last_or_default_metric(saved: dict):
        if saved:
            # take value from the highest existing threshold id
            return saved[sorted(saved.keys())[-1]]
        return Metrics.Thresholding.Properties[0] if Metrics.Thresholding.Properties else None

    @reactive.Effect
    @reactive.event(input.add_threshold)
    def set_defaults_for_new_2d_threshold():
        # after list updates, set UI defaults using previous choices
        ids = threshold_list.get()
        if not ids:
            return
        tid = ids[-1]
        default_x = _last_or_default_metric(metric_x_selections.get())
        default_y = _last_or_default_metric(metric_y_selections.get())
        ui.update_selectize(f"thresholding_metric_X_{tid}", choices=Metrics.Thresholding.Properties, selected=default_x)
        ui.update_selectize(f"thresholding_metric_Y_{tid}", choices=Metrics.Thresholding.Properties, selected=default_y)

    




    def _current_context(threshold_id):
        """Return (property_name, filter_type, quantile, reference_value) from UI inputs."""
        property_name = input[f"threshold_property_{threshold_id}"]()
        filter_type   = input[f"threshold_filter_{threshold_id}"]()
        quantile = None
        reference = None
        ref_val = None

        if filter_type == "Quantile":
            quantile = input[f"threshold_quantile_{threshold_id}"]()
        elif filter_type == "Relative to...":
            reference = input[f"reference_value_{threshold_id}"]()
            if reference == "My own value":
                try:
                    ref_val = input[f"my_own_value_{threshold_id}"]()
                except Exception:
                    ref_val = None
        return property_name, filter_type, quantile, reference, ref_val


    def _nearly_equal_pair(a, b, eps=EPS):
        try:
            return abs(float(a[0]) - float(b[0])) <= eps and abs(float(a[1]) - float(b[1])) <= eps
        except Exception:
            return False


    def _read_stored_pair(mem, threshold_id, prop, ftype, q, ref):
        try:
            if q is None and ref is None:
                val = mem[threshold_id][prop][ftype]["values"]
                ref_val = None
            elif q is not None and ref is None:
                val = mem[threshold_id][prop][ftype][q]["values"]
                ref_val = None
            elif ref is not None and q is None:
                # Handle "My own value" explicitly
                val = mem[threshold_id][prop][ftype][ref]["values"]
                ref_val = mem[threshold_id][prop]["My own value"]["my_value"] if ref == "My own value" else None
            else:
                return None, None

            if isinstance(val, (tuple, list)) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
                return tuple(val), (ref_val if isinstance(ref_val, (int, float)) else None)
        except Exception:
            pass
        return None, None


    def register_threshold_sync(threshold_id):

        # When the property changes, reset the min/max values
        @reactive.Effect
        @reactive.event(input[f"threshold_property_{threshold_id}"])
        def _reset_minmax_on_change():
            steps, values, ref_val, minimal, maximal = _get_filter_value_params(
                threshold_id=threshold_id,
                spot_data=UNFILTERED_SPOTSTATS.get(),
                track_data=UNFILTERED_TRACKSTATS.get(),
                property_name=input[f"threshold_property_{threshold_id}"](),
                filter_type=input[f"threshold_filter_{threshold_id}"](),
                memory=thresholding_1D_memory.get(),
                quantile=input[f"threshold_quantile_{threshold_id}"](),
                reference=input[f"reference_value_{threshold_id}"](),
                reference_value=input[f"my_own_value_{threshold_id}"]()
            )

            ui.update_slider(
                id=f"threshold_slider_{threshold_id}",
                min=minimal,
                max=maximal,
                step=steps,
                value=values
            )
            ui.update_numeric(f"bottom_threshold_value_{threshold_id}", value=float(values[0]))
            ui.update_numeric(f"top_threshold_value_{threshold_id}", value=float(values[1]))
            # ui.update_numeric(f"my_own_value_{threshold_id}", value=float(ref_val))

        # A) slider -> memory
        @reactive.Effect
        @reactive.event(input[f"threshold_slider_{threshold_id}"])
        def _slider_to_memory():
            vals = input[f"threshold_slider_{threshold_id}"]()
            if not (isinstance(vals, (tuple, list)) and len(vals) == 2 and all(v is not None for v in vals)):
                return

            prop, ftype, q, ref, ref_val = _current_context(threshold_id)
            if not (prop and ftype):
                return

            mem = thresholding_1D_memory.get()
            cur_vals, cur_ref_val = _read_stored_pair(mem, threshold_id, prop, ftype, q, ref)

            vals = (float(vals[0]), float(vals[1]))
            if ref == "My own value" and ref_val is not None:
                try:
                    ref_val = float(ref_val)
                except Exception:
                    ref_val = None

            # Compute new memory once
            need_vals = (cur_vals is None or not _nearly_equal_pair(vals, cur_vals))
            need_ref  = (ref == "My own value" and ref_val is not None and (cur_ref_val is None or abs(float(ref_val) - float(cur_ref_val)) > 1e-12))

            if need_vals or need_ref:
                new_mem = _set_threshold_memory(mem.copy(), threshold_id, prop, ftype, vals, quantile=q, reference=ref, ref_val=ref_val)
                thresholding_1D_memory.set(new_mem)


        # B) manual numerics -> memory
        @reactive.Effect
        @reactive.event(
            input[f"bottom_threshold_value_{threshold_id}"],
            input[f"top_threshold_value_{threshold_id}"],
        )
        def _manual_to_memory():
            lo = input[f"bottom_threshold_value_{threshold_id}"]()
            hi = input[f"top_threshold_value_{threshold_id}"]()
            if not all(isinstance(v, (int, float)) for v in (lo, hi)):
                return

            # Ensure order
            lo, hi = (float(lo), float(hi))
            if lo > hi:
                lo, hi = hi, lo

            prop, ftype, q, ref, ref_val = _current_context(threshold_id)
            if not (prop and ftype):
                return

            mem = thresholding_1D_memory.get()
            cur_vals, cur_ref_val = _read_stored_pair(mem, threshold_id, prop, ftype, q, ref)

            new_pair = (lo, hi)

            if cur_vals is None or not _nearly_equal_pair(new_pair, cur_vals):
                thresholding_1D_memory.set(
                    _set_threshold_memory(mem.copy(), threshold_id, prop, ftype, new_pair, quantile=q, reference=ref, ref_val=ref_val)
                )

        # C) memory -> UI (push only when different)
        @reactive.Effect
        def _memory_to_ui():
            prop, ftype, q, ref, ref_val = _current_context(threshold_id)
            if not (prop and ftype):
                return

            mem = thresholding_1D_memory.get()

            # Avoid establishing reactive deps on inputs here
            with reactive.isolate():
                current_slider = input[f"threshold_slider_{threshold_id}"]()
                try:
                    cur_lo = input[f"bottom_threshold_value_{threshold_id}"]()
                except Exception:
                    cur_lo = None
                try:
                    cur_hi = input[f"top_threshold_value_{threshold_id}"]()
                except Exception:
                    cur_hi = None
                try:
                    current_val = input[f"my_own_value_{threshold_id}"]()
                except Exception:
                    current_val = None

            default_vals = current_slider if (isinstance(current_slider, (tuple, list)) and len(current_slider) == 2) else (None, None)
            vals, stored_ref = _get_threshold_memory(
                mem, threshold_id, prop, ftype, default_vals,
                quantile=q, reference=ref, ref_val=current_val
            )
            # Prefer stored; if missing, keep whatever user currently has (don't overwrite)
            effective_ref = stored_ref if isinstance(stored_ref, (int, float)) else current_val

            if not (isinstance(vals, (tuple, list)) and len(vals) == 2 and all(v is not None for v in vals)):
                return

            # Push to slider if needed
            if not _nearly_equal_pair(vals, current_slider if isinstance(current_slider, (tuple, list)) and len(current_slider) == 2 else (None, None)):
                ui.update_slider(f"threshold_slider_{threshold_id}", value=tuple(vals))

            # Push to numerics if needed
            if not (isinstance(cur_lo, (int, float)) and abs(float(cur_lo) - float(vals[0])) <= EPS):
                ui.update_numeric(f"bottom_threshold_value_{threshold_id}", value=float(vals[0]))
            if not (isinstance(cur_hi, (int, float)) and abs(float(cur_hi) - float(vals[1])) <= EPS):
                ui.update_numeric(f"top_threshold_value_{threshold_id}", value=float(vals[1]))
            if ref == "My own value":
                cur_val = current_val  # captured in isolate()
                # Only push if we actually have a stored value AND it differs from the UI
                if isinstance(effective_ref, (int, float)) and not (isinstance(cur_val, (int, float)) and abs(float(cur_val) - float(effective_ref)) <= EPS):
                    ui.update_numeric(f"my_own_value_{threshold_id}", value=float(effective_ref))


        # D) my-own-value -> memory
        @reactive.Effect
        @reactive.event(input[f"my_own_value_{threshold_id}"])
        def _myown_to_memory():
            prop, ftype, q, ref, ref_val = _current_context(threshold_id)
            if not (prop and ftype and ref == "My own value"):
                return

            # Ignore non-numeric or None
            if not isinstance(ref_val, (int, float)):
                return

            mem = thresholding_1D_memory.get()

            # Use current slider pair as the values to keep them in sync; if missing, default to (0,0)
            current_slider = input[f"threshold_slider_{threshold_id}"]()
            if isinstance(current_slider, (tuple, list)) and len(current_slider) == 2:
                pair = (float(current_slider[0]), float(current_slider[1]))
            else:
                pair = (0.0, 0.0)

            new_mem = _set_threshold_memory(
                mem.copy(), threshold_id, prop, ftype, pair,
                quantile=q, reference=ref, ref_val=float(ref_val)
            )
            thresholding_1D_memory.set(new_mem)



    # ======================= 2D THRESHOLDING ===========================

    # Memory: selected original-row indices for each 2D threshold and metric pair
    # Shape: { tid: { (propX, propY): set([...original row indices...]) } }
    # thresholding_memory_2d_selection = reactive.Value({})

    # def _get_series(prop: str, spot_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
    def _get_series(prop: str, spot_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
        """Return a compact 2-col frame [Track ID, prop] from the right source."""
        # if prop in Metrics.Thresholding.TrackProperties and not track_df.empty:
        if prop in Metrics.Thresholding.TrackProperties:
            return Threshold.Normalize_01(track_df, prop)
        # if prop in Metrics.Thresholding.SpotProperties and not spot_df.empty:
        if prop in Metrics.Thresholding.SpotProperties:
            return Threshold.Normalize_01(spot_df, prop)
        return pd.DataFrame(columns=[prop]).set_index(pd.Index([], name='INDEX'))

    # def _xy_for_2d_threshold(threshold_id: int, spot_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Build XY for *this* block, restricted by prior brushes.
    #     Prior brushes are stored as sets of Track IDs in `thresholding_memory_2d_selection`.
    #     """
    #     mem = thresholding_memory_2d_selection.get()

    #     # 1) Intersect all *previous* selections (by Track ID). If no prior selection -> no restriction.
    #     selected_tids: set | None = None
    #     for tid in threshold_list.get():
    #         if tid == threshold_id:
    #             break
    #         propX_prev = input[f"thresholding_metric_X_{tid}"]()
    #         propY_prev = input[f"thresholding_metric_Y_{tid}"]()
    #         if not (propX_prev and propY_prev):
    #             continue
    #         sel = mem.get(tid, {}).get((propX_prev, propY_prev), set())
    #         if not sel:
    #             # No selection at that step => pass-through
    #             continue
    #         selected_tids = sel if selected_tids is None else (selected_tids & sel)
    #         if selected_tids is not None and not selected_tids:
    #             break

    #     # 2) Build XY for THIS block and restrict to previous intersection (if any)
    #     propX = input[f"thresholding_metric_X_{threshold_id}"]()
    #     propY = input[f"thresholding_metric_Y_{threshold_id}"]()

    #     if not (propX and propY):
    #         return pd.DataFrame(columns=[propX, propY]).set_index(pd.Index([], name='INDEX'))
        
    #     xy_cur = Threshold.JoinByIndex(
    #         _get_series(propX, spot_df, track_df), 
    #         _get_series(propY, spot_df, track_df)
    #     )

    #     req(not xy_cur.empty)
    #     # if selected_tids:
    #     #     # keep only rows whose Track ID survived all previous brushes
    #     #     mask = pd.Series(xy_cur.index).isin(selected_tids).to_numpy()
    #     #     xy_cur = xy_cur.loc[mask]

    #     print("-----------------------------------------------------------------------------------")
    #     print(f"2D threshold {threshold_id} on props ({propX}, {propY}) with {len(xy_cur)} points")
    #     print(f"Current selection: {selected_tids if selected_tids is not None else 'none'}")
    #     print(f"Current data: {xy_cur}")

    #     return xy_cur

    def _xy_for_2d_threshold(threshold_id: int, spot_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build XY for *this* block, restricted by prior brushes.
        Prior brushes are stored as sets of Track IDs in `thresholding_memory_2d_selection`.
        """
        # mem = thresholding_memory_2d_selection.get()

        # 1) Intersect all *previous* selections (by Track ID). If no prior selection -> no restriction.
        # selected_tids: set | None = None
        # for tid in threshold_list.get():
        #     if tid == threshold_id:
        #         break
        #     propX_prev = input[f"thresholding_metric_X_{tid}"]()
        #     propY_prev = input[f"thresholding_metric_Y_{tid}"]()
        #     if not (propX_prev and propY_prev):
        #         continue
        #     sel = mem.get(tid, {}).get((propX_prev, propY_prev))
        #     if not sel:
        #         # No selection at that step => pass-through
        #         continue
        #     selected_tids = sel if selected_tids is None else (selected_tids & sel)
        #     if selected_tids is not None and not selected_tids:
        #         break

        # 2) Build XY for THIS block and restrict to previous intersection (if any)
        propX = input[f"thresholding_metric_X_{threshold_id}"]()
        propY = input[f"thresholding_metric_Y_{threshold_id}"]()

        if not (propX and propY):
            return pd.DataFrame(columns=[propX, propY]).set_index(pd.Index([], name='INDEX'))
        
        xy_cur = Threshold.JoinByIndex(
            _get_series(propX, spot_df, track_df), 
            _get_series(propY, spot_df, track_df)
        )

        # req(not xy_cur.empty)
        # if selected_tids:
        #     # keep only rows whose Track ID survived all previous brushes
        #     mask = pd.Series(xy_cur.index).isin(selected_tids).to_numpy()
        #     xy_cur = xy_cur.loc[mask]

        # print("===========================================================")
        # print(f"2D threshold {threshold_id} on props ({propX}, {propY}) with {len(xy_cur)} points")
        # print(f"Current selection: {selected_tids if selected_tids is not None else 'none'}")
        # print(f"Current data: {xy_cur}")

        return xy_cur







    # --- helpers already present ---
    # _normalize_0_1(series), _get_metric_series(prop, spot_df, track_df)

    thresholds_state = reactive.Value({int: dict})

    @reactive.Effect
    @reactive.event(input.run)
    def _pass_data():
        thresholds_state.set({0: {"spots": UNFILTERED_SPOTSTATS.get(), "tracks": UNFILTERED_TRACKSTATS.get()}})



    def render_threshold2d_widget(threshold_id):
        # A) Plot
        @output(id=f"threshold2d_widget_{threshold_id}")
        @render.plot
        def threshold2d_chart():

            # print("===========================================================")
            print(f"Rendering widget for id: {threshold_id}")

            state = thresholds_state.get()
            req(state is not None and isinstance(state, dict))
            # print(f"Acquired state: {state}") # works
            # print(f"Whats up: {list(enumerate(state))}")

            # print("---------------------------------------------------")
            # print(f"state {len(state)}")
            
            try:
                current_state = state.get(threshold_id)
            except Exception:
                current_state = None

            req(
                isinstance(current_state, dict)
                and isinstance(current_state["spots"], pd.DataFrame)
                and isinstance(current_state["tracks"], pd.DataFrame)
            )

            # print("---------------------------------------------------")
            # print(f"Rendering 2D threshold plot for ID {threshold_id}")

            # print(f"Current state: {current_state}")

            spot_df, track_df = current_state.get("spots"), current_state.get("tracks")
            req(not spot_df.empty and not track_df.empty)
            # print("---------------------------------------------------")
            # print(f"Widget spot df: {spot_df}")

            propX = input[f"thresholding_metric_X_{threshold_id}"]()
            propY = input[f"thresholding_metric_Y_{threshold_id}"]()
            req(propX and propY)

            # print("---------------------------------------------------")
            # print(f"2D threshold {threshold_id} on props ({propX}, {propY})")

            # depend on selection memory so earlier brushes refresh this plot
            # _ = thresholding_memory_2d_selection.get()

            df = _xy_for_2d_threshold(threshold_id, spot_df, track_df)
            req(not df.empty)
            # print("---------------------------------------------------")
            print(f"Widget df for {threshold_id}")

            current_state |= {"xy": df}
            # print("---------------------------------------------------")
            # print(f"option 1: {state[threshold_id]}")
            state[threshold_id] = current_state
            # print("---------------------------------------------------")
            # print(state)

            thresholds_state.set(state)
            # print("---------------------------------------------------")
            # print(f"Thresholds state: {thresholds_state.get()}")
            

            # current_state.append(df)

            p = (
                ggplot(df, aes(propX, propY))
                + geom_point(
                    size=input.threshold2d_dot_size() if input.threshold2d_dot_size() is not None else 0,
                    alpha=input.threshold2d_dot_opacity() if input.threshold2d_dot_opacity() is not None else 0,
                    color="black",
                    stroke=0
                )
                + coord_equal()
                + scale_x_continuous(
                    limits=(0, 1),
                    breaks=[0, 0.5, 1],
                    labels=["", "", ""],          # show ticks, no labels
                    minor_breaks=[0.25, 0.75],    # set minor ticks
                    expand=(0.035, 0.035),
                )
                + scale_y_continuous(
                    limits=(0, 1),
                    breaks=[0, 0.5, 1],
                    labels=["", "", ""],          # show ticks, no labels
                    minor_breaks=[0.25, 0.75],    # set minor ticks
                    expand=(0.035, 0.035),
                )
                + theme_minimal()
                + theme(
                    # keep titles off
                    axis_title_x=element_blank(),
                    axis_title_y=element_blank(),

                    # hide text; ticks will still render
                    axis_text_x=element_blank(),
                    axis_text_y=element_blank(),

                    # clean panel
                    panel_grid_major=element_blank(),
                    panel_grid_minor=element_blank(),

                    # draw only bottom x and left y axes
                    axis_line_x=element_line(color="#000000", size=0.5),
                    axis_line_y=element_line(color="#000000", size=0.5),

                    # draw major ticks (no minors)
                    axis_ticks_major_x=element_line(color="#000000", size=0.5),
                    axis_ticks_major_y=element_line(color="#000000", size=0.5),
                    axis_ticks_minor_x=element_line(color="#1E1E1E", size=0.2),
                    axis_ticks_minor_y=element_line(color="#1E1E1E", size=0.2),

                    # slightly longer ticks (points)
                    axis_ticks_length=3,
                ) + labs(x=propX, y=propY)
            )
            return p




    # def _uh(id):
    @reactive.effect
    # @reactive.event(threshold_list, input[f"threshold2d_widget_{id}_brush"])
    def uh():
        # print("=============================================================")
        # print(f"Threshold 2D Widget {id} Brush Event Triggered")

        
        
            
            
        for threshold_id in threshold_list.get():
            state = thresholds_state.get()
            if state is None or not isinstance(state, dict):
                break
            # print("--------------------------------------------------------------")
            # print("No state yet")
            print("--------------------------------------------------------------")
            print(f"Processing threshold ID {threshold_id}")
            print(f"State keys: {state.keys()}")
            current_state = state.get(threshold_id)

            # print("--------------------------------------------------------------")
            # print(f"Current state for threshold {threshold_id}: {current_state}")
            req(current_state is not None and isinstance(current_state, dict) and "spots" in current_state and "tracks" in current_state)
            # print(current_state)

            spot_df_input, track_df_input, xy_df = current_state.get("spots"), current_state.get("tracks"), current_state.get("xy")
            req(not spot_df_input.empty and not track_df_input.empty)
            # print("--------------------------------------------------------------")
            # print(xy_df)
            # print(spot_df_input)
            
            # print("Track DataFrame input:")
            # print(track_df_input)

            propX = input[f"thresholding_metric_X_{threshold_id}"]()
            propY = input[f"thresholding_metric_Y_{threshold_id}"]()

            brush = input[f"threshold2d_widget_{threshold_id}_brush"]()


            print("--------------------------------------------------------------")
            print(f"Brush: {brush}")

            if brush is not None and (xy_df is not None and not xy_df.empty):
                # print("--------------------------------------------------------------")
                # print("Brush was not none")

                xmin, xmax = brush.get("xmin"), brush.get("xmax")
                ymin, ymax = brush.get("ymin"), brush.get("ymax")
                print("--------------------------------------------------------------")
                print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

                brushed = xy_df.loc[(xy_df[propX] >= xmin) & (xy_df[propX] <= xmax) & (xy_df[propY] >= ymin) & (xy_df[propY] <= ymax)]

                # print(f"Brushed data:\n{brushed}")

            else:
                print("--------------------------------------------------------------")
                print("Brush was none")

                brushed = track_df_input

            if brushed.empty:
                print("--------------------------------------------------------------")
                print("Brushed data is empty")
                state |= {threshold_id + 1: {"spots": spot_df_input, "tracks": track_df_input}}
            else:
                print("--------------------------------------------------------------")
                print("Brushed data is not empty")
                spot_df_output = spot_df_input.loc[spot_df_input.index.intersection(brushed.index)]
                track_df_output = track_df_input.loc[track_df_input.index.intersection(brushed.index)]
                state |= {threshold_id + 1: {"spots": spot_df_output, "tracks": track_df_output}}

        thresholds_state.set(state)

        # selected_id = input.pass_selected()

        # render_threshold2d_widget(selected_id)



# - - - - Threshold modules management - - - -

    # REMOVED the original first set_threshold_modules() that read a non-existent
    # manual input and caused feedback loops. Its behavior is replaced by the
    # per-threshold sync registered below.


    # def _register_threshold_modules(id):
    @reactive.Effect
    @reactive.event(threshold_list)
    def register_threshold_modules():

        print("===========================================================")

        if threshold_dimension.get() == "1D":
            # Remove outputs for deleted thresholds
            for threshold_id in list(threshold_slider_outputs.keys()):
                if threshold_id not in threshold_list.get():
                    del threshold_slider_outputs[threshold_id]
            # Add outputs for new thresholds
            for threshold_id in threshold_list.get():
                if threshold_id not in threshold_slider_outputs:
                    threshold_slider_outputs[threshold_id] = render_threshold_slider(threshold_id)
                    thresholding_histogram_outputs[threshold_id] = render_threshold_histogram(threshold_id)
                    render_manual_threshold_values_setting(threshold_id)
                    # NEW: keep slider, numerics, and memory in sync (no loops)
                    register_threshold_sync(threshold_id)
        
        elif threshold_dimension.get() == "2D":
            for threshold_id in threshold_list.get():
                # print(f"---------------------------------------------------------")
                # print(f"dicts in thresholds_state: {len(thresholds_state.get())}")
                # print(f"Registering 2D threshold ID {threshold_id}")
                render_threshold2d_widget(threshold_id)

        # @reactive.invalidate_later(1000, session)

    def _update_threshold_module(id):
        @reactive.Effect
        @reactive.event(input[f"pass_selected_{id}"])
        def update_threshold_module():
            print("===========================================================")
            print(f"Updating threshold module {id+1}")
            render_threshold2d_widget(id + 1)


    @reactive.effect
    def _():
        for threshold_id in threshold_list.get():
            _update_threshold_module(threshold_id)
            # _uh(threshold_id)

    @reactive.Effect
    def cache_threshold_selections():

        if threshold_dimension.get() != "1D":
            return
        else:
            _property_selections = {}
            _filter_type_selections = {}
            _quantile_selections = {}
            _reference_selections = {}

            for threshold_id in threshold_list.get():
                property_name = input[f"threshold_property_{threshold_id}"]()
                filter_type = input[f"threshold_filter_{threshold_id}"]()
                quantile = input[f"threshold_quantile_{threshold_id}"]()
                reference = input[f"reference_value_{threshold_id}"]()

                if (
                    isinstance(property_name, str)
                    and property_name is not None
                    and property_name != 0
                    and property_name in Metrics.Thresholding.Properties
                ):
                    _property_selections[threshold_id] = property_name

                if (
                    isinstance(filter_type, str)
                    and filter_type is not None
                    and filter_type != 0
                    and filter_type in Modes.Thresholding
                ):
                    _filter_type_selections[threshold_id] = filter_type

                if quantile is not None:
                    _quantile_selections[threshold_id] = quantile
                    
                if isinstance(reference, (str)) and reference is not None:
                    _reference_selections[threshold_id] = reference

            _property_selections = {tid: val for tid, val in _property_selections.items() if tid in threshold_list.get()}
            property_selections.set(_property_selections)

            _filter_type_selections = {tid: val for tid, val in _filter_type_selections.items() if tid in threshold_list.get()}
            filter_type_selections.set(_filter_type_selections)

            _quantile_selections = {tid: val for tid, val in _quantile_selections.items() if tid in threshold_list.get()}
            quantile_selections.set(_quantile_selections)

            _reference_selections = {tid: val for tid, val in _reference_selections.items() if tid in threshold_list.get()}
            reference_selections.set(_reference_selections)




    @reactive.Effect
    def property_selectize():
        selected = property_selections.get()
        for threshold_id in threshold_list.get():
            select = selected[threshold_id] if threshold_id in selected else Metrics.Thresholding.Properties[0]
            ui.update_selectize(
                id=f"threshold_property_{threshold_id}",
                choices=Metrics.Thresholding.Properties,
                selected=select
            )

    @reactive.Effect
    def filter_type_selectize():
        selected = filter_type_selections.get()
        for threshold_id in threshold_list.get():
            select = selected[threshold_id] if threshold_id in selected else Modes.Thresholding[0]
            ui.update_selectize(
                id=f"threshold_filter_{threshold_id}",
                choices=Modes.Thresholding,
                selected=select
            )

    @reactive.Effect
    def quantile_selectize():
        selected = quantile_selections.get()
        for threshold_id in threshold_list.get():
            if threshold_id not in selected:
                pass
            else:
                select = selected[threshold_id] 
                ui.update_selectize(
                    id=f"threshold_quantile_{threshold_id}",
                    choices=[200, 100, 50, 25, 20, 10, 5, 4, 2],
                    selected=select
                )

    @reactive.Effect
    def reference_selectize():
        selected = reference_selections.get()
        for threshold_id in threshold_list.get():
            select = selected[threshold_id] if threshold_id in selected else "Mean"
            ui.update_selectize(
                id=f"reference_value_{threshold_id}",
                choices=["Mean", "Median", "My own value"],
                selected=select
            )
            # # Update the numeric input for "My own value"
            # if select == "My own value":
            #     my_own_value = input[f"my_own_value_{threshold_id}"]()
            #     ui.update_numeric(
            #         id=f"my_own_value_{threshold_id}",
            #         value=my_own_value if isinstance(my_own_value, (int, float)) else None
            #     )





    # --- Helpers to route properties to the correct table ---
    def _is_spot_prop(prop: str) -> bool:
        return prop in getattr(Metrics.Thresholding, "SpotProperties", [])

    def _is_track_prop(prop: str) -> bool:
        return prop in getattr(Metrics.Thresholding, "TrackProperties", [])

    def _normalized(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if pd.notna(mn) and pd.notna(mx) and mx != mn else pd.Series(0.0, index=s.index)

    def _literal_mask(vals: pd.Series, lo: float, hi: float) -> pd.Series:
        s = pd.to_numeric(vals, errors="coerce")
        return (s >= lo) & (s <= hi)

    def _normalized_mask(vals: pd.Series, lo: float, hi: float) -> pd.Series:
        n = _normalized(vals)
        return (n >= lo) & (n <= hi)

    def _quantile_mask(vals: pd.Series, lo_pct: float, hi_pct: float) -> pd.Series:
        s = pd.to_numeric(vals, errors="coerce").dropna()
        if s.empty:
            return pd.Series(False, index=vals.index)
        qlo = np.quantile(s, lo_pct/100.0)
        qhi = np.quantile(s, hi_pct/100.0)
        base = pd.Series(False, index=vals.index)
        return base.mask((vals >= qlo) & (vals <= qhi), True)

    def _relative_mask(vals: pd.Series, ref_mode: str, my_value: float | None, lo: float, hi: float) -> pd.Series:
        s = pd.to_numeric(vals, errors="coerce")
        if ref_mode == "Mean":
            ref = float(s.mean())
        elif ref_mode == "Median":
            ref = float(s.median())
        elif ref_mode == "My own value" and isinstance(my_value, (int, float)):
            ref = float(my_value)
        else:
            ref = float(s.mean())
        d = (s - ref).abs()
        return (d >= lo) & (d <= hi)





    




    



    # ======================= APPLY THRESHOLDING PIPELINE =======================

    def _read_pair_from_ui(threshold_id: int):
        """Safely read the active slider pair for this threshold."""
        try:
            lo, hi = input[f"threshold_slider_{threshold_id}"]()
            lo = float(lo); hi = float(hi)
            if lo > hi: lo, hi = hi, lo
            return lo, hi
        except Exception:
            return None

    def _active_ctx(threshold_id: int):
        """Read current context for a threshold block."""
        prop, ftype, q, ref, ref_val = _current_context(threshold_id)
        pair = _read_pair_from_ui(threshold_id)
        return prop, ftype, q, ref, ref_val, pair

    @reactive.Effect
    def _build_chain_inputs_and_preview():
        """
        Build chained inputs for every threshold (1D & 2D) and a live preview
        of the final step's output. THRESH_INPUTS[tid] = {"spot": df_in, "track": df_in}.
        THRESH_PREVIEW_OUT = dataframe of the "last" step (shape depends on dimension).
        """
        # start from unfiltered
        spot0 = UNFILTERED_SPOTSTATS.get()
        track0 = UNFILTERED_TRACKSTATS.get()
        if spot0 is None: spot0 = pd.DataFrame()
        if track0 is None: track0 = pd.DataFrame()

        ids = threshold_list.get()
        inputs_map: dict[int, dict[str, pd.DataFrame]] = {}

        if threshold_dimension.get() == "1D":
            # --- unchanged logic for 1D ---
            spot_work = spot0.copy()
            track_work = track0.copy()
            last_df = pd.DataFrame()

            for tid in ids:
                inputs_map[tid] = {"spot": spot_work, "track": track_work}

                prop, ftype, q, ref, ref_val = _current_context(tid)
                pair = _read_pair_from_ui(tid)
                if not (prop and ftype and pair):
                    continue

                # choose base table by property kind
                if _is_spot_prop(prop):
                    base = spot_work
                elif _is_track_prop(prop):
                    base = track_work
                else:
                    continue

                if base.empty or prop not in base.columns:
                    last_df = base.copy()
                    continue

                # build mask
                if ftype == "Literal":
                    mask = _literal_mask(base[prop], pair[0], pair[1])
                elif ftype == "Normalized 0-1":
                    mask = _normalized_mask(base[prop], pair[0], pair[1])
                elif ftype == "Quantile":
                    lo_pct, hi_pct = pair
                    mask = _quantile_mask(base[prop], lo_pct, hi_pct)
                elif ftype == "Relative to...":
                    mask = _relative_mask(base[prop], ref, ref_val, pair[0], pair[1])
                else:
                    mask = pd.Series(True, index=base.index)

                # apply to the correct working table
                if _is_spot_prop(prop):
                    spot_work = base.loc[mask].copy()
                    last_df = spot_work
                else:
                    track_work = base.loc[mask].copy()
                    last_df = track_work

            THRESH_PREVIEW_OUT.set(last_df if not last_df.empty else (spot_work if not spot_work.empty else track_work))

        else:
            # --- 2D chain via lasso selections (Track-ID based) ---
            mem = thresholding_memory_2d_selection.get()  # { tid: { (propX,propY): set([track_id,...]) } }

            spot_work = spot0.copy()
            track_work = track0.copy()

            selected_tids: set | None = None
            final_df = pd.DataFrame()

            for tid in ids:
                # inputs for this step are whatever survived so far
                inputs_map[tid] = {"spot": spot_work, "track": track_work}

                propX = input[f"thresholding_metric_X_{tid}"]()
                propY = input[f"thresholding_metric_Y_{tid}"]()
                if not (propX and propY):
                    continue

                # Build current XY (normalized 0..1) from the *chained* inputs
                xy = Threshold.JoinByIndex(
                    _get_series(propX, spot_work, track_work), 
                    _get_series(propY, spot_work, track_work)
                )
                if xy.empty:
                    continue

                # Selection set stored for this (tid, pair) — it's a set of Track IDs
                sel_tids = mem.get(tid, {}).get((propX, propY), set())

                # If no selection at this step, treat as pass-through for the tracks present in 'xy'
                step_tids = set(xy.index) if not sel_tids else (sel_tids & set(xy.index))

                # Chain with previous steps
                selected_tids = step_tids if selected_tids is None else (selected_tids & step_tids)

                # Early exit if emptied
                if selected_tids is not None and not selected_tids:
                    final_df = xy.iloc[0:0]
                    break

                # Update working tables for the next thresholds: restrict by INDEX only
                if selected_tids:
                    if not spot_work.empty:
                        spot_work = spot_work.loc[spot_work.index.isin(selected_tids)].copy()
                    if not track_work.empty:
                        track_work = track_work.loc[track_work.index.isin(selected_tids)].copy()

                # Keep a view shaped like XY for live preview (INDEX filter)
                final_df = (xy.loc[xy.index.isin(selected_tids)].copy() if selected_tids else xy.copy())

            THRESH_PREVIEW_OUT.set(final_df)

        # expose per-threshold chained inputs
        THRESH_INPUTS.set(inputs_map)





    def _apply_1d_pipeline() -> pd.DataFrame:
        """
        Sequentially apply all 1D thresholds.
        Each step consumes the previous step's filtered table (spot OR track, depending on the property).
        The final dataframe from the last threshold is returned.
        """
        spot_df = UNFILTERED_SPOTSTATS.get().copy()
        track_df = UNFILTERED_TRACKSTATS.get().copy()
        if spot_df is None: spot_df = pd.DataFrame()
        if track_df is None: track_df = pd.DataFrame()

        last_df = pd.DataFrame()
        for tid in threshold_list.get():
            prop, ftype, q, ref, ref_val, pair = _active_ctx(tid)
            if not (prop and ftype and pair):
                # nothing to apply at this step -> keep last_df as-is
                continue

            # Route to the currently active working table for this property
            if _is_spot_prop(prop):
                base = spot_df
            elif _is_track_prop(prop):
                base = track_df
            else:
                continue

            if base.empty or prop not in base.columns:
                # property missing in current working set
                last_df = base.copy()
                continue

            # Build mask by filter type
            if ftype == "Literal":
                mask = _literal_mask(base[prop], pair[0], pair[1])
            elif ftype == "Normalized 0-1":
                mask = _normalized_mask(base[prop], pair[0], pair[1])
            elif ftype == "Quantile":
                lo_pct, hi_pct = pair
                mask = _quantile_mask(base[prop], lo_pct, hi_pct)
            elif ftype == "Relative to...":
                mask = _relative_mask(base[prop], ref, ref_val, pair[0], pair[1])
            else:
                mask = pd.Series(True, index=base.index)

            # Apply mask to the appropriate working table
            if _is_spot_prop(prop):
                spot_df = base.loc[mask].copy()
                last_df = spot_df
            else:
                track_df = base.loc[mask].copy()
                last_df = track_df

        return last_df if not last_df.empty else (spot_df if not spot_df.empty else track_df)

    def _apply_2d_pipeline() -> pd.DataFrame:
        """
        Use the stored lasso selections per threshold and metric pair.
        Chains by intersection of Track IDs.
        Returns an XY-shaped preview of the *last* step.
        """
        spot_df = UNFILTERED_SPOTSTATS.get()
        track_df = UNFILTERED_TRACKSTATS.get()
        mem = thresholding_memory_2d_selection.get()

        final_df = pd.DataFrame()
        selected_tids: set | None = None

        for tid in threshold_list.get():
            propX = input[f"thresholding_metric_X_{tid}"]()
            propY = input[f"thresholding_metric_Y_{tid}"]()
            if not (propX and propY):
                continue

            xy = Threshold.JoinByIndex(
                _get_series(propX, spot_df, track_df), 
                _get_series(propY, spot_df, track_df)
            )
            if xy.empty:
                continue

            sel = mem.get(tid, {}).get((propX, propY), set())
            # If no selection at this step, treat as pass-through
            step_tids = set(xy['Track ID']) if not sel else (sel & set(xy['Track ID']))

            selected_tids = step_tids if selected_tids is None else (selected_tids & step_tids)
            if selected_tids is not None and not selected_tids:
                final_df = xy.iloc[0:0]
                break

            # Keep a view shaped like XY for live preview
            if selected_tids:
                final_df = xy.loc[pd.Series(xy['Track ID']).isin(selected_tids).to_numpy()]
            else:
                final_df = xy.copy()

        return final_df


    def _apply_thresholds():
        if threshold_dimension.get() == "1D":
            out = _apply_1d_pipeline()
        else:
            out = _apply_2d_pipeline()

        # Store into the universal output
        THRESH_INPUTS.set(out)

    # Enable the Apply button when there is something to apply
    # @reactive.Effect
    # def _enable_apply_threshold():
    #     has_any_data = not (UNFILTERED_SPOTSTATS.get().empty and UNFILTERED_TRACKSTATS.get().empty)
    #     session.send_input_message("apply_threshold", {"disabled": not has_any_data})


    


    # Tie to the UI button
    @reactive.Effect
    @reactive.event(input.apply_threshold)
    def _on_apply_threshold():
        # Take the currently chained live output and make it official
        THRESHOLDED_DF.set(THRESH_PREVIEW_OUT.get())

    # def render_mfkin_table_bruh(threshold_id):
    #     # @output("in_brush")
    #     @render.table
    #     def in_brush(threshold_id=threshold_id):
    #         # return brushed_points(
    #         #     ugh.get(),
    #         #     input.threshold_widget_0_brush(),
    #         #     all_rows=True
    #         # )

    #         fuck = input[f"threshold2d_widget_{threshold_id}_brush"]()
    #         propx = input[f"thresholding_metric_X_{threshold_id}"]()
    #         propy = input[f"thresholding_metric_Y_{threshold_id}"]()

    #         xmin, xmax = (fuck["xmin"], fuck["xmax"]) if isinstance(fuck, dict) else (fuck.xmin, fuck.xmax)
    #         ymin, ymax = (fuck["ymin"], fuck["ymax"]) if isinstance(fuck, dict) else (fuck.ymin, fuck.ymax)

    #         df = ugh.get()

    #         sel = df[(df[propx] >= xmin) & (df[propx] <= xmax) & (df[propy] >= ymin) & (df[propy] <= ymax)]
    #         return sel.reset_index()
        
    # @reactive.Effect
    # # @reactive.event(threshold_list)
    # def _on_threshold_list_change():
    #     for tid in threshold_list.get():
    #         render_mfkin_table_bruh(tid)

    
    # @render.table
    # def in_brush():
    #     fuck = input.threshold2d_widget_0_brush()
    #     propx = input.thresholding_metric_X_0()
    #     propy = input.thresholding_metric_Y_0()

    #     xmin, xmax = (fuck["xmin"], fuck["xmax"]) if isinstance(fuck, dict) else (fuck.xmin, fuck.xmax)
    #     ymin, ymax = (fuck["ymin"], fuck["ymax"]) if isinstance(fuck, dict) else (fuck.ymin, fuck.ymax)

    #     df = ugh.get()

    #     sel = df[(df[propx] >= xmin) & (df[propx] <= xmax) & (df[propy] >= ymin) & (df[propy] <= ymax)]
    #     return sel.reset_index()

    # - - - - Rendering Data Frames - - - -
    
    @render.data_frame
    def render_spot_stats():
        spot_stats = THRESHOLDED_DF.get()
        if spot_stats is not None or not spot_stats:
            return spot_stats
        else:
            pass

    @render.data_frame
    def render_track_stats():
        track_stats = UNFILTERED_TRACKSTATS.get()
        if track_stats is not None or not track_stats:
            return track_stats
        else:
            pass

    @render.data_frame
    def render_time_stats():
        time_stats = UNFILTERED_TIMESTATS.get()
        if time_stats is not None or not time_stats:
            return time_stats
        else:
            pass
    
    # - - - - - - - - - - - - - - - - - - - -



    # (Other outputs and logic remain unchanged...)

# --- Mount the app ---
app = App(app_ui, server)

# TODO - Keep all the raw data (columns) - rather format them (stripping of _ and have them not all caps)
# TODO - Make the 2D filtering logic work on the same logic as does the D filtering logic
# TODO - make both 1D and 2D thresholding operational
# TODO - Make it possible to save/load threshold configurations
# TODO - Find a way to program all the functions so that functions do not refresh/re-render unnecessarily on just any reactive action
# TODO - Time point definition
# TODO - Make it possible for the user to title their charts
# TODO - Make it possible for the user to manually set threshold values
# TODO - Mean directional change rate
# TODO - Select which p-tests should be shown in the superplot chart
# TODO - P-test

