from shiny import App, Inputs, Outputs, Session, render, reactive, req, ui
from shiny.types import FileInfo
from shinywidgets import render_plotly, render_altair, output_widget, render_widget

from utils.Select import Metrics, Styles, Markers, Modes
from utils.Function import DataLoader, Process, Calc
from utils.ratelimit import debounce, throttle
from utils.Customize import Format

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objs as go

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
        ui.input_task_button("apply_threshold", label="Apply thresholding", label_busy="Applying...", type="secondary", disabled=True),
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
    threshold_dimension = reactive.Value("1D")
    dimension_button_label = reactive.Value("2D")
    threshold_list = reactive.Value([0])  # Start with one threshold
    property_selections = reactive.Value({})
    filter_type_selections = reactive.Value({})
    quantile_selections = reactive.Value({})
    reference_selections = reactive.Value({})
    metric_x_selections = reactive.Value({})
    metric_y_selections = reactive.Value({})
    threshold_slider_outputs = {}
    thresholding_histogram_outputs = {}

    thresholding_memory = reactive.Value({})  # initialize empty first

    @reactive.effect
    @reactive.event(threshold_list)
    def initialize_thresholding_memory():
        memory = thresholding_memory.get()
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
        thresholding_memory.set(memory)



    # - - - - Data frame placeholders - - - -
    RAWDATA = reactive.Value(pd.DataFrame())         # Placeholder for raw data
    UNFILTERED_SPOTSTATS = reactive.Value(pd.DataFrame())    # Placeholder for spot statistics
    UNFILTERED_TRACKSTATS = reactive.Value(pd.DataFrame())   # Placeholder for track statistics
    UNFILTERED_TIMESTATS = reactive.Value(pd.DataFrame())    # Placeholder for time statistics
    SPOTSTATS = reactive.Value(pd.DataFrame())       # Placeholder for processed spot statistics
    TRACKSTATS = reactive.Value(pd.DataFrame())      # Placeholder for processed track statistics
    TIMESTATS = reactive.Value(pd.DataFrame())       # Placeholder for processed time statistics



    # - - - - File input management - - - -

    @reactive.effect
    @reactive.event(input.add_input)
    def add_input():
        ids = input_list.get()
        new_id = max(ids) + 1 if ids else 1
        input_list.set(ids + [new_id])
        session.send_input_message("remove_input", {"disabled": len(ids) < 1})

    @reactive.effect
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

    @reactive.effect
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

    @reactive.effect
    def enable_run_button():
        files_uploaded = [input[f"input_file{idx}"]() for idx in input_list.get()]
        def is_busy(val):
            return isinstance(val, list) and len(val) > 0
        all_busy = all(is_busy(f) for f in files_uploaded)
        session.send_input_message("run", {"disabled": not all_busy})

    @reactive.effect
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
            RAWDATA.set(pd.concat(all_data, axis=0, ignore_index=True))
            UNFILTERED_SPOTSTATS.set(Calc.Spots(RAWDATA.get()))
            UNFILTERED_TRACKSTATS.set(Calc.Tracks(RAWDATA.get()))
            UNFILTERED_TIMESTATS.set(Calc.Time(RAWDATA.get()))
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

    @reactive.effect
    @reactive.event(input.add_threshold)
    def add_threshold():
        ids = threshold_list.get()
        threshold_list.set(ids + [max(ids)+1 if ids else 0])
        session.send_input_message("remove_threshold", {"disabled": False})

    @reactive.effect
    @reactive.event(input.remove_threshold)
    def remove_threshold():
        ids = threshold_list.get()
        if len(ids) > 1:
            threshold_list.set(ids[:-1])
        if len(threshold_list.get()) <= 1:
            session.send_input_message("remove_threshold", {"disabled": True})


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
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                    ui.markdown(""" <p> """),
                    ui.input_numeric("bins", "Number of bins", value=25, min=1, step=1),
                ),
            )
        elif threshold_dimension.get() == "2D":
            for i, threshold_id in enumerate(ids, 1):
                panels.append(
                    ui.accordion_panel(
                        f"Threshold {i}" if len(ids) >= 2 else "Threshold",
                        ui.panel_well(
                            ui.markdown(""" <h6>  Properties X;Y  </h6>"""),
                            ui.input_selectize(f"thresholding_metric_X_{threshold_id}", None, Metrics.Thresholding.Properties),
                            ui.input_selectize(f"thresholding_metric_Y_{threshold_id}", None, Metrics.Thresholding.Properties),
                            # For now only normalized view; more modes later if desired

                            # Plotly output (FigureWidget) and a clear-selection button
                            ui.div(
                                {"style": "position:relative; width:100%; padding-top:100%;"},
                                ui.div(
                                    {"style": "position:absolute; inset:0;"},
                                    output_widget(f"threshold2d_plot_{threshold_id}")
                                )
                            ),
                            ui.input_action_button(f"threshold2d_clear_{threshold_id}", "Clear selection", class_="btn-secondary"),
                        ),
                    ),
                ),
            panels.append(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                    ui.markdown(""" <p> """),
                    ui.markdown("""  Working on it dawg  """),
                ),
            ),
        return ui.accordion(*panels, id="thresholds_accordion", open=["Threshold", f"Threshold {len(ids)}", "Filter settings"])
    
    
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

            spot_data = UNFILTERED_SPOTSTATS.get()
            track_data = UNFILTERED_TRACKSTATS.get()
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
                memory=thresholding_memory.get(),
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

            spot_data = UNFILTERED_SPOTSTATS.get()
            track_data = UNFILTERED_TRACKSTATS.get()
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
                memory=thresholding_memory.get(),
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
            if input[f"threshold_property_{threshold_id}"]() in Metrics.Thresholding.SpotProperties:
                data = UNFILTERED_SPOTSTATS.get()
            if input[f"threshold_property_{threshold_id}"]() in Metrics.Thresholding.TrackProperties:
                data = UNFILTERED_TRACKSTATS.get()
            if data is None or data.empty:
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
    def cache_2d_metric_selections():
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

    @reactive.effect
    @reactive.event(threshold_list)
    def normalize_2d_selection_memory():
        mem = thresholding_memory_2d_selection.get()
        keep_ids = set(threshold_list.get())
        # drop removed thresholds only; keep all (propX,propY) pairs for survivors
        new_mem = {tid: pairs for tid, pairs in mem.items() if tid in keep_ids}
        if new_mem != mem:
            thresholding_memory_2d_selection.set(new_mem)



    def _last_or_default_metric(saved: dict):
        if saved:
            # take value from the highest existing threshold id
            return saved[sorted(saved.keys())[-1]]
        return Metrics.Thresholding.Properties[0] if Metrics.Thresholding.Properties else None

    @reactive.effect
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
        # A) slider -> memory
        @reactive.effect
        @reactive.event(input[f"threshold_slider_{threshold_id}"])
        def _slider_to_memory():
            vals = input[f"threshold_slider_{threshold_id}"]()
            if not (isinstance(vals, (tuple, list)) and len(vals) == 2 and all(v is not None for v in vals)):
                return

            prop, ftype, q, ref, ref_val = _current_context(threshold_id)
            if not (prop and ftype):
                return

            mem = thresholding_memory.get()
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
                thresholding_memory.set(new_mem)


        # B) manual numerics -> memory
        @reactive.effect
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

            mem = thresholding_memory.get()
            cur_vals, cur_ref_val = _read_stored_pair(mem, threshold_id, prop, ftype, q, ref)

            new_pair = (lo, hi)

            if cur_vals is None or not _nearly_equal_pair(new_pair, cur_vals):
                thresholding_memory.set(
                    _set_threshold_memory(mem.copy(), threshold_id, prop, ftype, new_pair, quantile=q, reference=ref, ref_val=ref_val)
                )

        # C) memory -> UI (push only when different)
        @reactive.effect
        def _memory_to_ui():
            prop, ftype, q, ref, ref_val = _current_context(threshold_id)
            if not (prop and ftype):
                return

            mem = thresholding_memory.get()

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
        @reactive.effect
        @reactive.event(input[f"my_own_value_{threshold_id}"])
        def _myown_to_memory():
            prop, ftype, q, ref, ref_val = _current_context(threshold_id)
            if not (prop and ftype and ref == "My own value"):
                return

            # Ignore non-numeric or None
            if not isinstance(ref_val, (int, float)):
                return

            mem = thresholding_memory.get()

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
            thresholding_memory.set(new_mem)



    # ======================= 2D THRESHOLDING (LASSO) =======================

    # Memory: selected original-row indices for each 2D threshold and metric pair
    # Shape: { tid: { (propX, propY): set([...original row indices...]) } }
    thresholding_memory_2d_selection = reactive.Value({})

    def _normalize_0_1(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        vmin, vmax = s.min(), s.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            return pd.Series(0.0, index=s.index, dtype=float)
        return (s - vmin) / (vmax - vmin)

    def _get_metric_series(prop: str, spot_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.Series:
        if prop in Metrics.Thresholding.SpotProperties and not spot_df.empty:
            return spot_df[prop]
        if prop in Metrics.Thresholding.TrackProperties and not track_df.empty:
            return track_df[prop]
        return pd.Series(dtype=float)

    def _get_2d_selected_set(mem: dict, tid: int, propX: str, propY: str) -> set:
        try:
            return set(mem[tid][(propX, propY)])
        except Exception:
            return set()

    def _set_2d_selected_set(mem: dict, tid: int, propX: str, propY: str, idx_set: set) -> dict:
        mem = mem.copy()
        mem.setdefault(tid, {})
        mem[tid][(propX, propY)] = set(idx_set)
        return mem

    def _mask_2d_from_selection(spot_df: pd.DataFrame, track_df: pd.DataFrame,
                                propX: str, propY: str, selected_idx: set) -> tuple[pd.Index, pd.Series]:
        # Build a joint table to align indices that have both metrics
        X_raw = _get_metric_series(propX, spot_df, track_df)
        Y_raw = _get_metric_series(propY, spot_df, track_df)
        tbl = pd.DataFrame({"_x": X_raw, "_y": Y_raw}).dropna()
        if tbl.empty:
            return tbl.index, pd.Series([], dtype=bool, index=tbl.index)
        # Selection is expressed in terms of ORIGINAL indices of tbl
        mask = tbl.index.to_series().isin(selected_idx)
        return tbl.index, mask

 
    def render_threshold2d_plot(threshold_id: int):
        @output(id=f"threshold2d_plot_{threshold_id}")
        @render_widget
        def threshold2d_plot():  # id must match output_widget id
            spot_df = UNFILTERED_SPOTSTATS.get()
            track_df = UNFILTERED_TRACKSTATS.get()
            if spot_df is None or track_df is None or (spot_df.empty and track_df.empty):
                return go.FigureWidget()

            propX = input[f"thresholding_metric_X_{threshold_id}"]()
            propY = input[f"thresholding_metric_Y_{threshold_id}"]()

            if not propX or not propY:
                return go.FigureWidget()

            # Build aligned table and normalize both axes to 0..1
            X_raw = _get_metric_series(propX, spot_df, track_df)
            Y_raw = _get_metric_series(propY, spot_df, track_df)
            tbl = pd.DataFrame({"_x": X_raw, "_y": Y_raw}).dropna()
            if tbl.empty:
                return go.FigureWidget()

            X = _normalize_0_1(tbl["_x"])
            Y = _normalize_0_1(tbl["_y"])

            # Recover previously selected points for this (propX, propY)
            mem = thresholding_memory_2d_selection.get()
            selected_set = _get_2d_selected_set(mem, threshold_id, propX, propY)

            # Map selected original indices -> trace point indices (0..N-1)
            # We'll keep original row index mapping on the widget to use in callbacks
            row_index = tbl.index.to_numpy()
            # selectedpoints expects positions in trace arrays
            selectedpoints = np.nonzero(np.isin(row_index, list(selected_set)))[0].tolist()

            # Build FigureWidget
            w = go.FigureWidget(
                data=[
                    go.Scattergl(
                        x=X.values, y=Y.values, mode="markers",
                        marker=dict(size=2.5, color="grey"),
                        selected=dict(marker=dict(color="#337ab7")),
                        unselected=dict(marker=dict(color="lightgrey")),
                        selectedpoints=selectedpoints,  # reflect memory in the view
                        hoverinfo="skip",
                    )
                ],
                layout=go.Layout(
                    autosize=True,
                    height=150,
                    width=150,
                    margin=dict(l=0, r=0, t=0, b=10),
                    xaxis=dict(
                        range=[-0.02, 1.02],
                        scaleanchor="y",          # lock x to y → 1:1 aspect
                        constrain="domain",       # keep axes inside plotting area
                        showgrid=False, showticklabels=False, ticks="", title=None, zeroline=False
                    ),
                    yaxis=dict(
                        range=[-0.02, 1.02],
                        constrain="domain",
                        showgrid=False, showticklabels=False, ticks="", title=None, zeroline=False
                    ),
                    dragmode="lasso",
                    paper_bgcolor="#f5f5f5",
                    plot_bgcolor="white",
                ),
            )

            # Stash mapping and the props on the widget for the callback
            w._row_index = row_index
            w._tid = threshold_id
            w._propX = propX
            w._propY = propY

            # Selection callback (lasso or box)
            def _on_selection(trace, points, state):
                # points.point_inds are the integer positions into this trace's x/y arrays
                inds = points.point_inds or []
                if len(inds) == 0:
                    # No selection: do not force to empty; let user click 'Clear selection'
                    return

                # Map trace positions -> original row indices
                sel_rows = set(w._row_index[np.array(inds, dtype=int)])
                cur = thresholding_memory_2d_selection.get()
                new_mem = _set_2d_selected_set(cur, w._tid, w._propX, w._propY, sel_rows)
                thresholding_memory_2d_selection.set(new_mem)

            w.data[0].on_selection(_on_selection)  # uses Plotly FigureWidget API
            return w

        # wire render function to specific id
        threshold2d_plot._id = f"threshold2d_plot_{threshold_id}"


    def register_threshold2d_clear(threshold_id: int):
        @reactive.effect
        @reactive.event(input[f"threshold2d_clear_{threshold_id}"])
        def _clear():
            propX = input[f"thresholding_metric_X_{threshold_id}"]()
            propY = input[f"thresholding_metric_Y_{threshold_id}"]()
            if not propX or not propY:
                return
            mem = thresholding_memory_2d_selection.get()
            # Clearing means: select-all (no restriction) or truly empty?
            # Here we choose "select-all": remove the key so no 2D filter is applied.
            if threshold_id in mem and (propX, propY) in mem[threshold_id]:
                mem = mem.copy()
                mem[threshold_id].pop((propX, propY), None)
                thresholding_memory_2d_selection.set(mem)

            # Also tell the live widget (if mounted) to clear selectedpoints
            # Grab the last-rendered widget via the output function (has .widget)
            try:
                out = globals()[f"threshold2d_plot"]  # the render function above
                if hasattr(out, "widget") and out.widget is not None:
                    out.widget.data[0].selectedpoints = []
            except Exception:
                pass







    # - - - - Threshold modules management - - - -

    # REMOVED the original first set_threshold_modules() that read a non-existent
    # manual input and caused feedback loops. Its behavior is replaced by the
    # per-threshold sync registered below.

    @reactive.effect
    @reactive.event(threshold_list)
    def register_threshold_sliders():
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
                # NEW: mount 2D plot + clear handler
                render_threshold2d_plot(threshold_id)
                register_threshold2d_clear(threshold_id)


    @reactive.Effect
    def cache_threshold_selections():
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
            # if reference == "My own value":
            #     my_own_value = input[f"my_own_value_{threshold_id}"]()
            #     if isinstance(my_own_value, (int, float)) and my_own_value is not None:
            #         _reference_selections[threshold_id] = float(my_own_value)

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




    



    

    # - - - - Rendering Data Frames - - - -
    
    @render.data_frame
    def render_spot_stats():
        spot_stats = UNFILTERED_SPOTSTATS.get()
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


# TODO Time point definition
# TODO Make it possible for the user to title their charts 
# TODO Make it possible for the user to manually set threshold values
# TODO Mean directional change rate
# TODO Select which p-tests should be shown in the superplot chart
# TODO P-test 