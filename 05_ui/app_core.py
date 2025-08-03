from shiny import App, Inputs, Outputs, Session, render, reactive, req, ui
from shiny.types import FileInfo
from shinywidgets import render_plotly, render_altair

from utils.Select import Metrics, Styles, Markers, Modes
from utils.Function import DataLoader, Process, Calc
from utils.ratelimit import debounce, throttle
from utils.Customize import Format

import asyncio
import pandas as pd

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
        id="sidebar", open="open", position="right", bg="f8f8f8",
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
    slider_values = reactive.Value({})
    property_selections = reactive.Value({})
    filter_type_selections = reactive.Value({})
    threshold_slider_outputs = {}

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
                        "Quantile": {"quantiles": None, "values": None},
                        "Relative to...": {"reference": None, "values": None},
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
            spot_stats = Calc.Spots(RAWDATA.get())
            UNFILTERED_SPOTSTATS.set(spot_stats)
            UNFILTERED_TRACKSTATS.set(Calc.Tracks(spot_stats))
            UNFILTERED_TIMESTATS.set(Calc.Time(spot_stats))
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
                                ui.input_selectize(f"threshold_quantile_{threshold_id}", "Quantile", choices=[200, 100, 50, 25, 20, 10, 5, 2], selected=100),
                            ),
                            ui.panel_conditional(
                                f"input.threshold_filter_{threshold_id} == 'Relative to...'",
                                ui.input_selectize(f"reference_value_{threshold_id}", "Reference value", choices=["Mean", "Median", "Min", "Max", "My own value"]),
                                ui.panel_conditional(
                                    f"input.reference_value_{threshold_id} == 'My own value'",
                                    ui.input_numeric(f"my_own_value_{threshold_id}", "My own value", value=0, step=1)
                                ),
                            ),
                            ui.output_ui(f"threshold_slider_placeholder_{threshold_id}"),
                        ),
                    ),
                )
            panels.append(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                    ui.markdown(""" <p> """),
                    ui.panel_conditional(
                        " || ".join([f"input.threshold_filter_{tid} != 'Quantile'" for tid in ids]),
                        ui.input_numeric("bins", "Number of bins", value=40, min=1, step=1),
                    ),
                    ui.input_radio_buttons("plot_distribution", "Histogram show:", choices=["Kernel density", "Hover info"], selected="Kernel density"),
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
                            ui.input_selectize(f"thresholding_filter_2D_{threshold_id}", "Thresholding values", ["Literal", "Normalized 0-1"]),
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

    def _get_threshold_memory(dict_memory, threshold_id, property_name, filter_type, default, quantile=None, reference=None):
        """
        Returns the stored tuple for the slider or the default if not yet set or invalid.
        """

        if filter_type == "Literal" or filter_type == "Normalized 0-1":
            try:
                val = dict_memory[threshold_id][property_name][filter_type]["values"]
                if (
                    isinstance(val, (tuple, list))
                    and len(val) == 2
                    and all(isinstance(x, (int, float)) for x in val)
                ):
                    return val
                else:
                    return default
            except Exception:
                return default
            
        elif filter_type == "Quantile":
            try:
                val = dict_memory[threshold_id][property_name][filter_type]["values"]
                quantile = dict_memory[threshold_id][property_name][filter_type]["quantile"]
                if (
                    isinstance(val, (tuple, list))
                    and len(val) == 2
                    and all(isinstance(x, (int, float)) for x in val)
                ) and (
                    isinstance(quantile, (int, float))
                    and quantile is not None
                ):
                    return val, quantile
                else:
                    return default, quantile
            except Exception:
                return default, quantile
            
        elif filter_type == "Relative to...":
            try:
                val = dict_memory[threshold_id][property_name][filter_type]["values"]
                reference = dict_memory[threshold_id][property_name][filter_type]["reference"]
                if (
                    isinstance(val, (tuple, list))
                    and len(val) == 2
                    and all(isinstance(x, (int, float)) for x in val)
                ) and (
                    isinstance(reference, (int, float))
                    and reference is not None
                ):
                    return val, reference
                else:
                    return default, 0
            except Exception:
                return default, 0


    def _set_threshold_memory(dict_memory, threshold_id, property_name, filter_type, values, quantile=None, reference=None):
        """
        Sets the values in the nested dict.
        """

        if filter_type == "Literal" or filter_type == "Normalized 0-1":
            if threshold_id not in dict_memory:
                dict_memory[threshold_id] = {}
            if property_name not in dict_memory[threshold_id]:
                dict_memory[threshold_id][property_name] = {}
            if filter_type not in dict_memory[threshold_id][property_name]:
                dict_memory[threshold_id][property_name][filter_type] = {}
            dict_memory[threshold_id][property_name][filter_type]["values"] = tuple(values)
        
        elif filter_type == "Quantile":
            if threshold_id not in dict_memory:
                dict_memory[threshold_id] = {}
            if property_name not in dict_memory[threshold_id]:
                dict_memory[threshold_id][property_name] = {}
            if filter_type not in dict_memory[threshold_id][property_name]:
                dict_memory[threshold_id][property_name][filter_type] = {}
            dict_memory[threshold_id][property_name][filter_type]["quantile"] = float(quantile)
            dict_memory[threshold_id][property_name][filter_type]["values"] = tuple(values)
            
        elif filter_type == "Relative to...":
            if threshold_id not in dict_memory:
                dict_memory[threshold_id] = {}
            if property_name not in dict_memory[threshold_id]:
                dict_memory[threshold_id][property_name] = {}
            if filter_type not in dict_memory[threshold_id][property_name]:
                dict_memory[threshold_id][property_name][filter_type] = {}
            dict_memory[threshold_id][property_name][filter_type]["reference"] = float(reference) 
            dict_memory[threshold_id][property_name][filter_type]["values"] = tuple(values) 
                
        return dict_memory


    # - - - - Threshold slider generator - - - -

    def _get_steps(lowest, highest):
        """
        Returns the step size for the slider based on the range.
        """
        if highest - lowest < 0.01:
            steps = 0.0001
        elif highest - lowest < 0.1:
            steps = 0.001
        elif highest - lowest < 1:
            steps = 0.01
        elif highest - lowest < 10:
            steps = 0.1
        elif highest - lowest < 100:
            steps = 1
        elif highest - lowest < 1000:
            steps = 10
        elif highest - lowest < 10000:
            steps = 100
        else:
            steps = 1

        return steps

    # Make threshold sliders dynamically based on the threshold ID
    def make_threshold_slider(threshold_id):
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

            memory = thresholding_memory.get()

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

                steps = _get_steps(lowest, highest)
                
                # Use memory if valid, otherwise use defaults
                # memory = thresholding_memory.get()
                values = _get_threshold_memory(memory, threshold_id, property_name, filter_type, default)

                slider = ui.input_slider(
                    f"threshold_slider_{threshold_id}",
                    None,
                    min=Process.Round(lowest, 3),
                    max=Process.Round(highest, 3),
                    value=values,
                    step=steps
                )


            elif filter_type == "Normalized 0-1":
                lowest, highest = 0, 1
                default = (lowest, highest)

                # Use memory if valid, otherwise use defaults
                # memory = thresholding_memory.get()
                values = _get_threshold_memory(memory, threshold_id, property_name, filter_type, default)

                slider = ui.input_slider(
                    f"threshold_slider_{threshold_id}",
                    None,
                    min=0,
                    max=1,
                    value=values,
                    step=0.01
                )


            elif filter_type == "Quantile":
                quantile = input[f"threshold_quantile_{threshold_id}"]()
                lowest, highest = 0, quantile
                default = (lowest, highest)

                # Use memory if valid, otherwise use defaults
                # memory = thresholding_memory.get()
                values, quantile = _get_threshold_memory(memory, threshold_id, property_name, filter_type, default, quantile=quantile)

                slider = ui.input_slider(
                    f"threshold_slider_{threshold_id}",
                    None,
                    min=0,
                    max=100,
                    value=values,
                    step=100/float(quantile),
                )

            elif filter_type == "Relative to...":
                reference = input[f"reference_value_{threshold_id}"]()
                if property_name in Metrics.Thresholding.SpotProperties:
                    min = spot_data[property_name].min()
                    max = spot_data[property_name].max()
                elif property_name in Metrics.Thresholding.TrackProperties:
                    min = track_data[property_name].min()
                    max = track_data[property_name].max()

                if reference == "Mean":
                    lowest = spot_data[property_name].mean()
                    if abs(lowest - min) < abs(max - lowest):
                        highest = (max - lowest)
                    else:
                        highest = lowest + (lowest - min)
                    default = (lowest, highest)

                elif reference == "Median":
                    lowest = spot_data[property_name].median()
                    if abs(lowest - min) < abs(max - lowest):
                        highest = (max - lowest)
                    else:
                        highest = lowest + (lowest - min)
                    default = (lowest, highest)

                elif reference == "Min":
                    lowest = min
                    highest = max
                    default = (lowest, highest)

                elif reference == "Max":
                    lowest = min
                    highest = max
                    default = (lowest, highest)

                elif reference == "My own value":
                    try:
                        reference = float(input[f"my_own_value_{threshold_id}"]())
                        if reference is None or not isinstance(reference, (int, float)):
                            reference = 0
                    except Exception:
                        reference = 0
                    
                    
                    if abs(reference - min) < abs(max - reference):
                        highest = (max - reference)
                    elif abs(reference - min) > abs(max - reference):
                        highest = (reference - min)

                    if reference < min or reference > max:
                        highest = max - min
                        
                    lowest = 0
                    default = (lowest, highest)

                steps = _get_steps(lowest, highest)

                values, reference = _get_threshold_memory(memory, threshold_id, property_name, filter_type, default, reference=reference)

                slider = ui.input_slider(
                    f"threshold_slider_{threshold_id}",
                    None,
                    min=Process.Round(lowest, 3),
                    max=Process.Round(highest, 3),
                    value=values,
                    step=steps
                )
                

            
            return slider
        

    # - - - - Threshold modules management - - - -

    @reactive.Effect
    def set_threshold_modules():
        memory = thresholding_memory.get()
        changed = False
        for threshold_id in threshold_list.get():
            property_name = input[f"threshold_property_{threshold_id}"]()
            slider_vals = input[f"threshold_slider_{threshold_id}"]()
            filter_type = input[f"threshold_filter_{threshold_id}"]()

            if filter_type == "Literal" or filter_type == "Normalized 0-1":
                if (
                    isinstance(slider_vals, (tuple, list))
                    and len(slider_vals) == 2
                    and all(x is not None for x in slider_vals)
                    and property_name
                    and filter_type
                ):
                    new_memory = _set_threshold_memory(memory, threshold_id, property_name, filter_type, slider_vals)
                    if new_memory != memory:
                        memory = new_memory
                        changed = True
                if changed:
                    thresholding_memory.set(memory)
                    return
                
            elif filter_type == "Quantile":
                quantile = input[f"threshold_quantile_{threshold_id}"]()
                if (
                    isinstance(slider_vals, (tuple, list))
                    and len(slider_vals) == 2
                    and all(x is not None for x in slider_vals)
                    and isinstance(quantile, (int, float))
                ) and (
                    quantile is not None
                    and property_name
                    and filter_type
                ):
                    new_memory = _set_threshold_memory(memory, threshold_id, property_name, filter_type, slider_vals, quantile=quantile)
                    if new_memory != memory:
                        memory = new_memory
                        changed = True
                if changed:
                    thresholding_memory.set(memory)
                    return
                
            elif filter_type == "Relative to...":
                if input[f"threshold_property_{threshold_id}"]() in Metrics.Thresholding.SpotProperties:
                    data = UNFILTERED_SPOTSTATS.get()
                else :
                    data = UNFILTERED_TRACKSTATS.get()
                
                reference = input[f"reference_value_{threshold_id}"]()
                if reference == "Mean":
                    reference = data[property_name].mean()
                elif reference == "Median":
                    reference = data[property_name].median()
                elif reference == "My own value":
                    reference = input[f"my_own_value_{threshold_id}"]()

                if reference is None or not isinstance(reference, (int, float)):
                    reference = 0

                if (
                    isinstance(slider_vals, (tuple, list))
                    and len(slider_vals) == 2
                    and all(x is not None for x in slider_vals)
                    and isinstance(reference, (int, float))
                ) and (
                    reference is not None
                    and property_name
                    and filter_type
                ):
                    new_memory = _set_threshold_memory(memory, threshold_id, property_name, filter_type, slider_vals, reference=reference)
                    if new_memory != memory:
                        memory = new_memory
                        changed = True
                if changed:
                    thresholding_memory.set(memory)
                    return
            
            
                
            # if filter_type == "Quantile":

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
                threshold_slider_outputs[threshold_id] = make_threshold_slider(threshold_id) 

    @reactive.Effect
    def set_threshold_modules():
        _property_selections = {}
        _filter_type_selections = {}

        for threshold_id in threshold_list.get():
            property_name = input[f"threshold_property_{threshold_id}"]()
            filter_type = input[f"threshold_filter_{threshold_id}"]()

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

        _property_selections = {tid: val for tid, val in _property_selections.items() if tid in threshold_list.get()}
        property_selections.set(_property_selections)

        _filter_type_selections = {tid: val for tid, val in _filter_type_selections.items() if tid in threshold_list.get()}
        filter_type_selections.set(_filter_type_selections)

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

    

    # - - - - Thresholding histograms - - - -

    # def _thresholded_histogram(property, filter_type, slider_range, dfA, dfB, bin_count):
        
    #     if metric in Track_metrics.get():
    #         data = dfA.get()
    #     elif metric in Spot_metrics.get():
    #         data = dfB.get()
    #     elif data.empty or data is None:
    #         return
    #     else:
    #         return

    #         if bin_count is None:
    #             bin_count = 40

    #         values = data[metric].dropna()

    #         if filter_type == "percentile":
    #             lower_bound = np.percentile(values, slider_range[0])
    #             upper_bound = np.percentile(values, slider_range[1])
    #         else:
    #             lower_bound = slider_range[0]
    #             upper_bound = slider_range[1]

    #         fig, ax = plt.subplots()
    #         n, bins, patches = ax.hist(values, bins=bin_count, density=False)

    #         # Color threshold
    #         for i in range(len(patches)):
    #             if bins[i] < lower_bound or bins[i+1] > upper_bound:
    #                 patches[i].set_facecolor('grey')
    #             else:
    #                 patches[i].set_facecolor('#337ab7')

    #         # Add KDE curve (scaled to match histogram)
    #         kde = gaussian_kde(values)
    #         x_kde = np.linspace(bins[0], bins[-1], 500)
    #         y_kde = kde(x_kde)
    #         # Scale KDE to histogram
    #         y_kde_scaled = y_kde * (n.max() / y_kde.max())
    #         ax.plot(x_kde, y_kde_scaled, color='black', linewidth=1)

    #         ax.set_xticks([])  # Remove x-axis ticks
    #         ax.set_yticks([])  # Remove y-axis ticks
    #         ax.spines[['top', 'left', 'right']].set_visible(False)

    #         return fig

    # - - - - - - - - - - - - - - - - - - - -





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
