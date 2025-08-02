# app_core.py

# Imports: UI, reactivity, utility functions, and styling
from shiny import App, Inputs, Outputs, Session, render, reactive, req, ui
from shinywidgets import render_plotly, render_altair

import utils.data_calcs as dc
import utils.funcs_plot as pu
from utils.Select import Metrics, Colors, Markers, Modes
import utils.select_modes as select_mode
import utils.select_metrics as select_metrics
from utils.ratelimit import debounce, throttle

from utils.Customize import Format

# -----------------------------------------------------------------------------
# UI DEFINITION
# -----------------------------------------------------------------------------

app_ui = ui.page_sidebar(
    # Sidebar for data filtering options and threshold management
    ui.sidebar(
        ui.tags.style(Format.Accordion),
        ui.markdown("""  <p>  """),
        ui.output_ui("sidebar_label"),
        ui.input_action_button("add_threshold", "Add threshold", class_="btn-primary"),
        ui.input_action_button("remove_threshold", "Remove threshold", class_="btn-primary", disabled=True),
        ui.output_ui("sidebar_accordion"),
        id="sidebar", open="open", position="right", bg="f8f8f8",
    ),

    # Main navigation bar containing different analysis and visualization panels
    ui.navset_bar(
        # Panel: Input of raw data, initial setup
        ui.nav_panel(
            "Input",
            ui.div(
                {"id": "data-inputs"},
                # Action buttons for managing input datasets
                ui.input_action_button("add_input", "Add data input", class_="btn-primary"),
                ui.input_action_button("remove_input", "Remove data input", class_="btn-primary", disabled=True),
                ui.input_action_button("run", "Run", class_="btn-secondary", disabled=True),
                ui.input_action_button("reset", "Reset", class_="btn-danger"),
                ui.input_action_button("input_help", "Show help"),
                ui.markdown("""___"""),
                # File inputs
                ui.row(
                    ui.column(4, ui.output_ui("input_file_pairs")),
                ),
                # Drag-and-drop window for mapping columns
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

        # Panel: Display and download processed data frames
        ui.nav_panel(
            "Data frames",
            ui.markdown(""" <p> """),

            # Input for loading pre-processed data
            ui.input_file("already_proccesed_input", "Got previously processed data?", placeholder="Drag and drop here!", accept=[".csv"], multiple=False),
            ui.markdown(""" ___ """),

            # Display: stats for spot, track, and time
            ui.layout_columns(
                ui.card(
                    ui.card_header("Spot stats"),
                    ui.output_data_frame("render_spot_stats"),
                    ui.download_button("download_spot_stats", "Download CSV"),
                ),
                ui.card(
                    ui.card_header("Track stats"),
                    ui.output_data_frame("render_track_stats"),
                    ui.download_button("download_track_stats", "Download CSV"),
                ),
                ui.card(
                    ui.card_header("Time stats"),
                    ui.output_data_frame("render_time_stats"),
                    ui.download_button("download_time_stats", "Download CSV"),
                ),
            ),
        ),

        # Panel: Main visualization area, organized by sub-panels
        ui.nav_panel(
            "Visualisation",

            # Sub-panels for tracks, time charts, superplots
            ui.navset_pill_list(
                ui.nav_panel(
                    "Tracks",
                    # Interactive controls for track visualization
                    ui.panel_well(
                        ui.markdown(
                            """
                            #### **Track visualization**
                            *made with*  `plotly`
                            <hr style="height: 4px; background-color: black; border: none" />
                            """
                        ),
                        ui.accordion(
                            # Dataset selection
                            ui.accordion_panel(
                                "Dataset",
                                ui.input_selectize("condition_tracks", "Condition:", ["all", "not all"]),
                                ui.panel_conditional(
                                    "input.condition_tracks != 'all'",
                                    ui.input_selectize("replicate_tracks", "Replicate:", []),
                                ),
                            ),
                            # Track line settings
                            ui.accordion_panel(
                                "Track lines",
                                ui.input_checkbox("show_tracks", "Show tracks", True),
                                ui.panel_conditional(
                                    "input.show_tracks",
                                    ui.input_numeric("smoothing", "Smoothing index:", 0),
                                    ui.input_numeric('track_line_width', 'Line width:', 0.85),
                                ),
                            ),
                            # Marker display settings
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
                            # Color and background options
                            ui.accordion_panel(
                                "Coloring",
                                ui.input_selectize("color_mode", "Color mode:", Colors.ColorMode),
                                ui.panel_conditional(
                                    "input.color_mode != 'random greys' && input.color_mode != 'random colors' && input.color_mode != 'only-one-color' && input.color_mode != 'differentiate conditions/replicates'",
                                    ui.input_selectize('lut_scaling', 'LUT scaling metric:', []),
                                ),
                                ui.panel_conditional(
                                    "input.color_mode == 'only-one-color'",
                                    ui.input_selectize('only_one_color', 'Color:', Colors.Color),
                                ),
                                ui.input_selectize('background', 'Background:', Colors.Background),
                                ui.input_checkbox("show_gridlines", "Gridlines", True),
                            ),
                            # Tooltip info selection
                            ui.accordion_panel(
                                "Hover info",
                                ui.input_selectize("let_me_look_at_these", "Let me look at these:", ["Condition", "Track length", "Net distance", "Speed mean"], multiple=True),
                                ui.input_action_button("hover_info", "See info"),
                            ),
                        ),
                    ),
                    ui.markdown(""" <p> """),
                    # Track plots and download options
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
                        ui.input_select("time_plot", "Plot:", choices=["Scatter", "Line", "Errorband"]),
                        ui.accordion(
                            # Dataset selection
                            ui.accordion_panel(
                                "Dataset",
                                ui.input_selectize("condition_time", "Condition:", ["all", "not all"]),
                                ui.panel_conditional(
                                    "input.condition_time != 'all'",
                                    ui.input_selectize("replicate_time", "Replicate:", ["all", "not all"]),
                                    ui.panel_conditional(
                                        "input.replicate_time == 'all'",
                                        ui.input_checkbox("time_separate_replicates", "Show replicates separately", False),
                                    ),
                                ),
                            ),
                            # Metric selection
                            ui.accordion_panel(
                                "Metric",
                                ui.input_selectize("time_metric", label=None, choices=Metrics.Time, selected='Mean confinement ratio'),
                                ui.input_radio_buttons("y_axis", "Y axis with", ["Absolute values", "Relative values"]),
                            ),
                            # Plot settings for different plot types
                            ui.accordion_panel(
                                "Plot settings",
                                ui.panel_conditional(
                                    "input.time_plot == 'Scatter'",
                                    ui.accordion(
                                        ui.accordion_panel(
                                            "Central tendency",
                                            ui.input_checkbox_group("central_tendency_scatter", label=None, choices=["mean", "median"], selected=["median"]),
                                        ),
                                        ui.accordion_panel(
                                            "Polynomial fit",
                                            ui.row(
                                                ui.column(3, ui.input_checkbox("tch_polynomial_fit", "Fit", True)),
                                                ui.input_switch("fit_best", "Automatic fit", True),
                                            ),
                                            ui.panel_conditional(
                                                "input.tch_polynomial_fit == true",
                                                ui.panel_conditional(
                                                    "input.fit_best == false",
                                                    ui.input_selectize("force_fit", "Fit:", list(Modes.FitModel.keys())),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                ui.panel_conditional(
                                    "input.time_plot == 'Line'",
                                    ui.accordion(
                                        ui.accordion_panel(
                                            "Central tendency",
                                            ui.input_checkbox_group("central_tendency_scatter", label=None, choices=["mean", "median"], selected=["median"]),
                                        ),
                                        ui.accordion_panel(
                                            "Line",
                                            ui.input_selectize("tch_line_interpolation", "Interpolation:", Modes.Interpolate),
                                        ),
                                    ),
                                ),
                                ui.panel_conditional(
                                    "input.time_plot == 'Errorband'",
                                    # (Empty for now)
                                ),
                            ),
                            # Aesthetics (color palette, bullets, line width, etc.)
                            ui.accordion_panel(
                                "Aesthetics",
                                ui.panel_conditional(
                                    "input.time_plot == 'Scatter'",
                                    ui.accordion(
                                        ui.accordion_panel(
                                            "Color palette",
                                            ui.input_selectize("tch_background", "Background:", Colors.Background),
                                            ui.input_selectize("tch_color_palette", "Color palette:", Colors.PaletteQualitative),
                                        ),
                                        ui.accordion_panel(
                                            "Bullets",
                                            ui.input_checkbox("tch_show_scatter", "Show scatter", True),
                                            ui.panel_conditional(
                                                "input.tch_show_scatter == true",
                                                ui.input_numeric("tch_bullet_opacity", "Opacity:", 0.5, min=0, max=1, step=0.01),
                                                ui.input_checkbox("tch_fill_bullets", "Fill bullets", True),
                                                ui.input_numeric("tch_bullet_size", "Bullet size:", 5, min=0, step=0.1),
                                                ui.panel_conditional(
                                                    "input.tch_fill_bullets == true",
                                                    ui.input_checkbox("tch_outline_bullets", "Outline bullets", False),
                                                    ui.panel_conditional(
                                                        "input.tch_outline_bullets == true",
                                                        ui.input_selectize("tch_bullet_outline_color", "Outline color:", Colors.Color, selected="black"),
                                                        ui.input_numeric("tch_bullet_outline_width", "Outline width:", 1, min=0, step=0.1),
                                                    ),
                                                ),
                                                ui.panel_conditional(
                                                    "input.tch_fill_bullets == false",
                                                    ui.input_numeric("tch_bullet_stroke_width", "Stroke width:", 1, min=0, step=0.1),
                                                ),
                                            ),
                                        ),
                                        ui.accordion_panel(
                                            "Line",
                                            ui.panel_conditional(
                                                "input.tch_polynomial_fit == true",
                                                ui.input_numeric("tch_line_width", "Line width:", 1, min=0, step=0.1),
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
                    # Additional cards can be added for line chart and errorband
                ),
                ui.nav_panel(
                    "Superplots",
                    ui.panel_well(
                        ui.input_selectize("testing_metric", "Test for metric:", []),
                        ui.input_selectize('palette', 'Color palette:', []),
                        # etc.
                    ),
                    ui.card(
                        ui.output_plot("seaborn_superplot"),
                        ui.download_button("download_seaborn_superplot_svg", "Download Seaborn Superplot SVG"),
                    ),
                ),
            ),
        ),
        # Task list and controls
        ui.nav_panel("Task list"),
        ui.nav_spacer(),
        ui.nav_control(ui.input_dark_mode(mode="light")),
        title="Peregrin"
    ),
)

# -----------------------------------------------------------------------------
# SERVER LOGIC
# -----------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):
    """
    Main server logic for the Peregrin UI application.

    Handles:
    - Dynamic UI components for threshold filtering (1D/2D) and file inputs.
    - State management for input/threshold lists.
    - Rendering of sidebar accordion and file input UI blocks.
    - Reactive updates on user interaction.
    """

    # -- Threshold Filtering State --

    threshold_dimension = reactive.Value("1D")            # Can be "1D" or "2D"
    dimension_button_label = reactive.Value("2D")         # Button label toggles to switch mode
    threshold_list = reactive.Value([0])                  # List of threshold IDs (for dynamic panels)

    @reactive.effect
    @reactive.event(input.add_threshold)
    def add_threshold():
        """Add another threshold panel to the sidebar (increments threshold_list)."""
        ids = threshold_list.get()
        threshold_list.set(ids + [max(ids)+1 if ids else 0])
        session.send_input_message("remove_threshold", {"disabled": False})

    @reactive.effect
    @reactive.event(input.remove_threshold)
    def remove_threshold():
        """Remove the last threshold panel, disable button if only one remains."""
        ids = threshold_list.get()
        if len(ids) > 1:
            threshold_list.set(ids[:-1])
        if len(threshold_list.get()) <= 1:
            session.send_input_message("remove_threshold", {"disabled": True})

    @output()
    @render.ui
    def sidebar_accordion():
        """
        Render the sidebar accordion with dynamic threshold panels,
        adapting layout for 1D or 2D mode.
        """
        ids = threshold_list.get()
        panels = []

        if threshold_dimension.get() == "1D":
            # 1D thresholding: one property, filter, slider per panel
            for i, threshold_id in enumerate(ids, 1):
                panels.append(
                    ui.accordion_panel(
                        f"Threshold {i}" if len(ids) >= 2 else "Threshold",
                        ui.panel_well(
                            ui.input_selectize(f"threshold_property_{threshold_id}", "Property", Metrics.SpotAndTrack),
                            ui.input_selectize(f"threshold_filter_{threshold_id}", "Filter values", Modes.Thresholding),
                            ui.panel_conditional(
                                f"input.threshold_filter_{threshold_id} == 'Relative to...'",
                                ui.input_selectize(f"reference_value_{threshold_id}", "Reference value", choices=["Mean", "Median", "Min", "Max", "My own value"]),
                                ui.panel_conditional(
                                    f"input.reference_value_{threshold_id} == 'My own value'",
                                    ui.input_numeric(f"my_own_value_{threshold_id}", "My own value", value=0, step=1)
                                ),
                            ),
                            ui.input_slider(f"threshold_values_{threshold_id}", "Threshold", min=0, max=100, value=(0, 100)),
                        ),
                    ),
                )
            # Settings panel for thresholds
            panels.append(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                    ui.markdown(""" <p> """),
                    ui.input_numeric("bins", "Number of bins", value=40, min=1, step=1),
                    ui.input_radio_buttons("plot_distribution", "Histogram show:", choices=["Kernel density", "Hover info"], selected="Kernel density"),
                ),
            )
        elif threshold_dimension.get() == "2D":
            # 2D thresholding: two properties per panel (X and Y)
            for i, threshold_id in enumerate(ids, 1):
                panels.append(
                    ui.accordion_panel(
                        f"Threshold {i}" if len(ids) >= 2 else "Threshold",
                        ui.panel_well(
                            ui.markdown(""" <h6>  Properties X;Y  </h6>"""),
                            ui.input_selectize(f"thresholding_metric_X_{threshold_id}", None, Metrics.SpotAndTrack),
                            ui.input_selectize(f"thresholding_metric_Y_{threshold_id}", None, Metrics.SpotAndTrack),
                            ui.input_selectize(f"thresholding_filter_2D_{threshold_id}", "Thresholding values", ["Literal", "Normalized 0-1"]),
                        ),
                    ),
                )
            panels.append(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                    ui.markdown(""" <p> """),
                    ui.markdown("""  Working on it dawg  """),   # TODO: Expand this for 2D mode
                ),
            )
        # Keep currently active panels open by default
        return ui.accordion(*panels, id="thresholds_accordion", open=["Threshold", f"Threshold {len(ids)}", "Filter settings"])

    @reactive.Effect
    @reactive.event(input.threshold_dimensional_toggle)
    def threshold_dimensional_toggle():
        """Toggle between 1D and 2D thresholding modes (updates button label and UI state)."""
        if threshold_dimension.get() == "1D":
            threshold_dimension.set("2D")
            dimension_button_label.set("1D")
        else:
            threshold_dimension.set("1D")
            dimension_button_label.set("2D")

    @output()
    @render.text
    def sidebar_label():
        """Display the current thresholding mode as sidebar label."""
        return ui.markdown(
            f""" <h5> <b>  {threshold_dimension.get()} Data filtering  </b> </h5> """
        )

    # -- File Input Panel State --

    input_list = reactive.Value([1])  # Always start with one input file/label pair

    @reactive.effect
    @reactive.event(input.add_input)
    def add_input():
        """Add another file/label input group."""
        ids = input_list.get()
        new_id = max(ids) + 1 if ids else 1
        input_list.set(ids + [new_id])
        session.send_input_message("remove_input", {"disabled": len(ids) < 1})

    @reactive.effect
    @reactive.event(input.remove_input)
    def remove_input():
        """Remove the last file/label input group, disable remove if only one left."""
        ids = input_list.get()
        if len(ids) > 1:
            input_list.set(ids[:-1])
        if len(input_list.get()) <= 1:
            session.send_input_message("remove_input", {"disabled": True})

    @output()
    @render.ui
    def input_file_pairs():
        """
        Render a list of input file + label UI blocks, one per dataset.
        Each group contains a text input (label) and a file input.
        """
        ids = input_list.get()
        ui_blocks = []
        for idx in ids:
            ui_blocks.append([
                ui.input_text(f"condition_label{idx}", f"Condition {idx}" if len(ids) > 1 else "Condition", placeholder=f"Label me!"),
                ui.input_file(f"input_file{idx}", "Upload files:", placeholder="Drag and drop here!", multiple=True),
                ui.markdown(""" <hr style="border: none; border-top: 1px dotted" /> """),
            ])
        return ui_blocks

    # (Other outputs and logic may be defined elsewhere...)

# -----------------------------------------------------------------------------
# APP MOUNTING
# -----------------------------------------------------------------------------

app = App(app_ui, server)
