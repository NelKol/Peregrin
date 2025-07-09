from shiny import App, ui, render, reactive
from shinywidgets import render_plotly, render_altair

import utils.data_calcs as dc
import utils.funcs_plot as pu
import utils.select_markers as select_markers
import utils.select_modes as select_mode
import utils.select_metrics as select_metrics
from utils.ratelimit import debounce, throttle

# --- UI Layout ---

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.markdown("""  <p>  """),
        ui.markdown(""" <h5> <b>  Data filtering  </b> </h5> """),
        ui.accordion(
            ui.accordion_panel(
                "Threshold 1",
                ui.panel_well(
                    ui.panel_conditional(
                        "input.dimensional_threshold == false",
                        ui.input_select("thresholding_properties", "Thresholding property", select_metrics.spots_n_tracks),
                        ui.input_select("thresholding_filter1D", "Thresholding values", ["literal", "percentile"]),
                        ui.input_slider("thresholding_values", label=None, min=0, max=100, step=1, value=(0, 100)),
                        ui.panel_conditional(
                            "input.thresholding_filter1D == 'literal'",
                            # TODO: implement histogram for literal values
                            # ui.output_plot("thresholding_histogram", height="300px"),
                            None
                        ),
                    ),
                    ui.panel_conditional(
                        "input.dimensional_threshold == true",
                        ui.markdown(""" <h6>  Properties X;Y  </h6>"""),
                        ui.input_select("thresholding_metric_X", None, select_metrics.spots_n_tracks),
                        ui.input_select("thresholding_metric_Y", None, select_metrics.spots_n_tracks),
                        ui.input_select("thresholding_filter_2D", "Thresholding values", ["literal", "percentile"]),
                    ),
                ),
                ui.input_task_button("apply_thresholding", "Set threshold"),
            ),
            ui.accordion_panel(
                "Filter settings",
                ui.input_switch("dimensional_threshold", "2D", value=False),
                ui.panel_conditional(
                    "input.dimensional_threshold == false",
                    ui.input_numeric("bins", "Number of bins", value=40, min=1, step=1),
                    ui.input_radio_buttons("plot_distribution", "Histogram show:", choices=["Kernel density", "Hover info"], selected="Kernel density"),
                ),
                ui.panel_conditional(
                    "input.dimensional_threshold == true",
                    "working on it dawg"
                ),
            ),
        ),
        open="open", position="right", bg="f8f8f8"
    ),
    ui.navset_bar(
        ui.nav_panel(
            "Input",
            ui.div(
                {"id": "data-inputs"},
                # Action buttons
                ui.input_action_button("add_input", "Add data input", class_="btn-primary"),
                ui.input_action_button("remove_input", "Remove data input", class_="btn-primary"),
                ui.input_action_button("run", "Run", class_="btn-secondary", disabled=True),
                ui.input_action_button("reset", "Reset", class_="btn-danger"),
                ui.input_action_button("input_help", "Show help"),
                # Line break for better spacing
                ui.markdown("""___"""),
                # Default data input
                ui.input_text("label1", "Condition no. 1", placeholder="label me :D"),
                ui.input_file("file1", "Input files:", multiple=True),
                # ... You can add more file inputs/labels dynamically
                ui.panel_absolute(
                    ui.panel_well(
                        ui.markdown("<h5>Select columns:</h5>"),
                        ui.input_select("select_id", "Track identifier:", ["e.g. TRACK_ID"]),
                        ui.input_select("select_time", "Time point:", ["e.g. POSITION_T"]),
                        ui.input_select("select_x", "X coordinate:", ["e.g. POSITION_X"]),
                        ui.input_select("select_y", "Y coordinate:", ["e.g. POSITION_Y"]),
                        ui.markdown("<span style='color:darkgrey; font-style:italic;'>You can drag me around!</span>")
                    ),
                    width="350px", right="300px", top="220px", draggable=True
                ),
            ),
        ),
        ui.nav_panel(
            "Data frames",
            ui.input_file("already_proccesed_spot_stats", "", accept=[".csv"], multiple=False),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Spot stats"),
                    ui.output_data_frame("render_spot_stats"),
                    ui.download_button("download_spot_stats", "Download Spot Stats CSV"),
                ),
                ui.card(
                    ui.card_header("Track stats"),
                    ui.output_data_frame("render_track_stats"),
                    ui.download_button("download_track_stats", "Download Track Stats CSV"),
                ),
                ui.card(
                    ui.card_header("Frame stats"),
                    ui.output_data_frame("render_time_stats"),
                    ui.download_button("download_time_stats", "Download Frame Stats CSV"),
                ),
            )
        ),
        ui.nav_panel(
            "Visualisation",
            ui.navset_pill_list(
                ui.nav_panel(
                    "Tracks",
                    # Interactive settings
                    ui.panel_well(
                        ui.input_selectize("let_me_look_at_these", "Let me look at these:", [], multiple=True),
                        ui.input_action_button('hover_info', 'see info'),
                        ui.input_numeric('marker_size', 'Marker size:', 5),
                        ui.input_checkbox('end_track_markers', 'markers at the end of the tracks', True),
                        ui.input_select('markers', 'Markers:', []),
                        ui.input_checkbox('I_just_wanna_be_normal', 'just be normal', True),
                    ),
                    # Plotly outputs
                    ui.card(
                        ui.output_plot("interactive_true_track_visualization"),
                        ui.download_button("download_true_interactive_visualization_html", "Download True Interactive Visualization HTML"),
                        ui.output_plot("interactive_normalized_track_visualization"),
                        ui.download_button("download_normalized_interactive_visualization_html", "Download Normalized Interactive Visualization HTML"),
                    ),
                    # Static matplotlib settings
                    ui.panel_well(
                        ui.input_numeric('arrow_size', 'Arrow size:', 6),
                        ui.input_checkbox('arrows', 'arrows at the end of tracks', True),
                        ui.input_checkbox("grid", "grid", True),
                    ),
                    # Static matplotlib outputs
                    ui.card(
                        ui.output_plot("true_track_visualization"),
                        ui.download_button("download_true_visualization_svg", "Download True Visualization SVG"),
                        ui.output_plot("normalized_track_visualization"),
                        ui.download_button("download_normalized_visualization_svg", "Download Normalized Visualization SVG"),
                        ui.download_button("download_lut_map_svg", "Download LUT Map SVG"),
                    ),
                    # Common settings
                    ui.panel_well(
                        ui.input_select("condition", "Condition:", []),
                        ui.input_select("replicate", "Replicate:", []),
                        ui.input_select("color_mode", "Color mode:", []),
                        ui.input_select('lut_scaling', 'LUT scaling metric:', []),
                        ui.input_select('only_one_color', 'Color:', []),
                        ui.input_select('background', 'Background:', []),
                        ui.input_numeric("smoothing", "Smoothing:", 0),
                        ui.input_numeric('track_line_width', 'Line width:', 0.85),
                        ui.input_checkbox('show_tracks', 'show tracks', True),
                    ),
                ),
                ui.nav_panel(
                    "Time series",
                    # ... Add time series UI here
                    ui.panel_well(
                        ui.input_radio_buttons("central_tendency", "Measure of central tendency:", ["mean", "median"]),
                        # ... rest of your time series controls
                    ),
                    ui.card(
                        ui.output_plot("time_series_poly_fit_chart"),
                        ui.download_button("download_time_series_poly_fit_chart__html", "Download Time Series Poly Fit Chart HTML"),
                        ui.download_button("download_time_series_poly_fit_chart_svg", "Download Time Series Poly Fit Chart SVG"),
                    ),
                    # ... more cards for line chart and errorband
                ),
                ui.nav_panel(
                    "Superplots",
                    # ... Add superplots UI here
                    ui.panel_well(
                        ui.input_select("testing_metric", "Test for metric:", []),
                        ui.input_select('palette', 'Color palette:', []),
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

def server(input, output, session):

    # Example: Render a text output
    # @output()
    # @render.text
    # def my_text():
    #     return "Hello, Peregrin!"

    # Add your other outputs below...

    @output()
    @render.data_frame
    def render_spot_stats():
        # TODO: return DataFrame to display
        pass

    @output()
    @render.download
    def download_spot_stats():
        # TODO: yield data for download
        pass

    @output()
    @render_plotly
    def interactive_true_track_visualization():
        # TODO: return plotly figure
        pass

    # ...and so on for each output in your app

# --- Mount the app ---
app = App(app_ui, server)
