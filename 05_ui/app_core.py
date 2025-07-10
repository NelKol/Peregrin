from shiny import App, Inputs, Outputs, Session, render, reactive, req, ui
from shinywidgets import render_plotly, render_altair

import utils.data_calcs as dc
import utils.funcs_plot as pu
import utils.select_markers as select_markers
import utils.select_modes as select_mode
import utils.select_metrics as select_metrics
from utils.ratelimit import debounce, throttle

from custom.formatting import Accordion


type_time_chart = reactive.Value("Scatter"),



# --- UI Layout ---

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.style(Accordion),
        ui.markdown("""  <p>  """),
        ui.output_ui("sidebar_label"),
        ui.input_action_button("add_threshold", "Add threshold", class_="btn-primary"),
        ui.input_action_button("remove_threshold", "Remove threshold", class_="btn-primary", disabled=True),
        ui.output_ui("sidebar_accordion"),
        id="sidebar", open="open", position="right", bg="f8f8f8",
    ),
    ui.navset_bar(
        ui.nav_panel(
            "Input",
            ui.div(
                {"id": "data-inputs"},
                # Action buttons
                ui.input_action_button("add_input", "Add data input", class_="btn-primary"),
                ui.input_action_button("remove_input", "Remove data input", class_="btn-primary", disabled=True),
                ui.input_action_button("run", "Run", class_="btn-secondary", disabled=True),
                ui.input_action_button("reset", "Reset", class_="btn-danger"),
                ui.input_action_button("input_help", "Show help"),
                # Line break for better spacing
                ui.markdown("""___"""),
                # Default data input
                ui.input_text("condition_label1", "Condition", placeholder="Label me!"),
                ui.input_file("input_file1", "Upload files:", placeholder="Drag and drop here!", multiple=True),
                # ... You can add more file inputs/labels dynamically
                ui.panel_absolute(
                    ui.panel_well(
                        ui.markdown("<h5>Select columns:</h5>"),
                        ui.input_selectize("select_id", "Track identifier:", ["e.g. TRACK_ID"]),
                        ui.input_selectize("select_time", "Time point:", ["e.g. POSITION_T"]),
                        ui.input_selectize("select_x", "X coordinate:", ["e.g. POSITION_X"]),
                        ui.input_selectize("select_y", "Y coordinate:", ["e.g. POSITION_Y"]),
                        ui.markdown("<span style='color:darkgrey; font-style:italic;'>You can drag me around!</span>")
                    ),
                    width="350px", right="300px", top="220px", draggable=True
                ),
            ),
        ),
        ui.nav_panel(
            "Data frames",
            ui.markdown(""" <p> """),
            ui.input_file("already_proccesed_input", "Got previously processed data?", placeholder="Drag and drop here!", accept=[".csv"], multiple=False),
            ui.markdown(""" ___ """),
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
                    ui.card_header("Frame stats"),
                    ui.output_data_frame("render_time_stats"),
                    ui.download_button("download_time_stats", "Download CSV"),
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
                                    ui.input_selectize("markers", "Markers:", []),
                                    ui.input_numeric("marker_size", "Marker size:", 5),
                                    ui.input_switch("just_be_normal", "Just normal", True),
                                ),
                            ),
                            ui.accordion_panel(
                                "Coloring",
                                ui.input_selectize("color_mode", "Color mode:", ["only-one-color", "not only-one-color"]),
                                ui.panel_conditional(
                                    "input.color_mode != 'only-one-color'",
                                    ui.input_selectize('lut_scaling', 'LUT scaling metric:', []),
                                ),
                                ui.panel_conditional(
                                    "input.color_mode == 'only-one-color'",
                                    ui.input_selectize('only_one_color', 'Color:', []),
                                ),
                                ui.input_selectize('background', 'Background:', []),
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
                        ui.input_select("time_plot", "Plot:", choices=["Scatter", "Line", "Errorband"]),
                        # type_time_chart.set(input.time_plot),
                        ui.accordion(
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
                            ui.accordion_panel(
                                "Plot metric",
                                ui.input_selectize("time_metric", "Metric:", select_metrics.time, selected='Mean confinement ratio'),
                                ui.input_radio_buttons("y_axis", "On Y axis with", ["absolute values", "relative values"], selected="absolute"),
                            ),
                            ui.accordion_panel(
                                # type_time_chart.get(),
                                "Plot settings",
                                ui.panel_conditional(
                                    "input.time_plot == 'Scatter'",
                                    ui.input_checkbox_group("central_tendency_scatter", "Central tendency", ["mean", "median"], selected=["median"]),
                                    ui.input_checkbox("time_polynomial_fit", "Polynomial fit", True),
                                    ui.panel_conditional(
                                        "input.time_polynomial_fit == true",
                                        ui.input_switch("fit_best", "Automatic fit", True),
                                    ),
                                ),
                                ui.panel_conditional(
                                    "input.time_plot == 'Line'",
                                    ui.input_checkbox_group("central_tendency_line", "Central tendency", ["mean", "median"], selected=["median"]),
                                ),
                                ui.panel_conditional(
                                    "input.time_plot == 'Errorband'",
                                    
                                ),
                            ),
                        ),
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
    # Use a reactive value to store the label state
    threshold_dimension = reactive.Value("1D")
    dimension_button_label = reactive.Value("2D")  # Reactive value for the label
    threshold_list = reactive.Value([0])  # Start with one threshold (ID = 0)


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
        if len(ids) <= 2:
            session.send_input_message("remove_threshold", {"disabled": True})

    
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
                            ui.input_selectize(f"threshold_property_{threshold_id}", "Property", select_metrics.spots_n_tracks),
                            ui.input_selectize(f"threshold_filter_{threshold_id}", "Filter values", ["literal", "percentile"]),
                            ui.input_slider(f"threshold_values_{threshold_id}", "Threshold", min=0, max=100, value=(0, 100)),
                        )
                    )
                )
            # Static panel at the end
            panels.append(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                    ui.markdown(""" <p> """),
                    ui.input_numeric("bins", "Number of bins", value=40, min=1, step=1),
                    ui.input_radio_buttons("plot_distribution", "Histogram show:", choices=["Kernel density", "Hover info"], selected="Kernel density"),
                )
            )

        elif threshold_dimension.get() == "2D":
            for i, threshold_id in enumerate(ids, 1):
                panels.append(
                    ui.accordion_panel(
                        f"Threshold {i}" if len(ids) >= 2 else "Threshold",
                        ui.panel_well(
                            ui.markdown(""" <h6>  Properties X;Y  </h6>"""),
                            ui.input_selectize("thresholding_metric_X", None, select_metrics.spots_n_tracks),
                            ui.input_selectize("thresholding_metric_Y", None, select_metrics.spots_n_tracks),
                            ui.input_selectize("thresholding_filter_2D", "Thresholding values", ["literal", "percentile"]),
                        )
                    )
                )
            # Static panel at the end
            panels.append(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_action_button("threshold_dimensional_toggle", dimension_button_label.get(), width="100%"),
                    ui.markdown(""" <p> """),
                    ui.markdown("""  Working on it dawg  """),
                )
            )
        
        # Set all panels open by default (can be a list of panel titles)
        return ui.accordion(*panels, id="thresholds_accordion", open=["Threshold", f"Threshold {len(ids)}", "Filter settings"])

    @render.text
    def sidebar_label():
        return ui.markdown(
            f""" <h5> <b>  {threshold_dimension.get()} Data filtering  </b> </h5> """
        )
    
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
    def threshold_values_display():
        # Example: Display all threshold values
        ids = threshold_list.get()
        vals = []
        for threshold_id in ids:
            prop = input.get(f"threshold_property_{threshold_id}") or ""
            ftr = input.get(f"threshold_filter_{threshold_id}") or ""
            rng = input.get(f"threshold_values_{threshold_id}") or (None, None)
            vals.append(f"Panel {threshold_id}: {prop}, {ftr}, {rng}")
        return " | ".join(vals)


    
    
    # @output()
    # @render.ui
    # def data_filtering():
    #     if threshold_dimension.get() == "1D":
    #         return [
    #             ui.input_selectize("thresholding_properties", "Thresholding property", select_metrics.spots_n_tracks),
    #             ui.input_selectize("thresholding_filter1D", "Thresholding values", ["literal", "percentile"]),
    #             ui.input_slider("thresholding_values", label=None, min=0, max=100, step=1, value=(0, 100)),
    #             ui.panel_conditional(
    #                 "input.thresholding_filter1D == 'literal'",
    #                 # TODO: implement histogram for literal values
    #             # ui.output_plot("thresholding_histogram", height="300px"),
    #             None
    #             ),
    #         ]
    #     elif threshold_dimension.get() == "2D":
    #         return [
    #             ui.markdown(""" <h6>  Properties X;Y  </h6>"""),
    #             ui.input_selectize("thresholding_metric_X", None, select_metrics.spots_n_tracks),
    #             ui.input_selectize("thresholding_metric_Y", None, select_metrics.spots_n_tracks),
    #             ui.input_selectize("thresholding_filter_2D", "Thresholding values", ["literal", "percentile"]),
    #         ]
    #     else:
    #         return None

    # @output()
    # @render.ui
    # def filtering_settings():
    #     if threshold_dimension.get() == "1D":
    #         return [
    #             ui.input_numeric("bins", "Number of bins", value=40, min=1, step=1),
    #             ui.input_radio_buttons("plot_distribution", "Histogram show:", choices=["Kernel density", "Hover info"], selected="Kernel density"),
    #         ]
    #     elif threshold_dimension.get() == "2D":
    #         return ui.markdown("Working on it dawg")
    #     else:
    #         return None


# --- Mount the app ---
app = App(app_ui, server)




