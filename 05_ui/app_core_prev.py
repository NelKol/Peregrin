from shiny import App, Inputs, Outputs, Session, render, reactive, req, ui
from shiny.types import FileInfo
from shinywidgets import render_plotly, render_altair, output_widget, render_widget

from utils.Select import Metrics, Styles, Markers, Modes
from utils.Function import DataLoader, Process, Calc, Threshold, Plot
from utils.RateLimit import Debounce, Throttle, MakeThrottled
from utils.Customize import Format

import asyncio
import io
import warnings
import tempfile

import pandas as pd
import numpy as np
from html import escape
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from math import floor, ceil
from scipy.stats import gaussian_kde
from datetime import date

import warnings
from shiny._deprecated import ShinyDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r".*panel_well\(\) is deprecated\. Use shiny\.ui\.card\(\) instead\.",
    category=ShinyDeprecationWarning,
)



# - - - -  UI design definition  - - - - 
app_ui = ui.page_sidebar(

    # ========== SIDEBAR - DATA FILTERING ==========
    ui.sidebar(
        ui.tags.style(Format.Accordion),
        ui.markdown("""  <p>  """),
        ui.output_ui(id="sidebar_label"),
        ui.input_action_button(id="add_threshold", label="Add threshold", class_="btn-primary"),
        ui.input_action_button(id="remove_threshold", label="Remove threshold", class_="btn-primary", disabled=True),
        ui.output_ui(id="sidebar_accordion_placeholder"),
        ui.input_task_button(id="filter_data", label="Filter Data", label_busy="Applying...", type="secondary", disabled=True),
        ui.markdown("<p style='line-height:0.1;'> <br> </p>"),
        ui.output_ui(id="filter_info"),
        ui.download_button(id="download_filter_info", label="Info SVG", width="100%", _class="space-x-2"),
        id="sidebar", open="closed", position="right", bg="f8f8f8", width="280px"
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
                # ui.input_action_button("reset", "Reset", class_="btn-danger"),
                # ui.input_action_button("input_help", "Show help"),
                ui.output_ui("initialize_loader1"),
                ui.markdown("""___"""),
                # File inputs
                ui.row(
                    ui.column(4, ui.div(
                        {"id": "input_file_container_1"},
                        ui.input_text(id=f"condition_label1", label=f"Label:", placeholder="Condition 1"),
                        ui.input_file(id=f"input_file1", label="Upload files:", placeholder="Drag and drop here!", multiple=True),
                        ui.markdown(""" <hr style="border: none; border-top: 1px dotted" /> """),
                    ))
                ),
                # Assigning selected columns - draggable window
                ui.panel_absolute(
                    ui.card(
                        ui.markdown("<h5>Select columns:</h5>"),
                        ui.input_selectize("select_id", "Track identifier:", ["e.g. TRACK_ID"]),
                        ui.input_selectize("select_time", "Time point:", ["e.g. POSITION_T"]),
                        ui.input_selectize("select_x", "X coordinate:", ["e.g. POSITION_X"]),
                        ui.input_selectize("select_y", "Y coordinate:", ["e.g. POSITION_Y"]),
                        ui.markdown("<span style='color:darkgrey; font-style:italic;'>You can drag me around!</span>"),
                        class_="bg-light border-tertiary rounded",
                    ),
                    width="350px", right="500px", top="220px", draggable=True,
                    class_="elevated-panel", style_="z-index: 1000;",
                ),
            ),
        ),

        # ========== DATA GATING ==========
        ui.nav_panel(
            "Gating",
            ui.markdown(
                """ 
                2D Data filtering?
                """
            )
        ),
        

        # ========== PROCESSED DATA DISPLAY ==========
        ui.nav_panel(
            "Data Tables",

            # Input for already processed data
            ui.markdown(
                """ 
                <p style='line-height:0.1;'> <br> </p>
                <h4 style="margin:0;"> Got previously processed data? </h4> 
                <i> Drop in <b>Spot Stats CSV</b> file here: </i>
                """
            ),
            ui.input_file(id="already_processed_input", label=None, placeholder="Drag and drop here!", accept=[".csv"], multiple=False),
            ui.output_ui("initialize_loader2"),
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
                    ui.output_data_frame("render_frame_stats"),
                    ui.download_button("download_frame_stats", "Download CSV"),
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
                                ui.input_selectize("let_me_look_at_these", "Let me look at these:", ["Condition", "Track length", "Track displacement", "Speed mean"], multiple=True),
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
                        ui.markdown(
                            """
                            #### **Superplots**
                            *made with*  `seaborn`
                            <hr style="height: 4px; background-color: black; border: none" />
                            """
                        ),

                        ui.input_selectize(id="type_superplot", label="Plot:", choices=["Swarms", "Violins"], selected="Swarms"),

                        ui.accordion(

                            ui.accordion_panel(
                                "Dataset",
                                # ui.input_selectize("sp_condition", "Condition:", ["all", "not all"]),
                                # ui.panel_conditional(
                                #     "input.sp_condition != 'all'",
                                #     ui.input_selectize("sp_replicate", "Replicate:", ["all", "not all"]),
                                # ),
                                ui.markdown(
                                    """
                                    **Note:** Superplots always show all conditions and replicates. <br>
                                    *In future versions, an option, in which desired conditions can be specified may be added.*
                                    """
                                ),
                            ),

                            ui.accordion_panel(
                                "Metric",
                                ui.input_selectize("sp_metric", label=None, choices=Metrics.Track, selected="Confinement ratio"),
                                # TODO: ui.input_radio_buttons("sp_y_axis", "Y axis with", ["Absolute values", "Relative values"]),
                            ),
                            ui.accordion_panel(
                                "General plot definition",
                                ui.input_selectize(id="sp_palette", label="Color palette:", choices=Styles.PaletteQualitative, selected="tab10"),
                                ui.input_checkbox(id="sp_show_swarms", label="Show swarms", value=True),
                                ui.input_checkbox(id="sp_show_violins", label="Show violins", value=True),
                                ui.input_checkbox(id="sp_show_kde", label="Show KDE", value=False),
                                ui.panel_conditional(
                                    "input.sp_show_kde == true",
                                    ui.input_checkbox(id="sp_kde_legend", label="Show KDE legend", value=False),
                                ),
                                ui.input_checkbox(id="sp_show_cond_mean", label="Show condition means as lines", value=False),
                                ui.input_checkbox(id="sp_show_cond_median", label="Show condition medians as lines", value=False),
                                ui.input_checkbox(id="sp_show_errbars", label="Show error bars", value=False),
                                ui.input_checkbox(id="sp_show_rep_means", label="Show replicate mean bullets", value=True),
                                ui.input_checkbox(id="sp_show_rep_medians", label="Show replicate median bullets", value=False),
                                ui.input_checkbox(id="sp_show_legend", label="Show legend", value=True),
                                ui.input_checkbox(id="sp_grid", label="Show grid", value=True),
                                ui.input_checkbox(id="sp_spine", label="Open axes top/right", value=True),
                                # TODO: ui.input_checkbox(id="sp_flip", label="Flip axes", value=False),
                            ),

                            ui.accordion_panel(
                                "Aesthetics",
                                ui.accordion(

                                    ui.accordion_panel(
                                        "Swarms",
                                        ui.panel_conditional(
                                            "input.sp_show_swarms == true",
                                            ui.input_numeric("sp_swarm_marker_size", "Dot size:", 1, min=0, step=0.5),
                                            ui.input_numeric("sp_swarm_marker_alpha", "Dot opacity:", 0.5, min=0, max=1, step=0.1),
                                            ui.input_selectize("sp_swarm_marker_outline", "Dot outline color:", Styles.Color, selected="black"),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_swarms == false",
                                            ui.markdown(
                                                """
                                                *Swarms not enabled.*
                                                """
                                            )
                                        )
                                    ),

                                    ui.accordion_panel(
                                        "Violins",
                                        ui.panel_conditional(
                                            "input.sp_show_violins == true",
                                            ui.input_selectize("sp_violin_fill", "Fill color:", Styles.Color, selected="whitesmoke"),
                                            ui.input_numeric("sp_violin_alpha", "Fill opacity:", 0.5, min=0, max=1, step=0.1),
                                            ui.input_selectize("sp_violin_outline", "Outline color:", Styles.Color, selected="lightgrey"),
                                            ui.input_numeric("sp_violin_outline_width", "Outline width:", 1, min=0, step=1),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_violins == false",
                                            ui.markdown(
                                                """
                                                *Violins not enabled.*
                                                """
                                            )
                                        ),
                                    ),

                                    ui.accordion_panel(
                                        "Kernel Density Estimate (KDE)",
                                        ui.markdown(
                                            """
                                            *KDEs are computed across data points of specific replicates in each condition, modeling the underlying data distribution* <br>
                                            """
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_kde == true",
                                            ui.input_numeric("sp_kde_line_width", "Outline width:", 1, min=0, step=0.1),
                                            ui.input_checkbox("sp_kde_fill", "Fill area", False),
                                            ui.panel_conditional(
                                                "input.sp_kde_fill == true",
                                                ui.input_numeric("sp_kde_fill_alpha", "Fill opacity:", 0.5, min=0, max=1, step=0.1),
                                            ),
                                            ui.input_numeric("sp_kde_bandwidth", "KDE bandwidth:", 0.75, min=0.1, step=0.1),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_kde == false",
                                            ui.markdown(
                                                """
                                                *KDE not enabled.*
                                                """
                                            )
                                        ),
                                    ),

                                    ui.accordion_panel(
                                        "Lines and error bars",
                                        ui.panel_conditional(
                                            "input.sp_show_cond_mean == true && input.sp_show_cond_median == true",
                                            ui.input_selectize("sp_set_as_primary", label="Set as primary:", choices=["mean", "median"], selected="mean"),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_cond_mean == true",
                                            ui.input_numeric(id="sp_mean_line_span", label="Mean line span length:", value=0.12, min=0, step=0.01),
                                            ui.input_selectize(id="sp_mean_line_color", label="Mean line color:", choices=Styles.Color, selected="black"),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_cond_median == true",
                                            ui.input_numeric(id="sp_median_line_span", label="Median line span length:", value=0.08, min=0, step=0.01),
                                            ui.input_selectize(id="sp_median_line_color", label="Median line color:", choices=Styles.Color, selected="darkblue"),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_cond_mean == true || input.sp_show_cond_median == true",
                                            ui.input_numeric(id="sp_lines_lw", label="Mean/Median Line width:", value=1, min=0, step=0.5),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_errbars == true",
                                            ui.input_numeric(id="sp_errorbar_capsize", label="Error bar cap size:", value=4, min=0, step=1),
                                            ui.input_numeric(id="sp_errorbar_lw", label="Error bar line width:", value=1, min=0, step=0.5),
                                            ui.input_selectize(id="sp_errorbar_color", label="Error bar color:", choices=Styles.Color, selected="black"),
                                            ui.input_numeric(id="sp_errorbar_alpha", label="Error bar opacity:", value=1, min=0, max=1, step=0.1),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_cond_means == false && input.sp_show_cond_medians == false && input.sp_show_errbars == false",
                                            ui.markdown(
                                                """
                                                *Condition means/medians/error bars not enabled.*
                                                """
                                            )
                                        ),
                                    ),

                                    ui.accordion_panel(
                                        "Bullets",
                                        ui.panel_conditional(
                                            "input.sp_show_rep_means == true",
                                            ui.input_numeric("sp_mean_bullet_size", "Mean bullet size:", 80, min=0, step=1),
                                            ui.input_selectize("sp_mean_bullet_outline", "Mean bullet outline color:", Styles.Color, selected="black"),
                                            ui.input_numeric("sp_mean_bullet_outline_width", "Mean bullet outline width:", 0.75, min=0, step=0.05),
                                            ui.input_numeric("sp_mean_bullet_alpha", "Mean bullet opacity:", 1, min=0, max=1, step=0.1),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_rep_medians == true",
                                            ui.input_numeric("sp_median_bullet_size", "Median bullet size:", 50, min=0, step=1),
                                            ui.input_selectize("sp_median_bullet_outline", "Median bullet outline color:", Styles.Color, selected="black"),
                                            ui.input_numeric("sp_median_bullet_outline_width", "Median bullet outline width:", 0.75, min=0, step=0.05),
                                            ui.input_numeric("sp_median_bullet_alpha", "Median bullet opacity:", 1, min=0, max=1, step=0.1),
                                        ),
                                        ui.panel_conditional(
                                            "input.sp_show_rep_means == false && input.sp_show_rep_medians == false",
                                            ui.markdown(
                                                """
                                                *Replicate means/medians not enabled.*
                                                """
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),

                        ui.markdown(""" <br> """),
                        ui.input_task_button(id="sp_generate", label="Generate", class_="btn-secondary", width="100%"),
                    ),
                    ui.markdown(""" <br> """),
                    ui.output_plot(id="swarmplot"),
                ),
                widths = (2, 10)
            ),
        ),
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
    threshold_list = reactive.Value([1])  # Start with one threshold
    property_selections = reactive.Value({})
    filter_type_selections = reactive.Value({})
    quantile_selections = reactive.Value({})
    reference_selections = reactive.Value({})
    metric_x_selections = reactive.Value({})
    metric_y_selections = reactive.Value({})
    threshold_slider_outputs = {}
    thresholding_histogram_outputs = {}

    thresholding_memory = reactive.Value({})  # initialize empty first

    @reactive.Effect
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
    UNFILTERED_FRAMESTATS = reactive.Value(pd.DataFrame())    # Placeholder for frame statistics
    SPOTSTATS = reactive.Value(pd.DataFrame())       # Placeholder for processed spot statistics
    TRACKSTATS = reactive.Value(pd.DataFrame())      # Placeholder for processed track statistics
    FRAMESTATS = reactive.Value(pd.DataFrame())       # Placeholder for processed frame statistics



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


    # Helper that builds one pair wrapped in a removable container
    def _input_container_ui(id: int):
        return ui.div(
            {"id": f"input_file_container_{id}"},
            ui.input_text(
                id=f"condition_label{id}",
                label="Label:",
                placeholder=f"Condition {id}",
            ),
            ui.input_file(
                id=f"input_file{id}",
                label="Upload files:",
                placeholder="Drag and drop here!",
                multiple=True,
            ),
            ui.markdown('<hr style="border: none; border-top: 1px dotted" />'),
        )

    # ADD: append the newest id from input_list right after #input_file_first (or wherever you want)
    @reactive.effect
    @reactive.event(input.add_input)
    def _add_container():
        ids = input_list.get()
        ui.insert_ui(
            ui=_input_container_ui(ids[-1]),
            selector=f"#input_file_container_{ids[-2]}",
            where="afterEnd"
        )

    # REMOVE: remove the container for the latest id in input_list
    @reactive.effect
    @reactive.event(input.remove_input)
    def _remove_container():
        ids = input_list.get()

        ui.insert_ui(
            ui.tags.script(
                f"Shiny.setInputValue('input_file{ids[-1]+1}', null, {{priority:'event'}});"
                f"Shiny.setInputValue('condition_label{ids[-1]+1}', '', {{priority:'event'}});"
                # Clear browser chooser if the element still exists
            ), 
            selector="body", 
            where="beforeEnd"
        )

        ui.remove_ui(
            selector=f"#input_file_container_{ids[-1]+1}",
            multiple=True
        )
    
    # - - - - - - - - - - - - - - - - - - - -




    # - - - - Required columns specification - - - -

    @reactive.Effect
    def column_selection():
        ids = input_list.get()
        ui.update_selectize(id="select_id", choices=["e.g. TRACK ID"])
        ui.update_selectize(id="select_time", choices=["e.g. POSITION T"])
        ui.update_selectize(id="select_x", choices=["e.g. POSITION X"])
        ui.update_selectize(id="select_y", choices=["e.g. POSITION Y"])

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



    # - - - - Already processed data input - - - -

    @reactive.Effect
    @reactive.event(input.already_processed_input)
    def load_processed_data():
        fileinfo = input.already_processed_input()
        try:
            df = DataLoader.GetDataFrame(fileinfo[0]["datapath"])

            UNFILTERED_SPOTSTATS.set(df)
            UNFILTERED_TRACKSTATS.set(Calc.Tracks(df))
            UNFILTERED_FRAMESTATS.set(Calc.Frames(df))

            SPOTSTATS.set(df)
            TRACKSTATS.set(Calc.Tracks(df))
            FRAMESTATS.set(Calc.Frames(df))

            # ui.update_sidebar(id="sidebar", show=True)
        except Exception as e:
            print(e)



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
            RAWDATA.set(pd.concat(all_data, axis=0))
            UNFILTERED_SPOTSTATS.set(Calc.Spots(RAWDATA.get()))
            UNFILTERED_TRACKSTATS.set(Calc.Tracks(RAWDATA.get()))
            UNFILTERED_FRAMESTATS.set(Calc.Frames(RAWDATA.get()))
            ui.update_sidebar(id="sidebar", show=True)
        else:
            pass

    @reactive.extended_task
    async def loader1():
        with ui.Progress(min=0, max=12) as p:
            p.set(message="Initialization in progress")

            for i in range(0, 10):
                p.set(i, message="Initializing Peregrin...")
                await asyncio.sleep(0.04)
        pass


    @render.text
    @reactive.event(input.run, ignore_none=True)
    def initialize_loader1():
        return loader1()
        


    # - - - - - - - - - - - - - - - - - - - -





    # - - - - Sidebar accordion layout for thresholds - - - -

    @output()
    @render.ui
    def sidebar_accordion_placeholder():
        if threshold_dimension.get() == "1D":
            return ui.accordion(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_numeric("bins", "Number of bins", value=25, min=1, step=1),
                    ui.markdown("<p style='line-height:0.1;'> <br> </p>"),
                    ui.input_action_button(id="threshold_dimensional_toggle", label=dimension_button_label.get(), width="100%"),
                ),
                ui.accordion_panel(
                    f"Threshold 1",
                    ui.panel_well(
                        ui.input_selectize(f"threshold_property_1", "Property", choices=Metrics.Thresholding.Properties),
                        ui.input_selectize(f"threshold_filter_1", "Filter type", choices=Modes.Thresholding),
                        ui.panel_conditional(
                            f"input.threshold_filter_1 == 'Quantile'",
                            ui.input_selectize(f"threshold_quantile_1", "Quantile", choices=[200, 100, 50, 25, 20, 10, 5, 4, 2], selected=100),
                        ),
                        ui.panel_conditional(
                            f"input.threshold_filter_1 == 'Relative to...'",
                            ui.input_selectize(f"reference_value_1", "Reference value", choices=["Mean", "Median", "My own value"]),
                            ui.panel_conditional(
                                f"input.reference_value_1 == 'My own value'",
                                ui.input_numeric(f"my_own_value_1", "My own value", value=0, step=1)
                            ),
                        ),
                        ui.output_ui(f"manual_threshold_value_setting_1"),
                        ui.output_ui(f"threshold_slider_placeholder_1"),
                        ui.output_plot(f"thresholding_histogram_placeholder_1"),
                    ),
                ),
                id="threshold_accordion",
                open="Threshold 1",
            )
        elif threshold_dimension.get() == "2D":
            return ui.accordion(
                ui.accordion_panel(
                    "Filter settings",
                    ui.input_numeric(id="threshold2d_array_size", label="Dot Size:", value=1, min=0, step=1),
                    # ui.input_selectize(id="threshold2d_array_color_selected", label="Color Selected:", choices=Metrics.Thresholding.ColorArray.ColorSelected, selected="default"),
                    # ui.input_selectize(id="threshold2d_array_color_unselected", label="Color Unselected:", choices=Metrics.Thresholding.ColorArray.ColorUnselected, selected="default"),
                    ui.markdown("<p style='line-height:0.1;'> <br> </p>"),
                    ui.input_action_button(id="threshold_dimensional_toggle", label=dimension_button_label.get(), class_="btn-secondary", width="100%"),
                ),
                ui.accordion_panel(
                    f"Threshold 1",
                    ui.panel_well(
                        ui.markdown(""" <h6>  Properties X;Y  </h6>"""),
                        ui.input_selectize(f"thresholding_metric_X_1", None, Metrics.Thresholding.Properties, selected="Confinement ratio"),
                        ui.input_selectize(f"thresholding_metric_Y_1", None, Metrics.Thresholding.Properties, selected="Track length"),
                        ui.div(
                            {"style": "position:relative; width:100%; padding-top:100%; padding-bottom:50%;"},
                            ui.div(
                                {"style": "position:absolute; inset:0;"},
                                output_widget(f"threshold2d_plot_1")
                            )
                        ),
                        ui.markdown(""" <p> </p> """),
                        ui.input_action_button(id=f"threshold2d_clear_1", label="Clear", class_="space-x-2", width="100%"),
                    ),
                ),
                id="threshold_accordion",
                open="Threshold 1"
            )
    

    # - - - - Adding and removing thresholds - - - -

    def render_threshold_accordion_panel(id):
        if threshold_dimension.get() == "1D":
            print("Rendering threshold panel", id)
            return ui.accordion_panel(
                f"Threshold {id}",
                ui.panel_well(
                    ui.input_selectize(f"threshold_property_{id}", "Property", choices=Metrics.Thresholding.Properties),
                    ui.input_selectize(f"threshold_filter_{id}", "Filter type", choices=Modes.Thresholding),
                    ui.panel_conditional(
                        f"input.threshold_filter_{id} == 'Quantile'",
                        ui.input_selectize(f"threshold_quantile_{id}", "Quantile", choices=[200, 100, 50, 25, 20, 10, 5, 4, 2], selected=100),
                    ),
                    ui.panel_conditional(
                        f"input.threshold_filter_{id} == 'Relative to...'",
                        ui.input_selectize(f"reference_value_{id}", "Reference value", choices=["Mean", "Median", "My own value"]),
                        ui.panel_conditional(
                            f"input.reference_value_{id} == 'My own value'",
                            ui.input_numeric(f"my_own_value_{id}", "My own value", value=0, step=1)
                        ),
                    ),
                    ui.output_ui(f"manual_threshold_value_setting_{id}"),
                    ui.output_ui(f"threshold_slider_placeholder_{id}"),
                    ui.output_plot(f"thresholding_histogram_placeholder_{id}"),
                ),
            )
        elif threshold_dimension.get() == "2D":
            return ui.accordion_panel(
                f"Threshold {id}",
                ui.panel_well(
                    ui.markdown(""" <h6>  Properties X;Y  </h6>"""),
                    ui.input_selectize(f"thresholding_metric_X_{id}", None, Metrics.Thresholding.Properties, selected="Confinement ratio"),
                    ui.input_selectize(f"thresholding_metric_Y_{id}", None, Metrics.Thresholding.Properties, selected="Track length"),
                    ui.div(
                        {"style": "position:relative; width:100%; padding-top:100%; padding-bottom:50%;"},
                        ui.div(
                            {"style": "position:absolute; inset:0;"},
                            output_widget(f"threshold2d_plot_{id}")
                        )
                    ),
                    ui.markdown(""" <p> </p> """),
                    ui.input_action_button(id=f"threshold2d_clear_{id}", label="Clear", class_="space-x-2", width="100%"),
                ),
            )


    @reactive.Effect
    @reactive.event(input.add_threshold)
    def add_threshold():
        ids = threshold_list.get()
        threshold_list.set(ids + [ids[-1] + 1] if ids else [1])
        id = ids[-1] + 1
        session.send_input_message("remove_threshold", {"disabled": False})

        print("Adding threshold panel", threshold_list.get())

        ui.insert_accordion_panel(
            id="threshold_accordion",
            panel=render_threshold_accordion_panel(id),
            position="after"
        )

    @reactive.Effect
    @reactive.event(input.remove_threshold)
    def remove_threshold():
        ids = threshold_list.get()
        id = ids[-1]
        if len(ids) > 1:
            threshold_list.set(ids[:-1])
        if len(threshold_list.get()) <= 1:
            session.send_input_message("remove_threshold", {"disabled": True})
        
        ui.remove_accordion_panel(
            id="threshold_accordion",
            target=f"Threshold {id}"
        )


    # - - - - Filtered info display - - - -

    @output()
    @render.ui
    def filter_info():

        # --- Threshold blocks (only if 1D and thresholds exist)
        try:
            if threshold_dimension.get() == "1D":
                blocks = []
                thresholds_state = thresholds1d_state.get()

                # iterate deterministically if keys are integers
                for t in sorted(thresholds_state.keys()):
                    if t not in threshold_list.get():
                        break
                    try:
                        t_state = thresholds_state.get(t)
                        t_state_after = thresholds_state.get(t + 1)
                        
                        data = len(t_state.get("tracks"))
                        data_after = len(t_state_after.get("tracks")) if t_state_after else data
                        out = data - data_after
                        out_percent = round(out / data * 100) if data else 0

                        prop = input[f"threshold_property_{t}"]()
                        ftype = input[f"threshold_filter_{t}"]()
                        if ftype == "Relative to...":
                            ref = input[f"reference_value_{t}"]()
                            if ref == "My own value":
                                ref_val = input[f"threshold_my_own_value_{t}"]()
                            else:
                                ref_val = ref
                            reference = f"<br>Reference: <br><i><b>{ref}</b> (<b>{ref_val}</b>)</i><br>" if not isinstance(ref_val, str) else f"<br>Reference: <br><i><b>{ref}</b></i><br>"
                        else:
                            reference =  ""
                        vals = input[f"threshold_slider_{t}"]()

                    except Exception:
                        break

                    blocks.append(
                        ui.markdown(
                            f"""
                            <div style="height:5px;"></div>
                                <hr style="border:0; border-top:1px solid #000000; margin:8px 0;">
                            <div style="height:5px;"></div>
                            <p style="margin-bottom:8px; margin-top:10px;">
                                <b><h5>Threshold {t + 1}</h5></b>
                                Filtered out: <br>
                                <i><b>{out}</b> (<b>{out_percent}%</b>)</i>
                            </p>
                            <p style="margin-bottom:8px; margin-top:0px;">
                                Property: <br>
                                <i><b>{prop}</b></i> <br>
                                Filter: <br>
                                <i><b>{ftype}</b></i> <br>
                                Range: <br>
                                <i><b>{vals[0]}</b> - <b>{vals[1]}</b></i>
                                {reference}
                            """
                        )
                    )


            elif threshold_dimension.get() == "2D":
                blocks = []
                thresholds_state = thresholds2d_state.get()

                # iterate deterministically if keys are integers
                for t in sorted(thresholds_state.keys()):
                    if t not in threshold_list.get():
                        break
                    try:
                        t_state = thresholds_state.get(t)
                        t_state_after = thresholds_state.get(t + 1)
                        
                        data = len(t_state.get("tracks"))
                        data_after = len(t_state_after.get("tracks")) if t_state_after else data
                        out = data - data_after
                        out_percent = round(out / data * 100) if data else 0

                        propX = input[f"thresholding_metric_X_{t}"]()
                        propY = input[f"thresholding_metric_Y_{t}"]()

                        # print('--------------------')

                        # print(propX, propY)
                        try: 
                            track_data = t_state_after.get("tracks")
                            spot_data = t_state_after.get("spots")
                        except Exception:
                            track_data = t_state.get("tracks")
                            spot_data = t_state.get("spots")
                        
                        dataX = track_data[propX] if propX in Metrics.Track else spot_data[propX]
                        dataY = track_data[propY] if propY in Metrics.Track else spot_data[propY]
                        
                        if propX == "Confinement ratio":
                            minX, maxX = f"{min(dataX):.2f}", f"{ceil(max(dataX)):.2f}"
                        else:
                            minX, maxX = floor(min(dataX)), ceil(max(dataX))
                        if propY == "Confinement ratio":
                            minY, maxY = f"{min(dataY):.2f}", f"{ceil(max(dataY)):.2f}"
                        else:
                            minY, maxY = floor(min(dataY)), ceil(max(dataY))


                    except Exception:
                        break

                    blocks.append( 
                        ui.markdown(
                            f"""
                            <div style="height:5px;"></div>
                                <hr style="border:0; border-top:1px solid #000000; margin:8px 0;">
                            <div style="height:5px;"></div>
                            <p style="margin-bottom:8px; margin-top:10px;">
                                <b><h5>Threshold {t + 1}</h5></b>
                                Filtered out: <br>
                                <i><b>{out}</b> (<b>{out_percent}%</b>)</i>
                            </p>
                            <p style="margin-bottom:8px; margin-top:0px;">
                                Properties: <br>
                            <div style="height:5px;"></div>
                                <i><b>{propX}</b></i> <br>
                                Range: <br>
                                <i><b>{minX}</b> - <b>{maxX}</b></i> <br>
                            <div style="height:5px;"></div>
                                <i><b>{propY}</b></i> <br>
                                Range: <br>
                                <i><b>{minY}</b> - <b>{maxY}</b></i>
                            <p>
                            """
                        )
                    )

        except Exception:
            pass

        total_tracks = len(UNFILTERED_TRACKSTATS.get())
        filtered_tracks = len(TRACKSTATS.get())

        filtered_tracks_percent = (
            round(filtered_tracks / total_tracks * 100) if total_tracks else 0
        )

        # --- Header + summary block
        blocks.insert(0,
            ui.markdown(
                f"""
                <p style="margin-bottom:0px; margin-top:0px;">
                    <h4> <b> Info </b> </h4>
                </p>
                <p style="margin-bottom:8px; margin-top:12px;">
                    Cells in total: <br>
                    <i><b>{total_tracks}</b> <br></i>
                </p>
                <p style="margin-bottom:8px; margin-top:0px;">
                    In focus: <br>
                    <i><b>{filtered_tracks}</b> (<b>{filtered_tracks_percent}%</b>)</i>
                </p>
                """
            )
        )

        # Return a single well with all blocks as children
        return ui.panel_well(*blocks)



    def GetInfoSVG(*, width: int = 190, txt_color: str = "#000000") -> str:
        """
        Build an SVG 'Info' panel using current Shiny reactives.
        Works for both 1D and 2D thresholding like in your filter_info().
        """
        

        # ---------- helpers ----------
        pad = 16
        title_size = 18
        body_size = 14
        line_gap = 8
        section_gap = 14
        rule_gap = 5
        rule_color = "#000000"
        font_family = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"

        def addy(y, inc):  # move the cursor
            return y + inc

        def tspan(text, cls=None):
            if cls:
                return f'<tspan class="{cls}">{escape(str(text))}</tspan>'
            return f"<tspan>{escape(str(text))}</tspan>"

        # ---------- totals ----------
        total_tracks = len(UNFILTERED_TRACKSTATS.get())
        filtered_tracks = len(TRACKSTATS.get())
        if total_tracks < 0:
            return ""
        if filtered_tracks < 0:
            filtered_tracks = total_tracks
        percent = 0 if total_tracks == 0 else round((filtered_tracks / total_tracks) * 100)

        # ---------- SVG header (height placeholder) ----------
        x = pad
        y = pad + title_size
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="__HEIGHT__" '
            f'viewBox="0 0 {width} __HEIGHT__" role="img" aria-label="Info panel">',
            '<style>.bold{font-weight:700}.ital{font-style:italic}</style>',
            f'<text x="{x}" y="{y}" font-family="{font_family}" font-size="{title_size}" '
            f'font-weight="700" fill="{txt_color}">Info</text>',
            f'<g font-family="{font_family}" font-size="{body_size}" fill="{txt_color}">'
        ]

        # ---------- header body ----------
        y = addy(y, section_gap + body_size)
        parts.append(f'<text x="{x}" y="{y}">Cells in total:</text>')
        y = addy(y, body_size + line_gap)
        parts.append(f'<text x="{x}" y="{y}"><tspan class="bold">{total_tracks}</tspan></text>')

        y = addy(y, section_gap + body_size)
        parts.append(f'<text x="{x}" y="{y}">In focus:</text>')
        y = addy(y, body_size + line_gap)
        parts.append(
            f'<text x="{x}" y="{y}">{tspan(filtered_tracks,"bold")} {tspan(f"({percent}%)","bold")}</text>'
        )

        # ---------- thresholds (read reactives exactly like your UI) ----------
        try:
            if threshold_dimension.get() == "1D":
                thresholds_state = thresholds1d_state.get()
                for t in sorted(thresholds_state.keys()):
                    if t not in threshold_list.get():
                        break
                    try:
                        t_state = thresholds_state.get(t)
                        t_state_after = thresholds_state.get(t + 1)

                        data = len(t_state.get("tracks"))
                        data_after = len(t_state_after.get("tracks")) if t_state_after else data
                        out = data - data_after
                        out_percent = round(out / data * 100) if data else 0

                        prop = input[f"threshold_property_{t}"]()
                        ftype = input[f"threshold_filter_{t}"]()
                        if ftype == "Relative to...":
                            ref = input[f"reference_value_{t}"]()
                            if ref == "My own value":
                                ref_val = input[f"threshold_my_own_value_{t}"]()
                            else:
                                ref_val = ref
                            reference = f"{ref} ({ref_val})" if not isinstance(ref_val, str) else f"{ref}"
                        else:
                            reference = ""

                        vmin, vmax = input[f"threshold_slider_{t}"]()
                    except Exception:
                        break

                    # hr
                    y = addy(y, rule_gap + section_gap)
                    parts.append(f'<line x1="{pad}" x2="{width-pad}" y1="{y}" y2="{y}" stroke="{rule_color}" stroke-width="1"/>')
                    y = addy(y, rule_gap)

                    # threshold header
                    y = addy(y, body_size + line_gap)
                    parts.append(f'<text x="{x}" y="{y}">{tspan(f"Threshold {t+1}","bold")}</text>')

                    # filtered out
                    y = addy(y, body_size + line_gap)
                    parts.append(
                        f'<text x="{x}" y="{y}">Filtered out: '
                        f'{tspan(out,"ital bold")} {tspan(f"({out_percent}%)","ital bold")}</text>'
                    )

                    # property / filter / range / reference
                    y = addy(y, body_size + section_gap)
                    parts.append(f'<text x="{x}" y="{y}">Property:</text>')
                    y = addy(y, body_size)
                    parts.append(f'<text x="{x}" y="{y}">{tspan(prop,"bold ital")}</text>')

                    y = addy(y, body_size + section_gap)
                    parts.append(f'<text x="{x}" y="{y}">Filter:</text>')
                    y = addy(y, body_size)
                    parts.append(f'<text x="{x}" y="{y}">{tspan(ftype,"bold ital")}</text>')

                    y = addy(y, body_size + section_gap)
                    parts.append(f'<text x="{x}" y="{y}">Range:</text>')
                    y = addy(y, body_size)
                    parts.append(
                        f'<text x="{x}" y="{y}">{tspan(vmin,"bold ital")} - {tspan(vmax,"bold ital")}</text>'
                    )

                    if reference:
                        y = addy(y, body_size + section_gap)
                        parts.append(f'<text x="{x}" y="{y}">Reference:</text>')
                        y = addy(y, body_size)
                        parts.append(f'<text x="{x}" y="{y}">{tspan(reference,"bold ital")}</text>')

            elif threshold_dimension.get() == "2D":
                thresholds_state = thresholds2d_state.get()
                for t in sorted(thresholds_state.keys()):
                    if t not in threshold_list.get():
                        break
                    try:
                        t_state = thresholds_state.get(t)
                        t_state_after = thresholds_state.get(t + 1)

                        data = len(t_state.get("tracks"))
                        data_after = len(t_state_after.get("tracks")) if t_state_after else data
                        out = data - data_after
                        out_percent = round(out / data * 100) if data else 0

                        propX = input[f"thresholding_metric_X_{t}"]()
                        propY = input[f"thresholding_metric_Y_{t}"]()

                        try:
                            track_data = t_state_after.get("tracks")
                            spot_data = t_state_after.get("spots")
                        except Exception:
                            track_data = t_state.get("tracks")
                            spot_data = t_state.get("spots")

                        dataX = track_data[propX] if propX in Metrics.Track else spot_data[propX]
                        dataY = track_data[propY] if propY in Metrics.Track else spot_data[propY]

                        if propX == "Confinement ratio":
                            minX, maxX = f"{min(dataX):.2f}", f"{ceil(max(dataX)):.2f}"
                        else:
                            minX, maxX = floor(min(dataX)), ceil(max(dataX))
                        if propY == "Confinement ratio":
                            minY, maxY = f"{min(dataY):.2f}", f"{ceil(max(dataY)):.2f}"
                        else:
                            minY, maxY = floor(min(dataY)), ceil(max(dataY))
                    except Exception:
                        break

                    # hr
                    y = addy(y, rule_gap + section_gap)
                    parts.append(f'<line x1="{pad}" x2="{width-pad}" y1="{y}" y2="{y}" stroke="{rule_color}" stroke-width="1"/>')
                    y = addy(y, rule_gap)

                    # threshold header
                    y = addy(y, body_size + line_gap)
                    parts.append(f'<text x="{x}" y="{y}">{tspan(f"Threshold {t+1}","bold")}</text>')

                    # filtered out
                    y = addy(y, body_size + line_gap)
                    parts.append(
                        f'<text x="{x}" y="{y}">Filtered out: '
                        f'{tspan(out,"ital bold")} {tspan(f"({out_percent}%)","ital bold")}</text>'
                    )

                    # properties for X and Y
                    y = addy(y, body_size + section_gap)
                    parts.append(f'<text x="{x}" y="{y}">Properties:</text>')

                    # X
                    y = addy(y, rule_gap)
                    parts.append(f'<text x="{x}" y="{y}">{tspan(propX,"bold ital")}</text>')
                    y = addy(y, body_size)
                    parts.append(f'<text x="{x}" y="{y}">Range:</text>')
                    y = addy(y, body_size)
                    parts.append(
                        f'<text x="{x}" y="{y}">{tspan(minX,"bold ital")} - {tspan(maxX,"bold ital")}</text>'
                    )

                    # Y
                    y = addy(y, rule_gap)
                    parts.append(f'<text x="{x}" y="{y}">{tspan(propY,"bold ital")}</text>')
                    y = addy(y, body_size)
                    parts.append(f'<text x="{x}" y="{y}">Range:</text>')
                    y = addy(y, body_size)
                    parts.append(
                        f'<text x="{x}" y="{y}">{tspan(minY,"bold ital")} - {tspan(maxY,"bold ital")}</text>'
                    )

        except Exception:
            pass

        # ---------- close and set height ----------
        parts.append("</g></svg>")
        height = y + pad
        svg = "".join(parts).replace("__HEIGHT__", str(height))
        return svg


    @render.download(filename=f"Filter Info {date.today()}.svg", media_type="svg")
    def download_filter_info():
        svg = GetInfoSVG()
        yield svg.encode("utf-8")

        

    # - - - - Threshold dimension toggle - - - -

    @reactive.Effect
    @reactive.event(input.threshold_dimensional_toggle)
    def _threshold_dimensional_toggle():
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
    



    # - - - - Initialize data memory - - - -

    thresholds1d_state = reactive.Value({int: dict})
    thresholds2d_state = reactive.Value({int: dict})

    @reactive.Effect
    @reactive.event(input.threshold_dimensional_toggle, input.run, input.already_processed_input)
    def _initialize_thresholding_memory():
        threshold_list.unset()
        threshold_list.set([1])

        if threshold_dimension.get() == "1D":
            thresholds1d_state.set({1: {"spots": UNFILTERED_SPOTSTATS.get(), "tracks": UNFILTERED_TRACKSTATS.get()}})
        elif threshold_dimension.get() == "2D":
            thresholds2d_state.set({1: {"spots": UNFILTERED_SPOTSTATS.get(), "tracks": UNFILTERED_TRACKSTATS.get()}})

        SPOTSTATS.set(UNFILTERED_SPOTSTATS.get())
        TRACKSTATS.set(UNFILTERED_TRACKSTATS.get())
        FRAMESTATS.set(UNFILTERED_FRAMESTATS.get())


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
            ref = float(my_value) if isinstance(my_value, (int, float)) else 0.0
        else:
            ref = float(vals.mean())

        max_delta = float(np.max(np.abs(vals - ref)))
        return ref, max_delta


    # - - - - Threshold slider generator - - - -

    def _get_steps(highest):
        """
        Returns the step size for the slider based on the range.
        """
        if highest < 0.01:
            steps = 0.0001
        elif 0.01 <= highest < 0.1:
            steps = 0.001
        elif 0.1 <= highest < 1:
            steps = 0.01
        elif 1 <= highest < 10:
            steps = 0.1
        elif 10 <= highest < 1000:
            steps = 1
        elif 1000 <= highest < 100000:
            steps = 10
        elif 100000 < highest:
            steps = 100
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
            steps = _get_steps(highest)
            values, ref_val = _get_threshold_memory(memory, threshold_id, property_name, filter_type, default)
            minimal = floor(lowest)
            maximal = ceil(highest)

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

            steps = _get_steps(highest)
            values, ref_val = _get_threshold_memory(
                memory, threshold_id, property_name, filter_type, default,
                reference=reference, ref_val=reference_value
            )
            minimal = floor(lowest)
            maximal = ceil(highest)
            
        return steps, values, ref_val, minimal, maximal


    threshold1d_df_memory = reactive.Value({0: {"spots": pd.DataFrame(), "tracks": pd.DataFrame()}})

    @reactive.Effect
    @reactive.event(UNFILTERED_SPOTSTATS, UNFILTERED_TRACKSTATS)
    def initialize_threshold1d_memory():
        data = {
            0: {
                "spots": UNFILTERED_SPOTSTATS.get(),
                "tracks": UNFILTERED_TRACKSTATS.get()
            }
        }
        threshold1d_df_memory.set(data)



    # Make threshold sliders dynamically based on the threshold ID
    def render_threshold_slider(threshold_id):
        @output(id=f"threshold_slider_placeholder_{threshold_id}")
        @render.ui
        def threshold_slider():

            # data_memory = threshold1d_df_memory.get()
            data_memory = thresholds1d_state.get()
            current_data = data_memory.get(threshold_id)
            req(current_data is not None and current_data.get("spots") is not None and current_data.get("tracks") is not None)

            spot_data = current_data.get("spots")
            track_data = current_data.get("tracks")


            # spot_data = UNFILTERED_SPOTSTATS.get()
            # track_data = UNFILTERED_TRACKSTATS.get()
            # if spot_data is None or spot_data.empty or track_data is None or track_data.empty:
                # return

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

            # data_memory = threshold1d_df_memory.get()
            data_memory = thresholds1d_state.get()
            current_data = data_memory.get(threshold_id)
            req(current_data is not None and current_data.get("spots") is not None and current_data.get("tracks") is not None)

            spot_data = current_data.get("spots")
            track_data = current_data.get("tracks")

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
                    f"floor_threshold_value_{threshold_id}",
                    label="min",
                    value=v_lo,
                    min=min_fmt,
                    max=max_fmt,
                    step=steps
                )),
                ui.column(6, ui.input_numeric(
                    f"roof_threshold_value_{threshold_id}",
                    label="max",
                    value=v_hi,
                    min=min_fmt,
                    max=max_fmt,
                    step=steps
                )),
            )


    # - - - - Threshold histogram generator - - - -

    @Debounce(1)
    @reactive.Calc
    def get_bins():
        return input.bins() if input.bins() is not None and input.bins() != 0 else 25

    def render_threshold_histogram(threshold_id):
        @output(id=f"thresholding_histogram_placeholder_{threshold_id}")
        @render.plot
        def threshold_histogram():

            # data_memory = threshold1d_df_memory.get()
            data_memory = thresholds1d_state.get()
            current_data = data_memory.get(threshold_id)
            req(current_data is not None and current_data.get("spots") is not None and current_data.get("tracks") is not None)

            if input[f"threshold_property_{threshold_id}"]() in Metrics.Thresholding.SpotProperties:
                data = current_data.get("spots")
            if input[f"threshold_property_{threshold_id}"]() in Metrics.Thresholding.TrackProperties:
                data = current_data.get("tracks")
            if data is None or data.empty:
                return
            
            property = input[f"threshold_property_{threshold_id}"]()
            filter_type = input[f"threshold_filter_{threshold_id}"]()
            try:
                slider_low_pct, slider_high_pct = input[f"threshold_slider_{threshold_id}"]()
            except Exception:
                return

            if filter_type == "Literal":

                bins = get_bins()
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
                        mv = input[f"my_own_value_{threshold_id}"]() if input[f"my_own_value_{threshold_id}"]() is not None else 0.0
                        reference_value = float(mv) if isinstance(mv, (int, float)) else 0.0
                    except Exception:
                        reference_value = 0.0
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



    
    def _filter_data_1d(df, threshold: tuple, property: str, filter_type: str, reference: str = None, reference_value: float = None):
        if df is None or df.empty or property is None or property not in df.columns:
            return df
        
        try:
            working_df = df[property].dropna()
        except Exception:
            return df
        
        _floor, _roof = threshold
        if (
            _floor is None or _roof is None
            or not isinstance(_floor, (int, float)) or not isinstance(_roof, (int, float))
        ):
            return working_df

        if filter_type == "Literal":
            return working_df[(working_df >= _floor) & (working_df <= _roof)]

        elif filter_type == "Normalized 0-1":
            normalized = Threshold.Normalize_01(df, property)
            return normalized[(normalized >= _floor) & (normalized <= _roof)]

        elif filter_type == "Quantile":
            
            q_floor, q_roof = _floor / 100, _roof / 100
            if not 0 <= q_floor <= 1 or not 0 <= q_roof <= 1:
                q_floor, q_roof = 0, 1

            lower_bound = np.quantile(working_df, q_floor)
            upper_bound = np.quantile(working_df, q_roof)
            return working_df[(working_df >= lower_bound) & (working_df <= upper_bound)]

        elif filter_type == "Relative to...":
            # req(reference is not None)
            if reference is None:
                reference = 0.0
            ref, _ = _compute_reference_and_span(working_df, reference, reference_value)

            print(f"Reference value: {ref}, Floor: {ref + _floor}, Roof: {ref + _roof}, -Floor: {ref - _floor}, -Roof: {ref - _roof}")

            return working_df[
                (working_df >= (ref + _floor)) 
                & (working_df <= (ref + _roof))
                | (working_df <= (ref - _floor)) 
                & (working_df >= (ref - _roof))    
            ]

        return df



    @reactive.Effect
    def _stash_threshold1d_df():
        for threshold_id in threshold_list.get():

            property_name = input[f"threshold_property_{threshold_id}"]()
            filter_type = input[f"threshold_filter_{threshold_id}"]()
            property_name = input[f"threshold_property_{threshold_id}"]()
            filter_type = input[f"threshold_filter_{threshold_id}"]()
            # quantile = input[f"threshold_quantile_{threshold_id}"]()
            reference = input[f"reference_value_{threshold_id}"]()
            floor_value = input[f"floor_threshold_value_{threshold_id}"]()
            roof_value = input[f"roof_threshold_value_{threshold_id}"]()
            

            for tid, i in enumerate(threshold_list.get(), start=threshold_id):
                try:
                    # data_memory = threshold1d_df_memory.get()
                    data_memory = thresholds1d_state.get()
                    current_data = data_memory[threshold_id]
                    req(current_data is not None and current_data.get("spots") is not None and current_data.get("tracks") is not None)

                    print("-----------------------------")
                    print(f"Applying 1D threshold ID {threshold_id} on property '{property_name}' with filter '{filter_type}'")

                    filter_df = _filter_data_1d(
                        df=current_data.get("tracks") if property_name in Metrics.Thresholding.TrackProperties else current_data.get("spots"),
                        threshold=(floor_value, roof_value),
                        property=property_name,
                        filter_type=filter_type,
                        reference=reference,
                        reference_value=input[f"my_own_value_{threshold_id}"]() if reference == "My own value" else None
                    )

                    print("Filtered DataFrame:")
                    print(filter_df)

                    spots_input = current_data.get("spots")
                    tracks_input = current_data.get("tracks")

                    spots_output = spots_input.loc[filter_df.index.intersection(spots_input.index)]
                    tracks_output = tracks_input.loc[filter_df.index.intersection(tracks_input.index)]

                    print(f"Spots after filtering: {len(spots_output)}")
                    print(f"Tracks after filtering: {len(tracks_output)}")

                    data_memory[tid + 1] = {
                        "spots": spots_output,
                        "tracks": tracks_output
                    }

                    # threshold1d_df_memory.set(data_memory)
                    thresholds1d_state.set(data_memory)

                    # render_threshold_slider(tid + 1)
                    # render_threshold_histogram(tid + 1)
                    # render_manual_threshold_values_setting(tid + 1)
                    # register_threshold_sync(tid + 1)

                except Exception:
                    pass



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
        @reactive.Effect
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
            if ref == "My own value":
                if not isinstance(ref_val, (int, float)):
                    ref_val = None

            # Compute new memory once
            need_vals = (cur_vals is None or not _nearly_equal_pair(vals, cur_vals))
            need_ref  = (ref == "My own value")
            # need_ref  = (ref == "My own value" and ref_val is not None and (cur_ref_val is None or abs(float(ref_val) - float(cur_ref_val)) > 1e-12))

            if need_vals or need_ref:
                new_mem = _set_threshold_memory(mem.copy(), threshold_id, prop, ftype, vals, quantile=q, reference=ref, ref_val=cur_ref_val)
                thresholding_memory.set(new_mem)


        # B) manual numerics -> memory
        @reactive.Effect
        @reactive.event(
            input[f"floor_threshold_value_{threshold_id}"],
            input[f"roof_threshold_value_{threshold_id}"],
        )
        def _manual_to_memory():
            lo = input[f"floor_threshold_value_{threshold_id}"]()
            hi = input[f"roof_threshold_value_{threshold_id}"]()
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
                    _set_threshold_memory(mem.copy(), threshold_id, prop, ftype, new_pair, quantile=q, reference=ref, ref_val=cur_ref_val)
                )


        # C) memory -> UI (push only when different)
        @reactive.Effect
        def _memory_to_ui():
            prop, ftype, q, ref, ref_val = _current_context(threshold_id)
            if not (prop and ftype):
                return

            mem = thresholding_memory.get()

            # Avoid establishing reactive deps on inputs here
            with reactive.isolate():
                current_slider = input[f"threshold_slider_{threshold_id}"]()
                try:
                    cur_lo = input[f"floor_threshold_value_{threshold_id}"]()
                except Exception:
                    cur_lo = None
                try:
                    cur_hi = input[f"roof_threshold_value_{threshold_id}"]()
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

            req(current_val is not None)

            # Prefer stored; if missing, keep whatever user currently has (don't overwrite)
            effective_ref = stored_ref if isinstance(stored_ref, (int, float)) else current_val

            if not (isinstance(vals, (tuple, list)) and len(vals) == 2 and all(v is not None for v in vals)):
                return

            # Push to slider if needed
            if not _nearly_equal_pair(vals, current_slider if isinstance(current_slider, (tuple, list)) and len(current_slider) == 2 else (None, None)):
                ui.update_slider(f"threshold_slider_{threshold_id}", value=tuple(vals))

            # Push to numerics if needed
            if not (isinstance(cur_lo, (int, float)) and abs(float(cur_lo) - float(vals[0])) <= EPS):
                ui.update_numeric(f"floor_threshold_value_{threshold_id}", value=float(vals[0]))
            if not (isinstance(cur_hi, (int, float)) and abs(float(cur_hi) - float(vals[1])) <= EPS):
                ui.update_numeric(f"roof_threshold_value_{threshold_id}", value=float(vals[1]))
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
                ref_val = None

            mem = thresholding_memory.get()

            # Use current slider pair as the values to keep them in sync; if missing, default to (0,0)
            current_slider = input[f"threshold_slider_{threshold_id}"]()
            if isinstance(current_slider, (tuple, list)) and len(current_slider) == 2:
                pair = (float(current_slider[0]), float(current_slider[1]))
            else:
                pair = (0.0, 0.0)

            new_mem = _set_threshold_memory(
                mem.copy(), threshold_id, prop, ftype, pair,
                quantile=q, reference=ref, ref_val=ref_val
            )
            thresholding_memory.set(new_mem)


        
            
    




    # ======================= 2D THRESHOLDING (LASSO) =======================

    # Memory: selected original-row indices for each 2D threshold and metric pair
    # Shape: { tid: { (propX, propY): set([...original row indices...]) } }
    thresholding_memory_2d_selection = reactive.Value({})

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

    def _get_series(prop: str, spot_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
        """Return a compact 2-col frame [Track ID, prop] from the right source."""
        # if prop in Metrics.Thresholding.TrackProperties and not track_df.empty:
        if prop in Metrics.Thresholding.TrackProperties:
            return Threshold.Normalize_01(track_df, prop)
        # if prop in Metrics.Thresholding.SpotProperties and not spot_df.empty:
        if prop in Metrics.Thresholding.SpotProperties:
            return Threshold.Normalize_01(spot_df, prop)
        return pd.DataFrame(columns=[prop]).set_index(pd.Index([], name='Track UID'))


    def _xy_for_2d_threshold(threshold_id: int, spot_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build XY for *this* block, restricted by prior brushes.
        Prior brushes are stored as sets of Track IDs in `thresholding_memory_2d_selection`.
        """
        
        # 2) Build XY for THIS block and restrict to previous intersection (if any)
        propX = input[f"thresholding_metric_X_{threshold_id}"]()
        propY = input[f"thresholding_metric_Y_{threshold_id}"]()

        if not (propX and propY):
            return pd.DataFrame(columns=[propX, propY]).set_index(pd.Index([], name='Track UID'))
        
        xy_cur = Threshold.JoinByIndex(
            _get_series(propX, spot_df, track_df), 
            _get_series(propY, spot_df, track_df)
        )

        req(not xy_cur.empty)
        return xy_cur


    @Debounce(1)
    @reactive.Calc
    def get_array_size():
        return input.threshold2d_array_size() if input.threshold2d_array_size() is not None or input.threshold2d_array_size() != 0 else 1
        
    def render_threshold2d_widget(threshold_id: int):
        @output(id=f"threshold2d_plot_{threshold_id}")
        @render_widget
        def threshold2d_plot():  # id must match output_widget id
            
            t_state = thresholds2d_state.get()
            req(t_state is not None and isinstance(t_state, dict))
            
            try:
                current_state = t_state.get(threshold_id)
            except Exception:
                current_state = None

            req(
                isinstance(current_state, dict)
                and isinstance(current_state["spots"], pd.DataFrame)
                and isinstance(current_state["tracks"], pd.DataFrame)
            )

            spot_df, track_df = current_state.get("spots"), current_state.get("tracks")
            req(not spot_df.empty and not track_df.empty)
            
            propX = input[f"thresholding_metric_X_{threshold_id}"]()
            propY = input[f"thresholding_metric_Y_{threshold_id}"]()
            req(propX and propY)

            df = _xy_for_2d_threshold(threshold_id, spot_df, track_df)
            req(not df.empty)

            if t_state.get(threshold_id + 1) is None:
                t_state[threshold_id + 1] = {"spots": spot_df, "tracks": track_df}
                thresholds2d_state.set(t_state)

            X = df[propX].to_numpy()
            Y = df[propY].to_numpy()

            # ---- density colors for each point -----------------------------------------
            # Use same range as the axes (you set [-0.025, 1.025])
            xmin, xmax = -0.025, 1.025
            ymin, ymax = -0.025, 1.025

            # Choose grid resolution ~ screen pixels; tune for speed/quality
            nx, ny = 300, 300

            # 2D histogram → H[y, x] counts and bin edges
            H, xedges, yedges = np.histogram2d(
                X, Y, bins=(nx, ny), range=[[xmin, xmax], [ymin, ymax]]
            )

            # For each point, find its (xbin, ybin) → count
            ix = np.clip(np.searchsorted(xedges, X, side="right") - 1, 0, nx - 1)
            iy = np.clip(np.searchsorted(yedges, Y, side="right") - 1, 0, ny - 1)
            dens = H[ix, iy]

            # Optional log scale helps dynamic range
            dens = np.log10(dens + 1.0)

            # Normalize 0..1 for colorscale
            dmin, dmax = dens.min(), dens.max()
            if dmax > dmin:
                dens_norm = (dens - dmin) / (dmax - dmin)
            else:
                dens_norm = np.zeros_like(dens)

            # FlowJo-like blue→cyan→green→yellow→red
            FLOWJO = [
                [0.00, "rgb(0,0,130)"],
                [0.25, "rgb(0,180,255)"],
                [0.50, "rgb(0,200,0)"],
                [0.75, "rgb(255,255,0)"],
                [1.00, "rgb(255,0,0)"],
            ]
            # ---------------------------------------------------------------------------

            # Recover previously selected points (your existing code)
            mem = thresholding_memory_2d_selection.get()
            selected_set = _get_2d_selected_set(mem, threshold_id, propX, propY)
            row_index = df.index.to_numpy()
            selectedpoints = np.nonzero(np.isin(row_index, list(selected_set)))[0].tolist()

            w = go.FigureWidget(
                data=[
                    go.Scattergl(
                        x=X, y=Y, mode="markers",
                        marker=dict(
                            size=get_array_size(),
                            color=dens_norm,
                            colorscale=FLOWJO,
                            cmin=0, cmax=1,
                            showscale=False
                        ),
                        selected=dict(marker=dict(opacity=1.0)),
                        unselected=dict(marker=dict(opacity=0.25)),
                        selectedpoints=selectedpoints,
                        hoverinfo="skip",
                    )
                ],
                layout=go.Layout(
                    autosize=True,
                    height=225,
                    width=150,
                    margin=dict(l=0, r=5, t=60, b=0),
                    xaxis=dict(
                        range=[-0.025, 1.025],
                        scaleanchor="y",          # lock x to y → 1:1 aspect
                        constrain="domain",       # keep axes inside plotting area
                        showgrid=False,
                        tickvals=[0, 0.5, 1], 
                        ticktext=["", "", ""],
                        ticks="outside",
                        ticklen=3,
                        minor=dict(
                            ticks="outside", 
                            ticklen=2,
                            tick0=0,
                            tickvals=[0.25, 0.75],
                            showgrid=False
                        ),
                        title=None, 
                        zeroline=False
                    ),
                    yaxis=dict(
                        range=[-0.025, 1.025],
                        constrain="domain",
                        showgrid=False, 
                        tickvals=[0, 0.5, 1], 
                        ticktext=["", "", ""],
                        ticks="outside",
                        ticklen=3,
                        minor=dict(
                            ticks="outside", 
                            ticklen=2,
                            tick0=0,
                            tickvals=[0.25, 0.75],
                            showgrid=False
                        ),
                        title=None, 
                        zeroline=False,
                        showline=True,
                    ),
                    dragmode="select",
                    paper_bgcolor="#f7f7f7",
                    plot_bgcolor="white",
                    shapes=[
                        # bottom x-axis line
                        dict(
                            type="line",
                            xref="x", yref="y",
                            x0=-0.025, y0=-0.025, x1=1.025, y1=-0.025,
                            line=dict(width=1)
                        ),
                        # left y-axis line
                        dict(
                            type="line",
                            xref="x", yref="y",
                            x0=-0.025, y0=-0.025, x1=-0.025, y1=1.025,
                            line=dict(width=1)
                        ),
                    ],
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
                    return

                else:
                    # Map trace positions -> original row indices
                    sel_rows = set(w._row_index[np.array(inds, dtype=int)])

                    cur = thresholding_memory_2d_selection.get()
                    new_mem = _set_2d_selected_set(cur, w._tid, w._propX, w._propY, sel_rows)
                    thresholding_memory_2d_selection.set(new_mem)

                    spots_filtered = spot_df.loc[spot_df.index.intersection(sel_rows)]
                    tracks_filtered = track_df.loc[track_df.index.intersection(sel_rows)]

                    for tid, i in enumerate(threshold_list.get(), start=threshold_id + 1):
                        t_state[tid].update({
                            "spots": spots_filtered,
                            "tracks": tracks_filtered
                        })
                    
                    thresholds2d_state.set(t_state)

            w.data[0].on_selection(_on_selection)  # uses Plotly FigureWidget API

            return w

        # wire render function to specific id
        threshold2d_plot._id = f"threshold2d_plot_{threshold_id}"
        

    def _clear_2d_selection_for_id(threshold_id: int):
        @reactive.Effect
        @reactive.event(input[f"threshold2d_clear_{threshold_id}"])
        def clear_2d_selection():
            
            for tid, i in enumerate(threshold_list.get(), start=threshold_id):
                if tid not in threshold_list.get():
                    break

                state = thresholds2d_state.get()
                if state is None or not isinstance(state, dict):
                    return None
                
                current_state = state.get(tid)
                clear_state = state.get(tid + 1)
                req(current_state is not None and isinstance(current_state, dict))
                req(clear_state is not None and isinstance(clear_state, dict))

                req(isinstance(current_state.get("spots"), pd.DataFrame) and isinstance(current_state.get("tracks"), pd.DataFrame))
                spot_df, track_df = current_state.get("spots"), current_state.get("tracks")
                req(not spot_df.empty and not track_df.empty)

                state[tid + 1].update({
                    "spots": spot_df,
                    "tracks": track_df
                })
                thresholds2d_state.set(state)

                selection_memory = thresholding_memory_2d_selection.get()
                if selection_memory.get(tid) is not None and isinstance(selection_memory.get(tid), dict):
                    selection_memory[tid] = set()
                    thresholding_memory_2d_selection.set(selection_memory)

                render_threshold2d_widget(tid)


    @reactive.Effect
    def _():
        for threshold_id in threshold_list.get():
            _clear_2d_selection_for_id(threshold_id)
            render_threshold2d_widget(threshold_id)



    # - - - - Threshold modules management - - - -

    # REMOVED the original first set_threshold_modules() that read a non-existent
    # manual input and caused feedback loops. Its behavior is replaced by the
    # per-threshold sync registered below.

    @reactive.Effect
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
                render_threshold2d_widget(threshold_id)
                # register_threshold2d_clear(threshold_id)





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
            # if reference == "My own value":z
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
            # Update the numeric input for "My own value"
            if select == "My own value":
                my_own_value = input[f"my_own_value_{threshold_id}"]()
                ui.update_numeric(
                    id=f"my_own_value_{threshold_id}",
                    value=my_own_value if isinstance(my_own_value, (int, float)) else None
                )


    # - - - - - - - - - - - - - - - - - - - - -




    # - - - - Passing filtered data to the app - - - -

    @reactive.Effect
    @reactive.event(input.filter_data)
    def pass_filtered_data_to_app():

        print("Passing filtered data to the app...")

        if threshold_dimension.get() == "1D":
            t_state = thresholds1d_state.get()
        elif threshold_dimension.get() == "2D":
            t_state = thresholds2d_state.get()
        req(t_state is not None and isinstance(t_state, dict))


        try:
            latest_state = t_state.get(list(t_state.keys())[-1])
            print("Latest state retrieved.")
        except Exception:
            latest_state = None

        spots_filtered = pd.DataFrame(latest_state.get("spots") if latest_state is not None and isinstance(latest_state, dict) else UNFILTERED_SPOTSTATS.get())
        tracks_filtered = pd.DataFrame(latest_state.get("tracks") if latest_state is not None and isinstance(latest_state, dict) else UNFILTERED_TRACKSTATS.get())
        frame_stats = UNFILTERED_FRAMESTATS.get()

        print(f"Filtered tracks: {len(tracks_filtered)}")

        SPOTSTATS.set(spots_filtered)
        TRACKSTATS.set(tracks_filtered)
        FRAMESTATS.set(Calc.Frames(spots_filtered) if spots_filtered is not None and not spots_filtered.empty else frame_stats)
    



    

    # - - - - Rendering Data Frames - - - -
    
    @render.data_frame
    def render_spot_stats():
        spot_stats = SPOTSTATS.get()
        if spot_stats is not None and not spot_stats.empty:
            return spot_stats
        else:
            pass

    @render.data_frame
    def render_track_stats():
        track_stats = TRACKSTATS.get()
        if track_stats is not None and not track_stats.empty:
            return track_stats
        else:
            pass

    @render.data_frame
    def render_frame_stats():
        frame_stats = FRAMESTATS.get()
        if frame_stats is not None and not frame_stats.empty:
            return frame_stats
        else:
            pass
    

    # - - - - DataFrame Downloads - - - -

    @render.download(filename=f"Spot stats {date.today()}.csv")
    def download_spot_stats():
        spot_stats = SPOTSTATS.get()
        if spot_stats is not None and not spot_stats.empty:
            with io.BytesIO() as buffer:
                spot_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass

    @render.download(filename=f"Track stats {date.today()}.csv")
    def download_track_stats():
        track_stats = TRACKSTATS.get()
        if track_stats is not None and not track_stats.empty:
            with io.BytesIO() as buffer:
                track_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass
    
    @render.download(filename=f"Frame stats {date.today()}.csv")
    def download_frame_stats():
        frame_stats = FRAMESTATS.get()
        if frame_stats is not None and not frame_stats.empty:
            with io.BytesIO() as buffer:
                frame_stats.to_csv(buffer, index=False)
                yield buffer.getvalue()
        else:
            pass




    # ======================= DATA VISUALIZATION =======================


    # - - - - - - Swarmplot - - - - - -    

    @ui.bind_task_button(button_id="sp_generate")
    @reactive.extended_task
    async def output_swarmplot(
        df,
        metric,
        palette,
        show_swarm,
        swarm_size,
        swarm_outline_color,
        swarm_alpha,
        show_violin,
        violin_fill_color,
        violin_edge_color,
        violin_alpha,
        violin_outline_width,
        show_mean,
        mean_span,
        mean_color,
        show_median,
        median_span,
        median_color,
        line_width,
        set_main_line,
        show_error_bars,
        errorbar_capsize,
        errorbar_color,
        errorbar_lw,
        errorbar_alpha,
        show_mean_balls,
        mean_ball_size,
        mean_ball_outline_color,
        mean_ball_outline_width,
        mean_ball_alpha,
        show_median_balls,
        median_ball_size,
        median_ball_outline_color,
        median_ball_outline_width,
        median_ball_alpha,
        show_kde,
        kde_inset_width,
        kde_outline,
        kde_alpha,
        kde_legend,
        kde_fill,
        p_test,
        show_legend,
        show_grid,
        open_spine,
    ):
        # run sync plotting off the event loop
        def build():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Starting a Matplotlib GUI outside of the main thread will likely fail",
                    category=UserWarning,
                )

                local_df = df.copy(deep=True) if df is not None else pd.DataFrame()
                return Plot.Superplots.SwarmPlot(
                    df=local_df,
                    metric=metric,
                    palette=palette,
                    show_swarm=show_swarm,
                    swarm_size=swarm_size,
                    swarm_outline_color=swarm_outline_color,
                    swarm_alpha=swarm_alpha,
                    show_violin=show_violin,
                    violin_fill_color=violin_fill_color,
                    violin_edge_color=violin_edge_color,
                    violin_alpha=violin_alpha,
                    violin_outline_width=violin_outline_width,
                    show_mean=show_mean,
                    mean_span=mean_span,
                    mean_color=mean_color,
                    show_median=show_median,
                    median_span=median_span,
                    median_color=median_color,
                    line_width=line_width,
                    set_main_line=set_main_line,
                    show_error_bars=show_error_bars,
                    errorbar_capsize=errorbar_capsize,
                    errorbar_color=errorbar_color,
                    errorbar_lw=errorbar_lw,
                    errorbar_alpha=errorbar_alpha,
                    show_mean_balls=show_mean_balls,
                    mean_ball_size=mean_ball_size,
                    mean_ball_outline_color=mean_ball_outline_color,
                    mean_ball_outline_width=mean_ball_outline_width,
                    mean_ball_alpha=mean_ball_alpha,
                    show_median_balls=show_median_balls,
                    median_ball_size=median_ball_size,
                    median_ball_outline_color=median_ball_outline_color,
                    median_ball_outline_width=median_ball_outline_width,
                    median_ball_alpha=median_ball_alpha,
                    show_kde=show_kde,
                    kde_inset_width=kde_inset_width,
                    kde_outline=kde_outline,
                    kde_alpha=kde_alpha,
                    kde_legend=kde_legend,
                    kde_fill=kde_fill,
                    p_test=p_test,
                    show_legend=show_legend,
                    show_grid=show_grid,
                    open_spine=open_spine,
                )

        # Either form is fine; pick one:
        # return await asyncio.get_running_loop().run_in_executor(None, build)
        return await asyncio.to_thread(build)
    

    @reactive.effect
    @reactive.event(input.sp_generate, ignore_none=False)
    def trigger_swarmplot():
        output_swarmplot.cancel()

        output_swarmplot(
            df=TRACKSTATS.get() if TRACKSTATS.get() is not None else pd.DataFrame(),
            metric=input.sp_metric(),
            palette=input.sp_palette(),

            show_swarm=input.sp_show_swarms(),
            swarm_size=input.sp_swarm_marker_size(),
            swarm_outline_color=input.sp_swarm_marker_outline(),
            swarm_alpha=input.sp_swarm_marker_alpha() if 0.0 <= input.sp_swarm_marker_alpha() <= 1.0 else 1.0,

            show_violin=input.sp_show_violins(),
            violin_fill_color=input.sp_violin_fill(),
            violin_edge_color=input.sp_violin_outline(),
            violin_alpha=input.sp_violin_alpha() if 0.0 <= input.sp_violin_alpha() <= 1.0 else 1.0,
            violin_outline_width=input.sp_violin_outline_width(),

            show_mean=input.sp_show_cond_mean(),
            mean_span=input.sp_mean_line_span(),
            mean_color=input.sp_mean_line_color(),
            show_median=input.sp_show_cond_median(),
            median_span=input.sp_median_line_span(),
            median_color=input.sp_median_line_color(),
            line_width=input.sp_lines_lw(),
            set_main_line=input.sp_set_as_primary(),
            show_error_bars=input.sp_show_errbars(),
            errorbar_capsize=input.sp_errorbar_capsize(),
            errorbar_color=input.sp_errorbar_color(),
            errorbar_lw=input.sp_errorbar_lw(),
            errorbar_alpha=input.sp_errorbar_alpha() if 0.0 <= input.sp_errorbar_alpha() <= 1.0 else 1.0,

            show_mean_balls=input.sp_show_rep_means(),
            mean_ball_size=input.sp_mean_bullet_size(),
            mean_ball_outline_color=input.sp_mean_bullet_outline(),
            mean_ball_outline_width=input.sp_mean_bullet_outline_width(),
            mean_ball_alpha=input.sp_mean_bullet_alpha() if 0.0 <= input.sp_mean_bullet_alpha() <= 1.0 else 1.0,
            show_median_balls=input.sp_show_rep_medians(),
            median_ball_size=input.sp_median_bullet_size(),
            median_ball_outline_color=input.sp_median_bullet_outline(),
            median_ball_outline_width=input.sp_median_bullet_outline_width(),
            median_ball_alpha=input.sp_median_bullet_alpha() if 0.0 <= input.sp_median_bullet_alpha() <= 1.0 else 1.0,

            show_kde=input.sp_show_kde(),
            kde_inset_width=input.sp_kde_bandwidth(),
            kde_outline=input.sp_kde_line_width(),
            kde_alpha=input.sp_kde_fill_alpha() if 0.0 <= input.sp_kde_fill_alpha() <= 1.0 else 1.0,
            kde_legend=input.sp_kde_legend(),
            kde_fill=input.sp_kde_fill(),

            p_test=False,
            show_legend=input.sp_show_legend(),
            show_grid=input.sp_grid(),
            open_spine=input.sp_spine()
        )

    # @output(id="swarmplot")
    @render.plot
    def swarmplot():
        # Only update when output_swarmplot task completes (not reactively)
        return output_swarmplot.result()



    @render.download(filename=f"Swarmplot {date.today()}.svg")
    def download_swarmplot_svg():
        track_df = TRACKSTATS.get() if TRACKSTATS.get() is not None else pd.DataFrame()
        fig = Plot.Superplots.SwarmPlot(
            df=track_df,
            metric=input.sp_metric(),
            palette=input.sp_palette(),

            show_swarm=input.sp_show_swarms(),
            swarm_size=input.sp_swarm_marker_size(),
            swarm_outline_color=input.sp_swarm_marker_outline(),
            swarm_alpha=input.sp_swarm_marker_alpha(),

            show_violin=input.sp_show_violins(),
            violin_fill_color=input.sp_violin_fill(),
            violin_edge_color=input.sp_violin_outline(),
            violin_alpha=input.sp_violin_alpha(),
            violin_outline_width=input.sp_violin_outline_width(),

            show_mean=input.sp_show_cond_mean(),
            mean_span=input.sp_mean_line_span(),
            mean_color=input.sp_mean_line_color(),
            show_median=input.sp_show_cond_median(),
            median_span=input.sp_median_line_span(),
            median_color=input.sp_median_line_color(),
            line_width=input.sp_lines_lw(),
            set_main_line=input.sp_set_as_primary(),
            show_error_bars=input.sp_show_errbars(),
            errorbar_capsize=input.sp_errorbar_capsize(),
            errorbar_color=input.sp_errorbar_color(),
            errorbar_lw=input.sp_errorbar_lw(),
            errorbar_alpha=input.sp_errorbar_alpha(),

            show_mean_balls=input.sp_show_rep_means(),
            mean_ball_size=input.sp_mean_bullet_size(),
            mean_ball_outline_color=input.sp_mean_bullet_outline(),
            mean_ball_outline_width=input.sp_mean_bullet_outline_width(),
            mean_ball_alpha=input.sp_mean_bullet_alpha(),
            show_median_balls=input.sp_show_rep_medians(),
            median_ball_size=input.sp_median_bullet_size(),
            median_ball_outline_color=input.sp_median_bullet_outline(),
            median_ball_outline_width=input.sp_median_bullet_outline_width(),
            median_ball_alpha=input.sp_median_bullet_alpha(),

            show_kde=input.sp_show_kde(),
            kde_inset_width=input.sp_kde_bandwidth(),
            kde_outline=input.sp_kde_line_width(),
            kde_alpha=input.sp_kde_fill_alpha(),
            kde_legend=input.sp_kde_legend(),
            kde_fill=input.sp_kde_fill(),

            p_test=False,
            show_legend=input.sp_show_legend(),
            show_grid=input.sp_grid(),
            open_spine=input.sp_spine()
        )
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="svg", bbox_inches='tight')
                yield buffer.getvalue()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # - - - - - - Initialization progress - - - - - -

    @reactive.extended_task
    async def loader2():
        with ui.Progress(min=0, max=20) as p:
            p.set(message="Initialization in progress")

            for i in range(1, 12):
                p.set(i, message="Initializing Peregrin...")
                await asyncio.sleep(0.12)
        pass

    @reactive.effect
    @reactive.event(input.already_processed_input, ignore_none=True)
    def initialize_loader2():
        return loader2()





    # (Other outputs and logic remain unchanged...)

# --- Mount the app ---
app = App(app_ui, server)


# TODO - Remodel the sidebar: 
#          - Only contain "1D filtering" 
#          - add a button that takes the user to the 2D filtering page
# TODO - Upgrade 2D filtering:
#          - Make it accessible trough the nav_bar panel "Gating"
#          - Logic - "1D" filtering as primary filtering, its output goes to "2D" filtering = "Gating"



# TODO - Track visualization plot with a slider

# TODO - define pre-sets for plot settings, so that the user can try out different looks easily
# TODO - Add a button to reset all thresholding settings to default

# TODO - Keep all the raw data (columns) - rather format them (stripping of _ and have them not all caps)
# TODO - Make the 2D filtering logic work on the same logic as does the D filtering logic
# TODO - Make it possible to save/load threshold configurations
# TODO - Find a way to program all the functions so that functions do not refresh/re-render unnecessarily on just any reactive action
# TODO - Time point definition
# TODO - Make it possible for the user to title their charts
# TODO - Mean directional change rate
# TODO - Select which p-tests should be shown in the superplot chart
# TODO - P-test
# TODO - Again add rendered text showing the total number of cells in the input and the number of output cells
# TODO - Option to download a simple legend showing how much data was filtered out and how so
# TODO - input_selectize("Plot:"... with options "Polar/Normalized" or "Cartesian/Raw"
# TODO - Differentiate between frame(s) annotations and time annotations


# TODO - add documentation into the markdown file with a pdf file user guide