from shiny import reactive
from shiny.express import input, render, ui
from shiny.types import FileInfo
from shinywidgets import render_plotly, render_altair

import asyncio
import io
import os.path as op

from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import utils.funcs_data as du
import utils.funcs_plot as pu
import utils.select_markers as select_markers
import utils.select_modes as select_mode
import utils.select_metrics as select_metrics
from utils.ratelimit import debounce, throttle

import webbrowser
import tempfile


pd.options.mode.chained_assignment = None



# ===========================================================================================================================================================================================================================================================================
# ===========================================================================================================================================================================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reactive and global variables


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating reactive values for the data input

raw_Buttered_df = reactive.value()      # Creating a reactive value for the stem/pre-processed base dataframe
raw_Spot_stats_df = reactive.value()    # Creating a reactive value for the stem/pre-processed spot stats dataframe
raw_Track_stats_df = reactive.value()   # Creating a reactive value for the stem/pre-processed track stats file dataframe
raw_Time_stats_df = reactive.value()    # Creating a reactive value for the stem/pre-processed time stats file dataframe


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating reactive values for thresholding the data

Spot_stats_df_T = reactive.value()      # Creating a reactive value for the mediating spot stats file used in thresholding
Track_stats_df_T = reactive.value()     # Creating a reactive value for the mediating track stats file used in thresholding


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating reactive variables for processed dataframe storage

Buttered_df = reactive.value()          # Creating a reactive value for the processed base dataframe
Spot_stats_df = reactive.value()        # Creating a reactive value for the processed spot stats file
Track_stats_df = reactive.value()       # Creating a reactive value for the processed track stats file
Time_stats_df = reactive.value()        # Creating a reactive value for the processed time stats file


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating other reactive variables 

Track_metrics = reactive.value()        # Creating a reactive value for the track metrics
Spot_metrics = reactive.value()         # Creating a reactive value for the spot metrics

slider_valuesT1 = reactive.value()      # Creating a reactive value for the slider values for thresholding 1
slider_valuesT2 = reactive.value()      # Creating a reactive value for the slider values for thresholding 2
slider_values = reactive.value()        # Creating a reactive value for the slider values

count = reactive.value(1)               # Data input counter
conditions = reactive.value()           # Creating a reactive value for the conditions


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating reactive values for the data input status

file_detected = reactive.value(False)
delayed_detection = reactive.value(False)
cells_in_possesion = reactive.value(False)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Optics parameters

# Definition of micron length per pixel
microns_per_pixel = 0.7381885238402274 # for 10x lens

# Define the desired dimensions in microns
x_min, x_max = 0, (1600 * microns_per_pixel)
y_min, y_max = 0, (1200 * microns_per_pixel)
x_axe_remainder = x_max-1150
x_add = 50 - x_axe_remainder
y_ax_remainder = y_max-850
x_substract = (x_max - y_max) + (y_ax_remainder - 50)

# Calculate the aspect ratio
aspect_ratio = x_max / y_max



# plot specs
title_size = 16
title_size2 = 12
label_size = 11
figtext_size = 9
compass_annotations_size = 15
figtext_color = 'grey'

# Color maps
cmap_cells = mcolors.LinearSegmentedColormap.from_list("", ["#9b598910", "#9b181eff"])
cmap_frames = plt.get_cmap('viridis')



smoothing_index = reactive.value(0)
arrow_size = reactive.value(6)
line_width = reactive.value(1)
marker_size = reactive.value(5)

dir = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating a reactive value for visualized tracks hover info
see_hover = reactive.value(['CONDITION', 'REPLICATE', 'TRACK_ID'])        # Creating a reactive value for the visualized tracks hover info





# ===========================================================================================================================================================================================================================================================================
# ===========================================================================================================================================================================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the app layout

ui.page_opts(
    title="Peregrin", 
    fillable=True
    )




# ===========================================================================================================================================================================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Data input panel
# 
# 1. Rendering a default data input slot (CSV file browser and condition labeling text window)
# 2. Adding and removing additional data input slots (CSV file browser and condition labeling text window)


with ui.nav_panel("Input"):
    
    with ui.div(id="data-inputs"):      # div container for flow content

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Buttons for adding and removing additional data inputs

        ui.input_action_button("add_input", "Add data input", class_="btn btn-primary")
        ui.input_action_button("remove_input", "Remove data input", class_="btn btn-primary")

        ui.markdown(
            """
            ___
            """
        )

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Default data input slot

        @render.ui
        def default_input():
            default_browser = ui.input_file("file1", "Input CSV", accept=[".csv"], multiple=True, placeholder="No files selected")
            default_label = ui.input_text("label1", "Condition", placeholder="label me :D")
            return default_label, default_browser


        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Reactive event for adding and removing additional data input slots

        @reactive.effect                                        # Reactive effect on "Add data input"
        @reactive.event(input.add_input)                        # "Add data input" button sensor
        def add_rowser():
            if input.add_input():                               # REACTION:
                count.set(count.get() + 1)                      # Increasing the input count
                adding = count.get()                            # Getting the current input count

                segmentator = ui.markdown(
                    """
                    <hr style="border: none; border-top: 1px dotted" />
                    """
                    )

                # CSV file browser
                browser = ui.input_file(                        
                    id=f"file{adding}", 
                    label=f"Input CSV {adding}", 
                    accept=[".csv"], 
                    multiple=True, 
                    placeholder="No files selected"
                    )
                
                # Data labeling text window (condition labeling)
                label = ui.input_text(                          
                    id=f"label{adding}", 
                    label=f"Condition", 
                    placeholder="label me :D"
                    )

                # Container rendering the additional input slot container
                ui.insert_ui(                                   
                    ui.div(                                     
                        {"id": f"additional-input-{adding}"}, 
                        segmentator, label, browser),
                        selector="#data-inputs",
                        where="beforeEnd",
                )

        @reactive.effect                                        # Reactive effect on "Remove data input"
        @reactive.event(input.remove_input)                     # "Remove data input" button sensor
        def remove_browser():
            if input.remove_input():                            # REACTION:
                removing = count.get()                          # Getting the current input count
                ui.remove_ui(f"#additional-input-{removing}")   # Removing the last input slot (one with the current input count)
                if count.get() > 1:                             # Decreasing the input count
                    count.set(removing - 1)                     
                else:
                    pass





    # =======================================================================================================================================================================================================================================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Data input parsing and processing function
    # 
    # 1. Loading the data from the CSV files
    # 2. Buttering the data (smoothing)
    # 3. Assigning the condition and replicate labels to the data
    # 4. Merging the data into a single DataFrame


    @reactive.calc 
    def parsed_file():                                                           
        
        default = pd.DataFrame()                                                        
        additional = pd.DataFrame()
                    
        inpt_file_list_dflt: list[FileInfo] | None = input.file1()                      # Getting the list of default input data files

        if inpt_file_list_dflt is None:
            default = pd.DataFrame()
        
        else:
            all_data_dflt = []
            for dflt_file_count, file_dflt in enumerate(inpt_file_list_dflt, start=1):  # Enumerate and cycle through the files
                df_dflt = pd.read_csv(file_dflt['datapath'])                            # Load each CSV file into a DataFrame
                buttered_dflt = du.butter(df_dflt)                                      # Butter the DataFrame
                                                  
                label_dflt = input.label1()                                             # Assigning the condition label to a 'CONDITION' column
                if not label_dflt or label_dflt is None:                                # If no label is provided, assign a default - numeric - one
                    buttered_dflt['CONDITION'] = 1
                else:                                                                   # Else, assign the lable
                    buttered_dflt['CONDITION'] = f'{label_dflt}'
                buttered_dflt['REPLICATE'] = dflt_file_count                            # Assigning the replicate number

                all_data_dflt.append(buttered_dflt)                                     # Stack the buttered and labeled DataFrames into a list

                default = pd.concat(all_data_dflt, axis=0)                              # Merge the DataFrames
                

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Processing the additional input files

        browse_count = count.get()                                                      # Getting the current additional input slot count
        all_data_addtnl = []                         
        for i in range(2, browse_count + 1, 1):                                         # Cycle trough the additional input slots

            inpt_file_list_addtnl: list[FileInfo] | None = input[f"file{i}"]()          # Getting the list of additional input data files

            if inpt_file_list_addtnl is None:
                additional = pd.DataFrame()
            
            else:
                for additnl_file_count, file_addtnl in enumerate(inpt_file_list_addtnl, start=1):   # Enumerate and cycle through additional input files
                    df_addtnl = pd.read_csv(file_addtnl["datapath"])                  
                    buttered_addtnl = du.butter(df_addtnl)

                    label_addtnl = input[f"label{i}"]()                                 # Assigning the condition label to a 'CONDITION' column
                    if not label_addtnl or label_addtnl is None:                        # If no label is provided, assign a default - numeric - one
                        buttered_addtnl['CONDITION'] = i
                    else:                                                               # Else, assign the given lable
                        buttered_addtnl['CONDITION'] = f'{label_addtnl}'
                    buttered_addtnl['REPLICATE'] = additnl_file_count                   # Assigning the replicate number

                    all_data_addtnl.append(buttered_addtnl)                             # Stack the buttered and labeled DataFrames into a list
                    

                    additional = pd.concat(all_data_addtnl, axis=0)                     # Merge the DataFrames


        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Merging the default and additional input files

        return pd.DataFrame(pd.concat([default, additional], axis=0))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating a reactive effect detecting any files is selected

@reactive.effect
def file_detection():
    for i in range(1, count.get() + 1, 1):
        if input[f"file{i}"]() != None:
            cells_in_possesion.set(True)
            file_detected.set(True)
        else:
            pass
        
            

            
 

# ===========================================================================================================================================================================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Data processing and showcasing panel
# 
# 1. Processing and calculating the data (spot stats, track stats, time stats)
# 2. Displaying the dataframes
# 3. Enabling the user to download the dataframes as .csv files


with ui.nav_panel("Data frames"):  


    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @reactive.effect
    def update_buttered_df():
        if file_detected.get() == False:
            return pd.DataFrame()
        
        df = parsed_file()
        raw_Buttered_df.set(df)


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @reactive.calc
    def process_spot_data():
        if file_detected.get() == False:
            return pd.DataFrame()
        
        buttered = raw_Buttered_df.get()

        distances_for_each_cell_per_frame_df = du.calculate_traveled_distances_for_each_cell_per_frame(buttered)        # Call the function to calculate distances for each cell per frame and create the Spot_statistics .csv file
        distances_for_each_cell_per_frame_df = du.calculate_track_length_net_distances_and_confinement_ratios_per_each_cell_per_frame(distances_for_each_cell_per_frame_df)
        direction_for_each_cell_per_frame_df = du.calculate_direction_of_travel_for_each_cell_per_frame(buttered)       # Call the function to calculate direction_for_each_cell_per_frame_df

        Spot_stats_dfs = [buttered, distances_for_each_cell_per_frame_df, direction_for_each_cell_per_frame_df]
        Spot_stats = du.merge_dfs(Spot_stats_dfs, on=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T']) # Merge the dataframes

        return Spot_stats


    @reactive.effect
    def update_Spot_stats_df():
        if file_detected.get() == False:
            return pd.DataFrame()
        
        else:
            Spot_stats = process_spot_data()
            raw_Spot_stats_df.set(Spot_stats)
            Spot_metrics.set(Spot_stats.columns)
        
        Spot_stats = process_spot_data()
        raw_Spot_stats_df.set(Spot_stats)
        Spot_metrics.set(Spot_stats.columns)


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @reactive.calc
    def process_track_data():
        if file_detected.get() == False:
            return pd.DataFrame()
        
        Spot_stats = raw_Spot_stats_df.get()

        if Spot_stats.empty:
            return pd.DataFrame()

        tracks_lengths_and_net_distances_df = du.calculate_track_lengths_and_net_distances(Spot_stats) # Calling function to calculate the total distance traveled for each cell from the distances_for_each_cell_per_frame_df
        confinement_ratios_df = du.calculate_confinement_ratio_for_each_cell(tracks_lengths_and_net_distances_df) # Call the function to calculate confinement ratios from the Track_statistics1_df and write it into the Track_statistics1_df
        track_directions_df = du.calculate_absolute_directions_per_cell(Spot_stats) # Call the function to calculate directions_per_cell_df
        frames_per_track = du.calculate_number_of_frames_per_cell(Spot_stats)
        speeds_per_cell = du.calculate_speed(Spot_stats, ['REPLICATE', 'TRACK_ID'])

        Track_stats_dfs = [tracks_lengths_and_net_distances_df, confinement_ratios_df, track_directions_df, frames_per_track, speeds_per_cell]
        Track_stats = du.merge_dfs(Track_stats_dfs, on=['CONDITION', 'REPLICATE', 'TRACK_ID'])

        Track_stats = Track_stats.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID'])

        return Track_stats
    

    @reactive.effect
    def update_Track_stats_df():
        if file_detected.get() == False:
            return pd.DataFrame()
        
        else:
            Track_stats = process_track_data()
            raw_Track_stats_df.set(Track_stats)
            Track_metrics.set(Track_stats.columns)

        Track_stats = process_track_data()
        raw_Track_stats_df.set(Track_stats)
        Track_metrics.set(Track_stats.columns)


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @reactive.calc
    def process_time_data():
        if file_detected.get() == False:
            return pd.DataFrame()

        Spot_stats = Spot_stats_df.get()

        if Spot_stats.empty:
            return pd.DataFrame()
        
        distances_per_frame_df = du.calculate_distances_per_frame(Spot_stats) # Call the function to calculate distances_per_frame_df
        absolute_directions_per_frame_df = du.calculate_absolute_directions_per_frame(Spot_stats) # Call the function to calculate directions_per_frame_df
        speeds_per_frame = du.calculate_speed(Spot_stats, ['REPLICATE', 'POSITION_T']) # Call the function to calculate speeds_per_frame
        mean_n_median_track_length_net_destance_confinement_ratios_per_frame = du.calculate_mean_median_std_cr_nd_tl_per_frame(Spot_stats)

        Time_stats_dfs = [mean_n_median_track_length_net_destance_confinement_ratios_per_frame, distances_per_frame_df, absolute_directions_per_frame_df, speeds_per_frame]

        Time_stats = du.merge_dfs(Time_stats_dfs, on=['CONDITION', 'REPLICATE', 'POSITION_T'])
        # Frame_stats = Frame_stats.merge(Spot_stats['POSITION_T'].drop_duplicates(), on='POSITION_T')

        Time_stats = Time_stats.sort_values(by=['CONDITION', 'REPLICATE', 'POSITION_T'])

        return Time_stats
    

    @reactive.effect
    def update_Time_stats_df():
        if file_detected.get() == False:
            return pd.DataFrame()
        
        else:
            Time_stats = process_time_data()
            raw_Time_stats_df.set(Time_stats)
            Time_stats_df.set(Time_stats)

        Time_stats = process_time_data()
        raw_Time_stats_df.set(Time_stats)
        Time_stats_df.set(Time_stats)





    # =============================================================================================================================================================================================================================================================================
    # Separately displaying Spot, Track and Time dataframes with possibility for CSV download


    with ui.layout_columns():  
        with ui.card():  
            ui.card_header("Spot stats")

            @render.data_frame
            def render_spot_stats():
                if file_detected.get() == False:
                    return pd.DataFrame()
        
                else:
                    Spot_stats = Spot_stats_df.get()
                    return render.DataGrid(Spot_stats)
                
            @render.download(label="Download", filename="Spot_stats.csv")
            def download_spot_stats():
                with io.BytesIO() as buf:
                    Spot_stats_df.get().to_csv(buf, index=False)
                    yield buf.getvalue()
            
        
        with ui.card():
            ui.card_header("Track stats")
            
            @render.data_frame
            def render_track_stats():
                if file_detected.get() == False:
                    return pd.DataFrame()
                else:
                    Track_stats = Track_stats_df.get()
                    return render.DataGrid(Track_stats)
                
            @render.download(label="Download", filename="Track_stats.csv")
            def download_track_stats():
                with io.BytesIO() as buf:
                    Track_stats_df.get().to_csv(buf, index=False)
                    yield buf.getvalue()
            
            
        with ui.card():
            ui.card_header("Frame stats")

            @render.data_frame
            def render_time_stats():
                if file_detected.get() == False:
                    return pd.DataFrame()
                else:
                    Time_stats = Time_stats_df.get()
                    return render.DataGrid(Time_stats)
                
            @render.download(label="Download", filename="Time stats.csv")
            def download_time_stats():
                with io.BytesIO() as buf:
                    Time_stats_df.get().to_csv(buf, index=False)
                    yield buf.getvalue()















# ===========================================================================================================================================================================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Thresholding utilities - functions


def _update_slider(filter_type, slider, slider_values):
    if filter_type == "percentile":
        ui.update_slider(id=slider, min=0, max=100, value=(0, 100), step=1)
    elif filter_type == "literal":
        values = slider_values.get()
        range = values[1] - values[0]

        if range <= 10:
            steps = 0.01
        elif range <= 100:
            steps = 0.1
        else:
            steps = 1
        
        if values:
            ui.update_slider(id=slider, min=values[0], max=values[1], value=values, step=steps)


def _update_slider_values(metric, filter, dfA, dfB, slider_values):
    if metric in Track_metrics.get():
        try:
            if filter == "literal":
                if dfA.empty:
                    slider_values.set([0, 100])
                else:
                    values = du.values_for_a_metric(dfA, metric)
                    slider_values.set(values)
            elif filter == "percentile":
                slider_values.set([0, 100])
        except Exception as e:
            slider_values.set([0, 100])
    elif metric in Spot_metrics.get():
        try:
            if filter == "literal":
                if dfB.empty:
                    slider_values.set([0, 100])
                else:
                    values = du.values_for_a_metric(dfB, metric)
                    slider_values.set(values)
            elif filter == "percentile":
                slider_values.set([0, 100])
        except Exception as e:
            slider_values.set([0, 100])
    

def _thresholded_histogram(metric, filter_type, slider_range, dfA, dfB):
    if file_detected.get() == False:
        return None
    elif dfA == None:
        return None
    elif dfB == None:
        return None
    else:
        if metric in Track_metrics.get():
            data = dfA.get()
        elif metric in Spot_metrics.get():
            data = dfB.get()
        elif data.empty:
            return plt.figure()
        else:
            return plt.figure()

        values = data[metric].dropna()

        if filter_type == "percentile":
            lower_percentile = np.percentile(values, slider_range[0])
            upper_percentile = np.percentile(values, slider_range[1])
            lower_bound = lower_percentile
            upper_bound = upper_percentile
        else:
            lower_bound = slider_range[0]
            upper_bound = slider_range[1]

        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(values, bins=40)

        for i in range(len(patches)):
            if bins[i] < lower_bound or bins[i+1] > upper_bound:
                patches[i].set_facecolor('grey')
            else:
                patches[i].set_facecolor('#337ab7')

        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.spines[['top','left','right']].set_visible(False)

        return fig


def _data_thresholding_numbers(df):
    raw = raw_Track_stats_df.get().shape[0]
    filtered = df.shape[0]
    filtered_out = raw - filtered

    # Filtered data in percents
    filtered_prcbt = filtered / raw * 100
    filtered_out_prcbt = filtered_out / raw * 100

    return f"Cells in total: {raw}", f"In focus: {round(filtered_prcbt)} % ({filtered})", f"Filtered out: {round(filtered_out_prcbt)} % ({filtered_out})"


def _thresholded_data(filter_type, metric, slider_range, dfA, dfB):
    if filter_type == "percentile":
        if metric in Track_metrics.get():
            return du.percentile_thresholding(dfA, metric, slider_range)
        elif metric in Spot_metrics.get():
            return du.percentile_thresholding(dfB, metric, slider_range)
    elif filter_type == "literal":
        if metric in Track_metrics.get():
            return du.literal_thresholding(dfA, metric, slider_range)
        elif metric in Spot_metrics.get():
            return du.literal_thresholding(dfB, metric, slider_range)
        

def _set_thresholded_data(dfA, dfB, df0A, df0B):
    dfA.set(df0A.get())
    dfB.set(df0B.get())


def _update_thresholded_data(metric, dfA, dfB, df0A, df0B, thresholded_df):
    if metric in Track_metrics.get():
        dfA.set(thresholded_df)
        dfB.set(du.dataframe_filter(df0B.get(), dfA.get()))
    elif metric in Spot_metrics.get():
        dfB.set(thresholded_df)
        dfA.set(du.dataframe_filter(df0A.get(), dfB.get()))



# ============================================================================================================================================================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sidebar

# pass_check = reactive.value(True)

with ui.sidebar(open="open", position="right", bg="f8f8f8"): 


    # ================================================================================================================================================================================================================================================================================
    # Thresholding window no. 1
    # 
    # 1. Thresholding metric selection
    # 2. Thresholding filter selection
    # 3. Thresholding slider
    # 4. Thresholding histogram
    # 5. Thresholding output - numbers (cells in total, filtered in focus, filtered out)


    ui.markdown(
        """
        ###### **Thresholding no. 1**

        """
        )

    with ui.panel_well():


        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Metirc and filter selection selection
        # Slider rendering

        ui.input_select(  
            "metricA",  
            "Thresholding metric:",  
            select_metrics.tracks 
            )  

        ui.input_select(
            "filterA",
            "Thresholding filter:",
            select_mode.thresholding
            )

        ui.input_slider(
            "sliderA",
            "Threshold",
            min=0,
            max=100,
            value=(0, 100)
            )


        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Updating the slider range
        
        @reactive.effect
        def update_sliderA():
            return _update_slider(input.filterA(), "sliderA", slider_valuesT1)

        @reactive.effect
        def update_slider_valuesA():
            return _update_slider_values(input.metricA(), input.filterA(), raw_Track_stats_df.get(), raw_Spot_stats_df.get(), slider_valuesT1)


        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Data filtering itself
        
        @reactive.calc
        def thresholded_dataA():
            return _thresholded_data(input.filterA(), input.metricA(), input.sliderA(), raw_Track_stats_df.get(), raw_Spot_stats_df.get())

        @reactive.effect
        def set_thresholded_dataA():
            if delayed_detection.get() == False:
                _set_thresholded_data(Track_stats_df_T, Spot_stats_df_T, raw_Track_stats_df, raw_Spot_stats_df)
            else:
                pass

        @reactive.effect
        def update_thresholded_dataA():
            return _update_thresholded_data(input.metricA(), Track_stats_df_T, Spot_stats_df_T, raw_Track_stats_df, raw_Spot_stats_df, thresholded_dataA())


        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Thresholding histogram

        @render.plot
        def threshold_histogramA():
            return _thresholded_histogram(input.metricA(), input.filterA(), input.sliderA(), raw_Track_stats_df, raw_Spot_stats_df)


        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Output - numbers

        @render.text
        def data_thresholding_numbersA1():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = _data_thresholding_numbers(raw_Track_stats_df.get())
            return a

        @render.text
        def data_thresholding_numbersA2():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = _data_thresholding_numbers(Track_stats_df_T.get())
            return b
            
        @render.text
        def data_thresholding_numbersA3():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = _data_thresholding_numbers(Track_stats_df_T.get())
            return c
        
            
        

    # ================================================================================================================================================================================================================================================================================
    # Thresholding well no. 2
    # 
    # 1. Thresholding metric selection
    # 2. Thresholding filter selection
    # 3. Thresholding slider
    # 4. Thresholding histogram
    # 5. Thresholding output - numbers (cells in total, filtered in focus, filtered out)


    ui.markdown(
        """
        ###### **Thresholding no. 1**

        """
        )

    with ui.panel_well():

        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Metirc and filter selection selection
        # Slider rendering

        ui.input_select(  
            id="metricB",  
            label="Thresholding metric:",  
            choices=select_metrics.spots_n_tracks ,
            selected="NUM_FRAMES"
            )   

        ui.input_select(
            "filterB",
            "Thresholding filter:",
            select_mode.thresholding
            )

        ui.input_slider(
            "sliderB",
            "Threshold",
            min=0,
            max=100,
            value=(0, 100)
            )


        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Updating the slider range
        
        @reactive.effect
        def update_sliderB():
            return _update_slider(input.filterB(), "sliderB", slider_valuesT2)

        @reactive.effect
        def update_slider_valuesB():
            return _update_slider_values(input.metricB(), input.filterB(), Track_stats_df_T.get(), Spot_stats_df_T.get(), slider_valuesT2)


        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Data filtering itself
        
        @reactive.calc
        def thresholded_dataB():
            return _thresholded_data(input.filterB(), input.metricB(), input.sliderB(), Track_stats_df_T.get(), Spot_stats_df_T.get())

        @reactive.effect
        def set_thresholded_dataB():
            if delayed_detection.get() == False:
                _set_thresholded_data(Track_stats_df, Spot_stats_df, Track_stats_df_T, Spot_stats_df_T)
            else:
                pass
            
        @reactive.effect
        def update_thresholded_dataB():
            return _update_thresholded_data(input.metricB(), Track_stats_df, Spot_stats_df, Track_stats_df_T, Spot_stats_df_T, thresholded_dataB())


        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Thresholding histogram

        @render.plot
        def threshold_histogramB():
            return _thresholded_histogram(input.metricB(), input.filterB(), input.sliderB(), Track_stats_df_T, Spot_stats_df_T)


        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Output - numbers

        @render.text
        def data_thresholding_numbersB1():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = _data_thresholding_numbers(Track_stats_df.get())
            return a

        @render.text
        def data_thresholding_numbersB2():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = _data_thresholding_numbers(Track_stats_df.get())
            return b

        @render.text
        def data_thresholding_numbersB3():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = _data_thresholding_numbers(Track_stats_df.get())
            return c
        








# ================================================================================================================================================================================================================================================================================
# ================================================================================================================================================================================================================================================================================
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Visualisation panel


with ui.nav_panel("Visualisation"):

    with ui.navset_pill_list(widths=(2, 9), selected="Time series"):

        # ==========================================================================================================================================================================================================================================================================
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Track visualization pill

        with ui.nav_panel("Tracks"):


            # ==================================================================================================================================================================================================================================================================================
            # Plotly - interactive visualization settings'

            with ui.panel_well():

                ui.markdown(
                    """
                    #### **Interactive track visualization**
                    *made with*  `plotly`
                    <hr style="height: 4px; background-color: black; border: none" />
                    """
                    )

                # 1. Hover info selection (metrics)

                ui.input_selectize(
                    'let_me_look_at_these',
                    'Let me look at these:',
                    select_metrics.tracks,
                    multiple=True,
                    selected=['CONDITION', 'REPLICATE', 'TRACK_ID'],
                    )
                
                ui.input_action_button(
                    'hover_info',
                    'see info',
                    class_="btn btn-primary",
                    style="padding: 4px 50px"
                    )
                
                @reactive.effect
                @reactive.event(input.hover_info)
                def update_hover_info():
                    see_hover.set(input.let_me_look_at_these())

                ui.markdown(
                    """
                    <hr style="border: none; border-top: 1px dotted" />
                    """
                    )


                # 2. Marker size setting
                # 3. Markers displayed at the end of the tracks
                # 4. Markers selection (for the end of the tracks)

                ui.input_numeric(
                    'marker_size',
                    'Marker size:',
                    5
                    )
                
                @debounce(1)
                @reactive.calc
                def update_marker_size():
                    return input.marker_size()
                
                
                ui.input_checkbox(
                    'end_track_markers',
                    'markers at the end of the tracks',
                    True
                    )
                
                ui.input_select(
                    'markers',
                    'Markers:',
                    select_markers.classic,
                    selected='circle',
                    )

                ui.input_checkbox(
                    'I_just_wanna_be_normal',
                    'just be normal',
                    True
                    )

                @reactive.effect
                @reactive.event(input.I_just_wanna_be_normal)
                def update_markers():
                    if input.I_just_wanna_be_normal():
                        ui.update_select(
                            id='markers',
                            choices=select_markers.classic,
                            selected='circle-open',
                            )
                        ui.update_numeric(
                            id='marker_size',
                            value=5,
                            )
                        marker_size.set(5)

                    else:
                        ui.update_select(
                            id='markers',
                            choices=select_markers.not_normal,
                            selected='scaled',
                            )
                        ui.update_numeric(
                            id='marker_size',
                            value=14,
                            )
                        marker_size.set(14)
                        
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Plotly - interactive visualization
            # 
            # 1. Plotly figure rendering
            # 2. Plotly figure download button (HTML)

            with ui.card():

                @render_plotly
                def interactive_true_track_visualization():
                    fig = pu.Visualize_tracks_plotly(
                        Spots_df=Spot_stats_df.get(),
                        Tracks_df=Track_stats_df.get(), 
                        condition=input.condition(), 
                        replicate=input.replicate(), 
                        c_mode=input.color_mode(), 
                        only_one_color=input.only_one_color(), 
                        lut_scaling_metric=input.lut_scaling(), 
                        let_me_look_at_these=see_hover.get(), 
                        background=input.background(),
                        smoothing_index=update_smoothing(),
                        lw=update_line_width(),
                        marker_size=update_marker_size(),
                        end_track_markers=input.end_track_markers(),
                        markers=input.markers(),
                        I_just_wanna_be_normal=input.I_just_wanna_be_normal(), 
                        metric_dictionary=select_metrics.tracks,
                        show_tracks=input.show_tracks(),
                        )
                    return fig
                

                @render.download(label="Download interactive HTML figure", filename="Track visualization.html")
                def download_true_interactive_visualization_html():
                    fig = pu.Visualize_tracks_plotly(
                        Spots_df=Spot_stats_df.get(),
                        Tracks_df=Track_stats_df.get(), 
                        condition=input.condition(), 
                        replicate=input.replicate(), 
                        c_mode=input.color_mode(), 
                        only_one_color=input.only_one_color(), 
                        lut_scaling_metric=input.lut_scaling(), 
                        let_me_look_at_these=see_hover.get(), 
                        background=input.background(),
                        smoothing_index=update_smoothing(),
                        lw=update_line_width(),
                        marker_size=update_marker_size(),
                        end_track_markers=input.end_track_markers(),
                        markers=input.markers(),
                        I_just_wanna_be_normal=input.I_just_wanna_be_normal(), 
                        metric_dictionary=select_metrics.tracks,
                        show_tracks=input.show_tracks(),
                        )
                    with io.BytesIO():
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                        fig.write_html(tmp_file.name)
                        tmp_file.close()
                        yield Path(tmp_file.name).read_bytes()
            

                @render_plotly
                def interactive_normalized_track_visualization():
                    fig = pu.Visualize_normalized_tracks_plotly(
                        Spots_df=Spot_stats_df.get(),
                        Tracks_df=Track_stats_df.get(),
                        condition=input.condition(),
                        replicate=input.replicate(),
                        c_mode=input.color_mode(),
                        only_one_color=input.only_one_color(),
                        lut_scaling_metric=input.lut_scaling(),
                        smoothing_index=update_smoothing(),
                        lw=update_line_width(),
                        marker_size=update_marker_size(),
                        end_track_markers=input.end_track_markers(),
                        markers=input.markers(),
                        I_just_wanna_be_normal=input.I_just_wanna_be_normal(),
                        metric_dictionary=select_metrics.tracks,
                        let_me_look_at_these=see_hover.get(),
                        show_tracks=input.show_tracks(),
                        )
                    return fig
                
                @render.download(label="Download interactive HTML figure", filename="Normalized track visualization.html")
                def download_normalized_interactive_visualization_html():
                    fig = pu.Visualize_normalized_tracks_plotly(
                        Spots_df=Spot_stats_df.get(),
                        Tracks_df=Track_stats_df.get(),
                        condition=input.condition(),
                        replicate=input.replicate(),
                        c_mode=input.color_mode(),
                        only_one_color=input.only_one_color(),
                        lut_scaling_metric=input.lut_scaling(),
                        smoothing_index=update_smoothing(),
                        lw=update_line_width(),
                        marker_size=update_marker_size(),
                        end_track_markers=input.end_track_markers(),
                        markers=input.markers(),
                        I_just_wanna_be_normal=input.I_just_wanna_be_normal(),
                        metric_dictionary=select_metrics.tracks,
                        let_me_look_at_these=see_hover.get(),
                        show_tracks=input.show_tracks(),
                        )
                    with io.BytesIO():
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                        fig.write_html(tmp_file.name)
                        tmp_file.close()
                        yield Path(tmp_file.name).read_bytes()
                        

            # ==================================================================================================================================================================================================================================================================================
            # Matplotlib - static visualization settings

            with ui.panel_well():

                ui.markdown(
                    """
                    #### **Static track visualization**
                    *made with*  `matplotlib`
                    <hr style="height: 4px; background-color: black; border: none" />
                    """
                    )

                # 1. Arrow size setting
                # 2. Arrows displayed at the end of the tracks
                # 3. Grid displayment

                ui.input_numeric(
                    'arrow_size',
                    'Arrow size:',
                    6
                    )
                
                @debounce(1)
                @reactive.calc
                def update_arrow_size():
                    return input.arrow_size()
                
                ui.input_checkbox(
                    'arrows',
                    'arrows at the end of tracks',
                    True
                    )
                
                ui.input_checkbox(
                    "grid", 
                    "grid", 
                    True
                    ) 

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Matplotlib - static visualization
            #
            # 1. Matplotlib figure rendering
            # 2. Matplotlib figure download button (SVG)

            with ui.card():

                @render.plot
                def true_track_visualization():
                    plot = pu.Visualize_tracks_matplotlib(
                        Spots_df=Spot_stats_df.get(),
                        Tracks_df=Track_stats_df.get(), 
                        condition=input.condition(), 
                        replicate=input.replicate(), 
                        c_mode=input.color_mode(), 
                        only_one_color=input.only_one_color(), 
                        lut_scaling_metric=input.lut_scaling(), 
                        background=input.background(),
                        grid=input.grid(),
                        smoothing_index=update_smoothing(),
                        lw=update_line_width(),
                        arrows=input.arrows(),
                        arrowsize=update_arrow_size(),
                        show_tracks=input.show_tracks(),
                        )
                    return plot
                
                @render.download(label="Download SVG figure", filename="Track visualization.svg")
                def download_true_visualization_svg():
                    plot = pu.Visualize_tracks_matplotlib(
                        Spots_df=Spot_stats_df.get(),
                        Tracks_df=Track_stats_df.get(), 
                        condition=input.condition(), 
                        replicate=input.replicate(), 
                        c_mode=input.color_mode(), 
                        only_one_color=input.only_one_color(), 
                        lut_scaling_metric=input.lut_scaling(), 
                        background=input.background(),
                        grid=input.grid(),
                        smoothing_index=update_smoothing(),
                        lw=update_line_width(),
                        arrows=input.arrows(),
                        arrowsize=update_arrow_size(),
                        show_tracks=input.show_tracks(),
                        )
                    with io.BytesIO() as buf:
                        plot.savefig(buf, format="svg")
                        yield buf.getvalue()


                @render.plot
                def normalized_track_visualization():
                    plot = pu.Visualize_normalized_tracks_matplotlib(
                        Spots_df=Spot_stats_df.get(),
                        Tracks_df=Track_stats_df.get(), 
                        condition=input.condition(), 
                        replicate=input.replicate(), 
                        lut_scaling_metric=input.lut_scaling(),
                        c_mode=input.color_mode(), 
                        only_one_color=input.only_one_color(), 
                        smoothing_index=update_smoothing(),
                        lw=update_line_width(),
                        grid=input.grid(),
                        arrows=input.arrows(),
                        arrowsize=update_arrow_size(),
                        show_tracks=input.show_tracks(),
                        )
                    return plot
                
                @render.download(label="Download SVG figure", filename="Normalized track visualization.svg")
                def download_normalized_visualization_svg():
                    plot = pu.Visualize_normalized_tracks_matplotlib(
                        Spots_df=Spot_stats_df.get(),
                        Tracks_df=Track_stats_df.get(), 
                        condition=input.condition(), 
                        replicate=input.replicate(), 
                        lut_scaling_metric=input.lut_scaling(),
                        c_mode=input.color_mode(), 
                        only_one_color=input.only_one_color(), 
                        smoothing_index=update_smoothing(),
                        lw=update_line_width(),
                        grid=input.grid(),
                        arrows=input.arrows(),
                        arrowsize=update_arrow_size(),
                        show_tracks=input.show_tracks(),
                        )
                    with io.BytesIO() as buf:
                        plot.savefig(buf, format="svg")
                        yield buf.getvalue()

                ui.markdown(
                    """
                    ___
                    """
                    )

                @render.download(label="Download LUT map as SVG", filename="LUT map.svg")
                def download_lut_map_svg():
                    lut_map = pu.Lut_map(
                        Tracks_df=Track_stats_df.get(),
                        c_mode=input.color_mode(), 
                        lut_scaling_metric=input.lut_scaling(), 
                        metrics_dict=select_metrics.tracks,
                        )
                    with io.BytesIO() as buf:
                        lut_map.savefig(buf, format="svg")
                        yield buf.getvalue()


            # ==================================================================================================================================================================================================================================================================================
            # Common track visualization settings

            with ui.panel_well():

                ui.markdown(
                    """
                    ##### **Common track visualization settings**
                    ___
                    """
                    )

                # 1. Condition selections
                # 2. Replicate selections

                ui.input_select(
                    "condition",
                    "Condition:",
                    []
                    )
                
                ui.input_select(
                    "replicate",
                    "Replicate:",
                    []
                    )

                @reactive.effect
                def select_cond():
                    dictionary = du.get_cond_repl(Track_stats_df.get())	

                    # Can use [] to remove all choices
                    if Track_stats_df.get().empty:
                        conditions = []

                    conditions = list(dictionary.keys())

                    ui.update_select(
                        id='condition',
                        choices=conditions
                    )

                @reactive.effect
                def select_repl():
                    condition = input.condition()
                    dictionary = du.get_cond_repl(Track_stats_df.get())

                    if Track_stats_df.get().empty:
                        replicates = []

                    if condition in dictionary:
                        replicates = dictionary[condition]
                    else:
                        replicates = []

                    ui.update_select(
                        id='replicate',
                        choices=replicates
                        )
                    

                ui.markdown(
                    """
                    <hr style="border: none; border-top: 1px dotted" />
                    """
                    )
                

                # 3. Color mode selection ()
                # 4. LUT scaling metric selection (utiised in color scaling
                # 5. Color selection (for single color mode)
                # 6. Background selection (dark/light)

                ui.input_select(
                    "color_mode",
                    "Color mode:",
                    select_mode.color_modes,
                    selected='random colors',
                    )
                
                ui.input_select(
                    'lut_scaling',
                    'LUT scaling metric:',
                    select_metrics.lut,
                    selected='NET_DISTANCE',
                    )

                ui.input_select(
                    'only_one_color',
                    'Color:',
                    select_mode.colors,
                    )
                
                ui.input_select(
                    'background',
                    'Background:',
                    select_mode.background,
                    selected='dark',
                    )
                
                ui.markdown(
                    """
                    <hr style="border: none; border-top: 1px dotted" />
                    """
                    )
                

                # 7. Smoothing index selection (for smoothing the tracks)
                # 8. Line (track) width setting
                
                ui.input_numeric(
                    "smoothing", 
                    "Smoothing:", 
                    0, 
                    min=1, 
                    max=100)
                
                @debounce(1)
                @reactive.calc
                def update_smoothing():
                    return input.smoothing()
                
                ui.input_numeric(
                    'line_width',
                    'Line width:',
                    0.85,
                    min=0,
                    step=0.05
                    )
                
                @debounce(1)
                @reactive.calc
                def update_line_width():
                    return input.line_width()
                
                ui.input_checkbox(
                    'show_tracks',
                    'show tracks',
                    True
                    )






                
        # ==========================================================================================================================================================================================================================================================================
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Time series panel

        with ui.nav_panel("Time series"):
            
            with ui.panel_well():
                
                ui.markdown(
                    """
                    #### **Time series with a polynomial fit**
                    *made with*  `altair`
                    <hr style="height: 4px; background-color: black; border: none" />
                    """
                    )

                ui.input_numeric(
                    "ts_degree",
                    "Fitting degree:",
                    1,
                    min=0,
                    max=15
                    )
                
                ui.markdown(
                    """
                    <hr style="border: none; border-top: 1px dotted" />
                    """
                    )
                
                ui.input_numeric(
                    "ts_scatter_size",
                    "Scatter size:",
                    60,
                    min=1,
                    )
                
                ui.input_numeric(
                    "ts_outline_width",
                    "Outline width:",
                    2.5,
                    min=1,
                    step=0.25
                    )
                
                ui.input_numeric(
                    "ts_opacity",
                    "Opacity:",
                    0.6,
                    min=0,
                    max=1,
                    step=0.05
                    )

                ui.input_checkbox(
                    "ts_fill",
                    "fill scatter points",
                    False
                    )
                
                ui.input_checkbox(
                    "ts_outline",
                    "outline scatter points (when filled)",
                    False
                    )
                
            with ui.card():

                @render_altair
                def time_series_poly_fit_chart():
                    chart = pu.Scatter_poly_fit_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()], 
                        degree=[input.ts_degree()], 
                        cmap=input.ts_cmap(), 
                        point_fill=input.ts_fill(), 
                        point_size=input.ts_scatter_size(), 
                        point_outline=input.ts_outline(), 
                        point_outline_width=input.ts_outline_width(), 
                        opacity=input.ts_opacity(),
                        )
                    return chart
                
                @render.download(label="Download interactive HTML chart", filename="Time series interactive - polynomial fit chart.html")
                def download_time_series_poly_fit_chart__html():
                    chart = pu.Scatter_poly_fit_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()], 
                        degree=[input.ts_degree()], 
                        cmap=input.ts_cmap(), 
                        point_fill=input.ts_fill(), 
                        point_size=input.ts_scatter_size(), 
                        point_outline=input.ts_outline(), 
                        point_outline_width=input.ts_outline_width(), 
                        opacity=input.ts_opacity(),
                        )
                    with io.BytesIO():
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                        chart.save(tmp_file.name)
                        tmp_file.close()
                        yield Path(tmp_file.name).read_bytes()
                
                @render.download(label="Download SVG figure", filename="Time series - polynomial fit figure.svg")
                def download_time_series_poly_fit_chart_svg():
                    chart = pu.Scatter_poly_fit_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()], 
                        degree=[input.ts_degree()], 
                        cmap=input.ts_cmap(), 
                        point_fill=input.ts_fill(), 
                        point_size=input.ts_scatter_size(), 
                        point_outline=input.ts_outline(), 
                        point_outline_width=input.ts_outline_width(), 
                        opacity=input.ts_opacity(),
                        )
                    with io.BytesIO():
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
                        chart.save(tmp_file.name)
                        tmp_file.close()
                        yield Path(tmp_file.name).read_bytes()
                
                

            with ui.panel_well():

                ui.markdown(
                    """
                    #### **Time series line chart**
                    *made with*  `altair`
                    <hr style="height: 4px; background-color: black; border: none" />
                    """
                    )

                ui.input_checkbox(
                    "ts_show_median",
                    "show median",
                    False
                    )
                
            with ui.card():
                
                @render_altair
                def time_series_line_chart_altair():
                    chart = pu.Line_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()],
                        cmap=input.ts_cmap(), 
                        interpolation=input.ts_interpolation(), 
                        show_median=input.ts_show_median(),
                        )
                    return chart
                
                @render.download(label="Download interactive HTML chart", filename="Time series interactive - line chart.html")
                def download_time_series_line_chart_html():
                    chart = pu.Line_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()],
                        cmap=input.ts_cmap(), 
                        interpolation=input.ts_interpolation(), 
                        show_median=input.ts_show_median(),
                        )
                    with io.BytesIO():
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                        chart.save(tmp_file.name)
                        tmp_file.close()
                        yield Path(tmp_file.name).read_bytes()

                @render.download(label="Download SVG figure", filename="Time series - line chart.svg")
                def download_time_series_line_chart_svg():
                    chart = pu.Line_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()],
                        cmap=input.ts_cmap(), 
                        interpolation=input.ts_interpolation(), 
                        show_median=input.ts_show_median(),
                        )
                    with io.BytesIO():
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
                        chart.save(tmp_file.name)
                        tmp_file.close()
                        yield Path(tmp_file.name).read_bytes()


            with ui.panel_well():

                ui.markdown(
                    """
                    #### **Errorband chart**
                    *made with*  `altair`
                    <hr style="height: 4px; background-color: black; border: none" />
                    """
                    )

                ui.input_select(
                    "ts_extent",
                    "Extent:",
                    select_mode.extent,
                    selected='orig_std'
                    )
                
                ui.input_checkbox(
                    "ts_show_mean",
                    "show mean",
                    True
                    )
                
            with ui.card():
                
                @render_altair
                def errorband_chart_altair():
                    chart = pu.Errorband_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()],
                        cmap=input.ts_cmap(), 
                        interpolation=input.ts_interpolation(), 
                        extent=input.ts_extent(), 
                        show_mean=input.ts_show_mean(),
                        )
                    return chart
                
                @render.download(label="Download interactive HTML chart", filename="Time series interactive - error band chart.html")
                def download_errorband_chart_html():
                    chart = pu.Errorband_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()],
                        cmap=input.ts_cmap(), 
                        interpolation=input.ts_interpolation(), 
                        extent=input.ts_extent(), 
                        show_mean=input.ts_show_mean(),
                        )
                    with io.BytesIO():
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                        chart.save(tmp_file.name)
                        tmp_file.close()
                        yield Path(tmp_file.name).read_bytes()

                @render.download(label="Download SVG figure", filename="Time series - error band chart.svg")
                def download_errorband_chart_svg():
                    chart = pu.Errorband_chart_altair(
                        Time_df=Time_stats_df.get(), 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        replicates_separately=input.ts_separate_replicates(), 
                        metric=input.ts_metric(), 
                        Metric=select_metrics.time[input.ts_metric()],
                        cmap=input.ts_cmap(), 
                        interpolation=input.ts_interpolation(), 
                        extent=input.ts_extent(), 
                        show_mean=input.ts_show_mean(),
                        )
                    with io.BytesIO():
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
                        chart.save(tmp_file.name)
                        tmp_file.close()
                        yield Path(tmp_file.name).read_bytes()


            with ui.panel_well():

                ui.markdown(
                    """
                    ##### **Common settings**
                    *made with*  `altair`
                    <hr style="height: 4px; background-color: black; border: none" />
                    """
                    )
                
                ui.input_select(
                    "ts_condition",
                    "Condition:",
                    []
                    )
                
                ui.input_select(
                    "ts_replicate",
                    "Replicate:",
                    []
                    )
                
                ui.input_checkbox(
                    "ts_separate_replicates",
                    "show replicates separately (if a condition is selected)",
                    False
                    )
                
                ui.input_select(
                    "ts_metric",
                    "Metric:",
                    select_metrics.time,
                    selected='MEAN_CONFINEMENT_RATIO'
                    )

                @reactive.effect
                def select_cond():
                    dictionary = du.get_cond_repl(Time_stats_df.get())	

                    # Can use [] to remove all choices
                    if Time_stats_df.get().empty:
                        conditions = []

                    conditions = list(dictionary.keys())

                    ui.update_select(
                        id='ts_condition',
                        choices=conditions
                    )

                @reactive.effect
                def select_repl():
                    condition = input.ts_condition()
                    dictionary = du.get_cond_repl(Time_stats_df.get())

                    if Time_stats_df.get().empty:
                        replicates = []

                    if condition in dictionary:
                        replicates = dictionary[condition]
                    else:
                        replicates = []
                        for key in dictionary:
                            replicates = []

                    ui.update_select(
                        id='ts_replicate',
                        choices=replicates
                        )
                    
                ui.markdown(
                    """
                    ___
                    ###### ***Common settings for the line and errorband charts***

                    """
                    )

                ui.input_select(
                    "ts_cmap",
                    "Color map:",
                    select_mode.cmaps_qualitative,
                    selected='tab10'
                    )
                
                ui.input_select(
                    "ts_interpolation",
                    "Interpolation type:",
                    select_mode.interpolation,
                    selected='catmull-rom'
                    )
                
                
                
                
                
            
                
            













        

        with ui.nav_panel("Corelation"):
            with ui.card():

                'uhh'




        with ui.nav_panel("Testing"):
            with ui.card():

                @render_plotly   
                def plotlyplot_normalized_tracks():  
                    normalized_tracks = pu.visualize_normalized_tracks_interactive(
                        df= Spot_stats_df.get(), 
                        c_mode='random colors', 
                        lut_metric='NET_DISTANCE'
                    )

                    return normalized_tracks  
                
                


                
                
                


































            #     with ui.nav_panel("Track visualisation"):
            #         with ui.layout_columns(
            #             col_widths=(6,6,6,6),
            #             row_heights=(3, 4),	
            #         ):
        
            #             with ui.card(full_screen=True):
            #                 ui.card_header("Raw tracks visualization")
            #                 @render.plot
            #                 def raw_tracks():
            #                     return pu.visualize_full_tracks(
            #                         df=Spot_stats_df.get(), 
            #                         df2=Track_stats_df.get(), 
            #                         threshold=None, 
            #                         lw=0.5
            #                         )

            #                 @render.download(label="Download", filename="Raw tracks visualization.png")
            #                 def download_raw_tracks():
            #                     figure = pu.visualize_full_tracks(
            #                         df=Spot_stats_df.get(), 
            #                         df2=Track_stats_df.get(), 
            #                         threshold=None, 
            #                         lw=0.5
            #                         )
            #                     with io.BytesIO() as buf:
            #                         figure.savefig(buf, format="png", dpi=300)
            #                         yield buf.getvalue()

            #             with ui.card(full_screen=True):
            #                 ui.card_header("Smoothened tracks visualization")
            #                 @render.plot
            #                 def smoothened_tracks():
            #                     return pu.visualize_smoothened_tracks(
            #                         df=Spot_stats_df.get(), 
            #                         df2=Track_stats_df.get(), 
            #                         threshold=None, 
            #                         smoothing_type='moving_average', 
            #                         smoothing_index=50, 
            #                         lw=0.8
            #                         )

            #                 @render.download(label="Download", filename="Smoothened tracks visualization.png")
            #                 def download_smoothened_tracks():
            #                     figure = pu.visualize_smoothened_tracks(
            #                         df=Spot_stats_df.get(), 
            #                         df2=Track_stats_df.get(), 
            #                         threshold=None, 
            #                         smoothing_type='moving_average', 
            #                         smoothing_index=50, 
            #                         lw=0.8
            #                         )
            #                     with io.BytesIO() as buf:
            #                         figure.savefig(buf, format="png", dpi=300)
            #                         yield buf.getvalue()

            #     with ui.nav_panel("Directionality plots"):
            #         with ui.layout_columns():
            #             with ui.card(full_screen=True):  
            #                 ui.card_header("Directionality")
            #                 with ui.layout_column_wrap(width=1 / 2):
            #                     with ui.card(full_screen=False):
            #                         ui.card_header("Scaled by confinement ratio")
            #                         @render.plot
            #                         def migration_direction_tracks1():
            #                             figure = pu.migration_directions_with_kde_plus_mean(
            #                                 df=Track_stats_df.get(), 
            #                                 metric='MEAN_DIRECTION_RAD', 
            #                                 subject='Cells', 
            #                                 scaling_metric='CONFINEMENT_RATIO', 
            #                                 cmap_normalization_metric=None, 
            #                                 cmap=cmap_cells, 
            #                                 threshold=None,
            #                                 title_size2=title_size2
            #                                 )
            #                             return figure
                                    
            #                         @render.download(label="Download", filename="Track directionality (scaled by confinement ratio).png")
            #                         def download_migration_direction_tracks1():
            #                             figure = pu.migration_directions_with_kde_plus_mean(
            #                                 df=Track_stats_df.get(), 
            #                                 metric='MEAN_DIRECTION_RAD', 
            #                                 subject='Cells', 
            #                                 scaling_metric='CONFINEMENT_RATIO', 
            #                                 cmap_normalization_metric=None, 
            #                                 cmap=cmap_cells, 
            #                                 threshold=None,
            #                                 title_size2=title_size
            #                                 )
            #                             with io.BytesIO() as buf:
            #                                 figure.savefig(buf, format="png", dpi=300)
            #                                 yield buf.getvalue()
                                    
                                    
                                
            #                     with ui.card(full_screen=False):
            #                         ui.card_header("Scaled by net distance")
            #                         @render.plot
            #                         def migration_direction_tracks2():
            #                             figure = pu.migration_directions_with_kde_plus_mean(
            #                                 df=Track_stats_df.get(), 
            #                                 metric='MEAN_DIRECTION_RAD', 
            #                                 subject='Cells', 
            #                                 scaling_metric='NET_DISTANCE', 
            #                                 cmap_normalization_metric=None, 
            #                                 cmap=cmap_cells, 
            #                                 threshold=None,
            #                                 title_size2=title_size2
            #                                 )
            #                             return figure

            #                         @render.download(label="Download", filename="Track directionality (scaled by net distance).png")
            #                         def download_migration_direction_tracks2():
            #                             figure = pu.migration_directions_with_kde_plus_mean(
            #                                 df=Track_stats_df.get(), 
            #                                 metric='MEAN_DIRECTION_RAD', 
            #                                 subject='Cells', 
            #                                 scaling_metric='NET_DISTANCE', 
            #                                 cmap_normalization_metric=None, 
            #                                 cmap=cmap_cells, 
            #                                 threshold=None,
            #                                 title_size2=title_size
            #                                 )
            #                             with io.BytesIO() as buf:
            #                                 figure.savefig(buf, format="png", dpi=300)
            #                                 yield buf.getvalue()
                            
            #             with ui.card(full_screen=True):
            #                 ui.card_header("Migration heatmaps")
            #                 with ui.layout_column_wrap(width=1 / 2):
            #                     with ui.card(full_screen=False):
            #                         ui.card_header("Standard")        
            #                         @render.plot
            #                         def tracks_migration_heatmap():
            #                             return pu.df_gaussian_donut(
            #                                 df=Track_stats_df.get(), 
            #                                 metric='MEAN_DIRECTION_RAD', 
            #                                 subject='Cells', 
            #                                 heatmap='inferno', 
            #                                 weight=None, 
            #                                 threshold=None,
            #                                 title_size2=title_size2,
            #                                 label_size=label_size,
            #                                 figtext_color=figtext_color,
            #                                 figtext_size=figtext_size
            #                                 )
                                    
            #                         @render.download(label="Download", filename="Cell migration heatmap.png")
            #                         def download_tracks_migration_heatmap():
            #                             figure = pu.df_gaussian_donut(
            #                                 df=Track_stats_df.get(), 
            #                                 metric='MEAN_DIRECTION_RAD', 
            #                                 subject='Cells', 
            #                                 heatmap='inferno', 
            #                                 weight=None, 
            #                                 threshold=None,
            #                                 title_size2=title_size2,
            #                                 label_size=label_size,
            #                                 figtext_color=figtext_color,
            #                                 figtext_size=figtext_size
            #                                 )
            #                             with io.BytesIO() as buf:
            #                                 figure.savefig(buf, format="png", dpi=300)
            #                                 yield buf.getvalue()

            #                     with ui.card(full_screen=False):
            #                         ui.card_header("Weighted")
            #                         with ui.value_box(
            #                         full_screen=False,
            #                         theme="text-red"
            #                         ):
            #                             ""
            #                             "Currently unavailable"
            #                             ""

#                 with ui.nav_panel("Whole dataset histograms"):
                  
#                     with ui.layout_column_wrap(width=2 / 2):
#                         with ui.card(full_screen=False): 
#                             with ui.layout_columns(
#                                 col_widths=(12,12)
#                             ): 
#                                 with ui.card(full_screen=True):
#                                     ui.card_header("Net distances travelled")
#                                     @render.plot(
#                                             width=3600,
#                                             height=500
#                                             )
#                                     def cell_histogram_1():
#                                         figure = pu.histogram_cells_distance(
#                                             df=Track_stats_df.get(), 
#                                             metric='NET_DISTANCE', 
#                                             str='Net'
#                                             )
#                                         return figure
                                    
#                                     @render.download(label="Download", filename="Net distances travelled.png")
#                                     def download_cell_histogram_1():
#                                         figure = pu.histogram_cells_distance(
#                                             df=Track_stats_df.get(), 
#                                             metric='NET_DISTANCE', 
#                                             str='Net'
#                                             )
#                                         with io.BytesIO() as buf:
#                                             figure.savefig(buf, format="png", dpi=300)
#                                             yield buf.getvalue()

#                                 with ui.card(full_screen=True):
#                                     ui.card_header("Track lengths")
#                                     @render.plot(
#                                             width=3800,
#                                             height=1000
#                                             )
#                                     def cell_histogram_2():
#                                         figure = pu.histogram_cells_distance(
#                                             df=Track_stats_df.get(), 
#                                             metric='TRACK_LENGTH', 
#                                             str='Total'
#                                             )
#                                         return figure
                                    
#                                     @render.download(label="Download", filename="Track lengths.png")
#                                     def download_cell_histogram_2():
#                                         figure = pu.histogram_cells_distance(
#                                             df=Track_stats_df.get(), 
#                                             metric='TRACK_LENGTH', 
#                                             str='Total'
#                                             )
#                                         with io.BytesIO() as buf:
#                                             figure.savefig(buf, format="png", dpi=300)
#                                             yield buf.getvalue()
                                    


                
#         with ui.nav_panel("Frames"):
            
#             with ui.navset_card_tab(id="tab2"):
#                 with ui.nav_panel("Histograms"):
#                     with ui.layout_columns(
#                         col_widths={"sm": (12,6,6)},
#                         row_heights=(3,4),
#                         # height="700px",
#                     ):
                        
#                         with ui.card(full_screen=True):
#                             ui.card_header("Speed histogram")
#                             @render.plot
#                             def migration_histogram():
#                                 figure = pu.histogram_frame_speed(df=Frame_stats_df.get())
#                                 return figure

#                             @render.download(label="Download", filename="Speed histogram.png")
#                             def download_migration_histogram():
#                                 figure = pu.histogram_frame_speed(df=Frame_stats_df.get())
#                                 with io.BytesIO() as buf:
#                                     figure.savefig(buf, format="png", dpi=300)
#                                     yield buf.getvalue()

#                 with ui.nav_panel("Directionality plots"):
#                     with ui.layout_columns():
#                         with ui.card(full_screen=True):
#                             ui.card_header("Directionality")
#                             with ui.layout_column_wrap(width=1 / 2):
#                                 with ui.card(full_screen=False):
#                                     ui.card_header("Standard - Scaled by mean distance")
#                                     @render.plot
#                                     def migration_direction_frames1():
#                                         return pu.migration_directions_with_kde_plus_mean(
#                                             df=Frame_stats_df.get(), 
#                                             metric='MEAN_DIRECTION_RAD', 
#                                             subject='Frames (weighted)', 
#                                             scaling_metric='MEAN_DISTANCE', 
#                                             cmap_normalization_metric='POSITION_T', 
#                                             cmap=cmap_frames, 
#                                             threshold=None,
#                                             title_size2=title_size2
#                                             )
                                    
#                                     @render.download(label="Download", filename="Frame directionality (standard - scaled by mean distance).png")
#                                     def download_migration_direction_frames1():
#                                         figure = pu.migration_directions_with_kde_plus_mean(
#                                             df=Frame_stats_df.get(), 
#                                             metric='MEAN_DIRECTION_RAD', 
#                                             subject='Frames (weighted)', 
#                                             scaling_metric='MEAN_DISTANCE', 
#                                             cmap_normalization_metric='POSITION_T', 
#                                             cmap=cmap_frames, 
#                                             threshold=None,
#                                             title_size2=title_size2
#                                             )
#                                         with io.BytesIO() as buf:
#                                             figure.savefig(buf, format="png", dpi=300)
#                                             yield buf.getvalue()

#                                 with ui.card(full_screen=False):
#                                     ui.card_header("Weighted - Scaled by mean distance")
#                                     @render.plot
#                                     def migration_direction_frames2():
#                                         return pu.migration_directions_with_kde_plus_mean(
#                                             df=Frame_stats_df.get(), 
#                                             metric='MEAN_DIRECTION_RAD_weight_mean_dis', 
#                                             subject='Frames (weighted)', 
#                                             scaling_metric='MEAN_DISTANCE', 
#                                             cmap_normalization_metric='POSITION_T', 
#                                             cmap=cmap_frames, 
#                                             threshold=None,
#                                             title_size2=title_size2
#                                             )
                                    
#                                     @render.download(label="Download", filename="Frame directionality (weighted - scaled by mean distance).png")
#                                     def download_migration_direction_frames2():
#                                         figure = pu.migration_directions_with_kde_plus_mean(
#                                             df=Frame_stats_df.get(), 
#                                             metric='MEAN_DIRECTION_RAD_weight_mean_dis', 
#                                             subject='Frames (weighted)', 
#                                             scaling_metric='MEAN_DISTANCE', 
#                                             cmap_normalization_metric='POSITION_T', 
#                                             cmap=cmap_frames, 
#                                             threshold=None,
#                                             title_size2=title_size2
#                                             )
#                                         with io.BytesIO() as buf:
#                                             figure.savefig(buf, format="png", dpi=300)
#                                             yield buf.getvalue()
                
#                         with ui.card(full_screen=True):
#                             ui.card_header("Migration heatmaps")
#                             with ui.layout_column_wrap(width=1 / 2):
#                                 with ui.card(full_screen=False):
#                                     ui.card_header("Standard")        
#                                     @render.plot
#                                     def frame_migration_heatmap_1():
#                                         return pu.df_gaussian_donut(
#                                             df=Frame_stats_df.get(), 
#                                             metric='MEAN_DIRECTION_RAD', 
#                                             subject='Frames', 
#                                             heatmap='viridis', 
#                                             weight=None, 
#                                             threshold=None,
#                                             title_size2=title_size2,
#                                             label_size=label_size,
#                                             figtext_color=figtext_color,
#                                             figtext_size=figtext_size
#                                             )
                                    
#                                     @render.download(label="Download", filename="Frame migration heatmap (standard).png")
#                                     def download_frame_migration_heatmap_1():
#                                         figure = pu.df_gaussian_donut(
#                                             df=Frame_stats_df.get(), 
#                                             metric='MEAN_DIRECTION_RAD', 
#                                             subject='Frames', 
#                                             heatmap='viridis', 
#                                             weight=None, 
#                                             threshold=None,
#                                             title_size2=title_size2,
#                                             label_size=label_size,
#                                             figtext_color=figtext_color,
#                                             figtext_size=figtext_size
#                                             )
#                                         with io.BytesIO() as buf:
#                                             figure.savefig(buf, format="png", dpi=300)
#                                             yield buf.getvalue()

#                                 with ui.card(full_screen=False):
#                                     ui.card_header("Weighted")
#                                     @render.plot
#                                     def frame_migration_heatmap_2():
#                                         return pu.df_gaussian_donut(
#                                             df=Frame_stats_df.get(), 
#                                             metric='MEAN_DIRECTION_RAD_weight_mean_dis', 
#                                             subject='Frames', 
#                                             heatmap='viridis', 
#                                             weight='mean distance traveled', 
#                                             threshold=None,
#                                             title_size2=title_size2,
#                                             label_size=label_size,
#                                             figtext_color=figtext_color,
#                                             figtext_size=figtext_size
#                                             )
                                    
#                                     @render.download(label="Download", filename="Frame migration heatmap (weighted).png")
#                                     def download_frame_migration_heatmap_2():
#                                         figure = pu.df_gaussian_donut(
#                                             df=Frame_stats_df.get(), 
#                                             metric='MEAN_DIRECTION_RAD_weight_mean_dis', 
#                                             subject='Frames', 
#                                             heatmap='viridis', 
#                                             weight='mean distance traveled', 
#                                             threshold=None,
#                                             title_size2=title_size2,
#                                             label_size=label_size,
#                                             figtext_color=figtext_color,
#                                             figtext_size=figtext_size
#                                             )
#                                         with io.BytesIO() as buf:
#                                             figure.savefig(buf, format="png", dpi=300)
#                                             yield buf.getvalue()








# ===========================================================================================================================================================================================================================================================================
# Stistics
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Spot_subdataframes_global = reactive.value()
Track_subdataframes_global = reactive.value()
Frame_subdataframes_global = reactive.value()
addtnl_labels_global = reactive.value()


with ui.nav_panel("Statistics"):

    with ui.layout_column_wrap(height='100%'):
        with ui.card(full_screen=False):
            @render.plot
            def swarmplot():
                metric = input.testing_metric()

                if metric in Track_metrics.get():
                    df = Track_stats_df.get()
                if df.empty:
                    return plt.figure()
                # else:
                #     return plt.figure()
                return pu.swarm_plot(df, metric, select_metrics.spots_n_tracks[metric], show_violin=input.violins(), show_swarm=input.swarm(), show_mean=input.mean(), show_median=input.median(), show_error_bars=input.errorbars(), show_legend=input.legend(), p_testing=input.p_test())
            
            @render.download(label="Download figure", filename="Swarmplot.svg")
            def download_swarmplot():
                metric = input.testing_metric()

                if metric in Track_metrics.get():
                    df = Track_stats_df.get()
                elif df.empty:
                    return plt.figure()
                # else:
                #     return plt.figure()
            
                figure = pu.swarm_plot(df, metric, select_metrics.spots_n_tracks[metric], show_violin=input.violins(), show_swarm=input.swarm(), show_mean=input.mean(), show_median=input.median(), show_error_bars=input.errorbars(), show_legend=input.legend(), p_testing=input.p_test())
                with io.BytesIO() as buf:
                    figure.savefig(buf, format="svg")
                    yield buf.getvalue()
            


    with ui.panel_well():

        ui.input_select(  
                "testing_metric",  
                "Test for metric:",  
                select_metrics.tracks 
            )  
        
        ui.input_checkbox(
                'violins',
                'show violins',
                True
            )

        ui.input_checkbox(
                'swarm',
                'show swarm',
                True
            )
    
        ui.input_checkbox(
                'mean',
                'show mean',
                True
            )

        ui.input_checkbox(
                'median',
                'show median',
                True
            )

        ui.input_checkbox(
                'errorbars',
                'show error bars',
                True
            )
        
        ui.input_checkbox(
                'legend',
                'show legend',
                True
            )

        ui.input_checkbox(
                'p_test',
                'P-test',
                False
            )





















ui.nav_spacer()  
with ui.nav_control():  
    ui.input_dark_mode(mode="light")




# ===========================================================================================================================================================================================================================================================================
# Action buttons for additional browse windows used for inputting other data frames and also making a window for each input which will leave a mark on the dataframe e.g. Treatment CK12 - which will be also written into a column specifying the conditions 
# Merging the dataframes
# exporting downloads in form of a rasterized file