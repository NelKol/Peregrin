from shiny import reactive
from shiny.express import input, render, ui
from shiny.types import FileInfo

import asyncio
import time

import pandas as pd
import numpy as np
import io

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from peregrin.scripts import PlotParams
import utils.data_utils as du
import utils.plot_utils as pu


# ===========================================================================================================================================================================================================================================================================
# Page specs and layout

ui.page_opts(
    title="Peregrin", 
    fillable=False
    )


# ===========================================================================================================================================================================================================================================================================
# Creating reactive variables for raw dataframe storage

raw_Buttered_df = reactive.value()
raw_Spot_stats_df = reactive.value()
raw_Track_stats_df = reactive.value()
raw_Time_stats_df = reactive.value()


# ===========================================================================================================================================================================================================================================================================
# Creating reactive variables for processed dataframe storage

Buttered_df = reactive.value()
Spot_stats_df = reactive.value()
Track_stats_df = reactive.value()
Time_stats_df = reactive.value()

# ===========================================================================================================================================================================================================================================================================
# Creating reactive values for thresholding the data

Spot_stats_df_T = reactive.value()
Track_stats_df_T = reactive.value()


# ===========================================================================================================================================================================================================================================================================
# Creating other reactive variables 

Track_metrics = reactive.value()     # Creating a reactive value for the track metrics
Spot_metrics = reactive.value()      # Creating a reactive value for the spot metrics

slider_valuesT1 = reactive.value()   # Creating a reactive value for the slider values for thresholding 1
slider_valuesT2 = reactive.value()   # Creating a reactive value for the slider values for thresholding 2
slider_valuesT3 = reactive.value()   # Creating a reactive value for the slider values for thresholding 3
count = reactive.value(1)            # Data input counter


slider_values = reactive.value()     # Creating a reactive value for the slider values
conditions = reactive.value()        # Creating a reactive value for the conditions


dict_Track_metrics = {
    "TRACK_LENGTH": "Track length", 
    "NET_DISTANCE": "Net distance", 
    "CONFINEMENT_RATIO": "Confinement ratio",
    "TRACK_POINTS": "Number of points in track",
    "SPEED_MEAN": "Mean speed",
    "SPEED_MEDIAN": "Median speed",
    "SPEED_MAX": "Max speed",
    "SPEED_MIN": "Min speed",
    "SPEED_STD_DEVIATION": "Speed standard deviation",
    "MEAN_DIRECTION_DEG": "Mean direction (degrees)",
    "MEAN_DIRECTION_RAD": "Mean direction (radians)",
    "STD_DEVIATION_DEG": "Standard deviation (degrees)",
    "STD_DEVIATION_RAD": "Standard deviation (radians)",
}

dict_Spot_metrics = {
    "POSITION_T": "Position T",
    "POSITION_X": "Position X",
    "POSITION_Y": "Position Y",
    "CONDITION": "Condition",
    "REPLICATE": "Replicate",
    "DISTANCE": "Distance",
    "TRACK_LENGTH": "Track Length",
    "NET_DISTANCE": "Net Distance",
    "CONFINEMENT_RATIO": "Confinement Ratio",
}

dict_Metrics = dict_Track_metrics | dict_Spot_metrics

dict_Time_metrics = {
    "CONDITION": "Condition",
    "REPLICATE": "Replicate",
    "POSITION_T": "Position T",
    "MEAN_TRACK_LENGTH": "Mean track length",
    "MEAN_NET_DISTANCE": "Mean net distance",
    "MEAN_CONFINEMENT_RATIO": "Mean confinement ratio",
    "MEDIAN_TRACK_LENGTH": "Median track length",
    "MEDIAN_NET_DISTANCE": "Median net distance",
    "MEDIAN_CONFINEMENT_RATIO": "Median confinement ratio",
    "min_DISTANCE": "Minimum distance",
    "STD_DEVIATION_DEG": "Standard deviation (degrees)",
    "MEDIAN_DIRECTION_DEG": "Median direction (degrees)",
    "MEAN_DIRECTION_RAD": "Mean direction (radians)",
    "STD_DEVIATION_RAD": "Standard deviation (radians)",
    "MEDIAN_DIRECTION_RAD": "Median direction (radians)",
    "SPEED_MIN": "Minimum speed",
    "SPEED_MAX": "Maximum speed",
    "SPEED_MEAN": "Mean speed",
    "SPEED_STD_DEVIATION": "Speed standard deviation",
    "SPEED_MEDIAN": "Median speed",
}

Thresholding_filters = {
    "literal": "Literal",
    "percentile": "Percentile",
}

# ===========================================================================================================================================================================================================================================================================
# Data panel
# ===========================================================================================================================================================================================================================================================================
# Reading a selected CSV file and cleaning it
# Extracting separate dataframes











# Directionality metric 
# When downloading, I could make it possible to download the merged df and the separate datasets as well, in which case I would exclude the CONDITION column but rather include it in the name of the file
# I may include some metadata as well for download such as the data of the analysis and what not idk
# # I could also make it possible to download the data as a .txt file or .xlsx file
# # # 2D visualization - gating
# # # remove the percentile thresholding opptions
# # # make the thresholding the way that inputs are, so that you can add and remove them
# # # maybe an option in 1D/2D thresholding











with ui.nav_panel("Input"):
    
    with ui.div(id="data-inputs"): # div container for flow content

        # =============================================================================================================================================================================================================================================================================
        # Buttons for adding and removing additional data input

        ui.input_action_button("add_input", "Add data input")
        ui.input_action_button("remove_input", "Remove data input")


        # =============================================================================================================================================================================================================================================================================
        # Default data input slot

        @render.ui
        def default_input():
            default_browser = ui.input_file("file1", "Input CSV", accept=[".csv"], multiple=True, placeholder="No files selected")
            default_label = ui.input_text("label1", "Condition", placeholder="write something")
            return default_label, default_browser


        # =============================================================================================================================================================================================================================================================================
        # Additional data input slots - reacting on the buttons

        @reactive.effect
        @reactive.event(input.add_input)                             # "Add data input" button sensor
        def add_rowser():
            if input.add_input():                                    # REACTION:
                count.set(count.get() + 1)                      # Increasing the input count
                adding = count.get()                            # Getting the current input count

                browser = ui.input_file(                        # CSV file browser
                    id=f"file{adding}", 
                    label=f"Input CSV {adding}", 
                    accept=[".csv"], 
                    multiple=True, 
                    placeholder="No files selected"
                    )
                label = ui.input_text(                          # Data labeling text window
                    id=f"label{adding}", 
                    label=f"Condition", 
                    placeholder="write something"
                    )

                ui.insert_ui(                                   # Rendering the additional input slot container
                    ui.div(                                     # container consisting of the label, browser and use button
                        {"id": f"additional-input-{adding}"}, 
                        label, browser),
                        selector="#data-inputs",
                        where="beforeEnd",
                )

        @reactive.effect
        @reactive.event(input.remove_input)                             # "Remove data input" button sensor
        def remove_browser():
            if input.remove_input():                                    # REACTION:
                removing = count.get()                          # Getting the current input count
                ui.remove_ui(f"#additional-input-{removing}")   # Removing the last input slot (one with the current input count)
                if count.get() > 1:                             # Decreasing the input count
                    count.set(removing - 1)                     
                else:
                    pass


    @reactive.calc 
    def parsed_file():                                                          # File-reading 
        
        # =============================================================================================================================================================================================================================================================================
        # Processing the default input files

        default = pd.DataFrame()
        additional = pd.DataFrame()
                    
        inpt_file_list_dflt: list[FileInfo] | None = input.file1()                      # Getting the list of default input files

        if inpt_file_list_dflt is None:
            default = pd.DataFrame()
        
        else:
            all_data_dflt = []
            for dflt_file_count, file_dflt in enumerate(inpt_file_list_dflt, start=1):       # Enumerate and cycle through default input files
                df_dflt = pd.read_csv(file_dflt['datapath'])                     
                buttered_dflt = du.butter(df_dflt)                                      # Process the DataFrame
                                                  
                label_dflt = input.label1()                                             # Getting the label to assign the 'CONDITION' column parameter
                if not label_dflt or label_dflt is None:                                # If no label is provided, assign a default one
                    buttered_dflt['CONDITION'] = 1
                else:                                                                   # Else, assign the given lable
                    buttered_dflt['CONDITION'] = f'{label_dflt}'
                buttered_dflt['REPLICATE'] = dflt_file_count

                all_data_dflt.append(buttered_dflt)                                       # Store processed DataFrame

                default = pd.concat(all_data_dflt, axis=0)                              # Join the DataFrames
                
        # =============================================================================================================================================================================================================================================================================
        # Processing the additional input files

        browse_count = count.get()                                                      # Getting the current additional input slot count
        all_data_addtnl = []                                                            # List storing processed DataFrames                            
        for i in range(2, browse_count + 1, 1):                                         # Cycle trough the additional input slots 

            inpt_file_list_addtnl: list[FileInfo] | None = input[f"file{i}"]()         # Getting the list of files

            if inpt_file_list_addtnl is None:
                additional = pd.DataFrame()
            
            else:
                for additnl_file_count, file_addtnl in enumerate(inpt_file_list_addtnl, start=1):                             # Enumerate and cycle through additional input files
                    df_addtnl = pd.read_csv(file_addtnl["datapath"])                  
                    buttered_addtnl = du.butter(df_addtnl)

                    label_addtnl = input[f"label{i}"]()                                # Getting the label to assign the 'CONDITION' column parameter
                    if not label_addtnl or label_addtnl is None:                                        # If no label is provided, assign a default one
                        buttered_addtnl['CONDITION'] = i
                    else:                                                                               # Else, assign the given lable
                        buttered_addtnl['CONDITION'] = f'{label_addtnl}'
                    buttered_addtnl['REPLICATE'] = additnl_file_count

                    all_data_addtnl.append(buttered_addtnl)                                           # Store processed DataFrame
                    

                    additional = pd.concat(all_data_addtnl, axis=0)                     # Join the DataFrames

        # =============================================================================================================================================================================================================================================================================
        # Merging the default and additional input files

        return pd.DataFrame(pd.concat([default, additional], axis=0))





file_detected = reactive.value(False)
delayed_detection = reactive.value(False)
cells_in_possesion = reactive.value(False)

@reactive.effect
def file_detection():
    for i in range(1, count.get() + 1, 1):
        if input[f"file{i}"]() != None:
            cells_in_possesion.set(True)
            file_detected.set(True)
        else:
            pass


@reactive.calc
async def file_detection_delay():
    if file_detected.get():
        await asyncio.sleep(2.5)
        return True
    else:
        return False
        
            

            
 



with ui.nav_panel("Data frames"):  # Data panel

    # =============================================================================================================================================================================================================================================================================
    # Executing the functions 
    # Creating separate dataframes
    # Itermidiate caching of the dataframes

    
    @reactive.effect
    def update_buttered_df():
        if file_detected.get() == False:
            return pd.DataFrame()
        
        df = parsed_file()  # Call the expensive function.
        raw_Buttered_df.set(df)


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
    # Separately displaying the dataframes
    # Enabling the user to download the dataframes as .csv files
    # Enabling data filtering?

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
# Sidebar
# ===========================================================================================================================================================================================================================================================================


def update_slider(filter_type, slider, slider_values):
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

def update_slider_values(metric, filter, dfA, dfB, slider_values):
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
    
def thresholded_histogram(metric, filter_type, slider_range, dfA, dfB):
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

def data_thresholding_numbers(df):
    raw = raw_Track_stats_df.get().shape[0]
    filtered = df.shape[0]
    filtered_out = raw - filtered

    # Filtered data in percents
    filtered_prcbt = filtered / raw * 100
    filtered_out_prcbt = filtered_out / raw * 100

    return f"Cells in total: {raw}", f"In focus: {round(filtered_prcbt)} % ({filtered})", f"Filtered out: {round(filtered_out_prcbt)} % ({filtered_out})"

def thresholded_data(filter_type, metric, slider_range, dfA, dfB):
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
        
def set_thresholded_data(dfA, dfB, df0A, df0B):
    dfA.set(df0A.get())
    dfB.set(df0B.get())

def update_thresholded_data(metric, dfA, dfB, df0A, df0B, thresholded_df):
    if metric in Track_metrics.get():
        dfA.set(thresholded_df)
        dfB.set(du.dataframe_filter(df0B.get(), dfA.get()))
    elif metric in Spot_metrics.get():
        dfB.set(thresholded_df)
        dfA.set(du.dataframe_filter(df0A.get(), dfB.get()))



# ===========================================================================================================================================================================================================================================================================
# Sidebar

pass_check = reactive.value(True)

with ui.sidebar(open="open", position="right", bg="f8f8f8"): 


    @render.text
    def first_thresholding():
        return "Filter no. 1"

    with ui.panel_well():

    # ===========================================================================================================================================================================================================================================================================
    # Creating a possibility for thresholding metric selection
    # Creating a possibility for thresholding filter selection
    # Creating a slider for thresholding

        ui.input_select(  
            "metricA",  
            "Thresholding metric:",  
            dict_Metrics 
        )  

        ui.input_select(
            "filterA",
            "Thresholding filter:",
            Thresholding_filters
        )

        ui.input_slider(
            "sliderA",
            "Threshold",
            min=0,
            max=100,
            value=(0, 100)
        )


        # ===========================================================================================================================================================================================================================================================================
        # Reactive functions updating the slider values
        
        @reactive.effect
        def update_sliderA():
            return update_slider(input.filterA(), "sliderA", slider_valuesT1)

        @reactive.effect
        def update_slider_valuesA():
            return update_slider_values(input.metricA(), input.filterA(), raw_Track_stats_df.get(), raw_Spot_stats_df.get(), slider_valuesT1)


        # ===========================================================================================================================================================================================================================================================================
        # Thresholding the data based on percentiles
        
        @reactive.calc
        def thresholded_dataA():
            return thresholded_data(input.filterA(), input.metricA(), input.sliderA(), raw_Track_stats_df.get(), raw_Spot_stats_df.get())

        @reactive.effect
        def set_thresholded_dataA():
            if delayed_detection.get() == False:
                set_thresholded_data(Track_stats_df_T, Spot_stats_df_T, raw_Track_stats_df, raw_Spot_stats_df)
            else:
                pass

        @reactive.effect
        # @reactive.event(input.threshold1, input.)
        def update_thresholded_dataA():
            return update_thresholded_data(input.metricA(), Track_stats_df_T, Spot_stats_df_T, raw_Track_stats_df, raw_Spot_stats_df, thresholded_dataA())

        @render.plot
        def threshold_histogramA():
            return thresholded_histogram(input.metricA(), input.filterA(), input.sliderA(), raw_Track_stats_df, raw_Spot_stats_df)

        @render.text
        def data_thresholding_numbersA1():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = data_thresholding_numbers(raw_Track_stats_df.get())
            return a

        @render.text
        def data_thresholding_numbersA2():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = data_thresholding_numbers(Track_stats_df_T.get())
            return b
            
        @render.text
        def data_thresholding_numbersA3():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = data_thresholding_numbers(Track_stats_df_T.get())
            return c
    
    # @ui.bind_task_button(button_id="threshold1")
    # @reactive.extended_task
    # async def thresholding1():
    #     await asyncio.sleep(4.5)
        
    # ui.input_task_button("threshold1", "Apply")
    

            
        

# ===========================================================================================================================================================================================================================================================================
# Thresholding 2 panel

# with ui.accordion_panel(title="Tresholding 2"):

    @render.text
    def second_thresholding():
        return "Filter no. 2"

    with ui.panel_well():

        # ===========================================================================================================================================================================================================================================================================
        # Creating a possibility for thresholding metric selection
        # Creating a possibility for thresholding filter selection
        # Creating a slider for thresholding

        ui.input_select(  
            id="metricB",  
            label="Thresholding metric:",  
            choices=dict_Metrics ,
            selected="NUM_FRAMES"
        )  

        ui.input_select(
            "filterB",
            "Thresholding filter:",
            Thresholding_filters
        )

        ui.input_slider(
            "sliderB",
            "Threshold",
            min=0,
            max=100,
            value=(0, 100)
        )


        # ===========================================================================================================================================================================================================================================================================
        # Reactive functions updating the slider values
        
        @reactive.effect
        def update_sliderB():
            return update_slider(input.filterB(), "sliderB", slider_valuesT2)

        @reactive.effect
        def update_slider_valuesB():
            return update_slider_values(input.metricB(), input.filterB(), Track_stats_df_T.get(), Spot_stats_df_T.get(), slider_valuesT2)


        # ===========================================================================================================================================================================================================================================================================
        # Thresholding the data based on percentiles
        
        @reactive.calc
        def thresholded_dataB():
            return thresholded_data(input.filterB(), input.metricB(), input.sliderB(), Track_stats_df_T.get(), Spot_stats_df_T.get())

        @reactive.effect
        def set_thresholded_dataB():
            if delayed_detection.get() == False:
                set_thresholded_data(Track_stats_df, Spot_stats_df, Track_stats_df_T, Spot_stats_df_T)
            else:
                pass
            
        @reactive.effect
        # @reactive.event(input.threshold1, input.threshold2)
        def update_thresholded_dataB():
            return update_thresholded_data(input.metricB(), Track_stats_df, Spot_stats_df, Track_stats_df_T, Spot_stats_df_T, thresholded_dataB())

        @render.plot
        def threshold_histogramB():
            return thresholded_histogram(input.metricB(), input.filterB(), input.sliderB(), Track_stats_df_T, Spot_stats_df_T)

        @render.text
        def data_thresholding_numbersB1():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = data_thresholding_numbers(Track_stats_df.get())
            return a

        @render.text
        def data_thresholding_numbersB2():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = data_thresholding_numbers(Track_stats_df.get())
            return b

        @render.text
        def data_thresholding_numbersB3():
            if cells_in_possesion.get() == False:
                return None
            a, b, c = data_thresholding_numbers(Track_stats_df.get())
            return c
        

    # @ui.bind_task_button(button_id="threshold2")
    # @reactive.extended_task
    # async def thresholding2():
    #     await asyncio.sleep(4.5)
        
    # ui.input_task_button("threshold2", "Apply")


            











# ===========================================================================================================================================================================================================================================================================	
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



# ===========================================================================================================================================================================================================================================================================
# Globally used callables

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




# ===========================================================================================================================================================================================================================================================================
# Visualisation panel
# ===========================================================================================================================================================================================================================================================================
# Data visualisation
# Plots
# Statistical testing?

color_modes = {
    'greyscale': 'B&W',
    'color1': 'jet',
    'color3': 'brg',
    'color4': 'hot',
    'color5': 'viridis',
    'color6': 'rainbow',
    'color7': 'turbo',
    'color8': 'nipy-spectral',
    'color9': 'gist-ncar'
    }

smoothing_index = reactive.value(0)
arrow_size = reactive.value(6)
line_width = reactive.value(1)



with ui.nav_panel("Visualisation"):

    # ===========================================================================================================================================================================================================================================================================
    # Tracks tab

    with ui.navset_pill_list():
        with ui.nav_panel("Tracks"):
            with ui.card():
                ui.card_header("Tracks visualisation")
                @render.plot
                def tracks_plot():
                    return pu.visualize_tracks(
                        df=Spot_stats_df.get(),
                        df2=Track_stats_df.get(),
                        condition=input.condition(), 
                        replicate=input.replicate(), 
                        c_mode=input.color_mode(), 
                        grid=input.grid(),
                        smoothing_index=smoothing_index.get(),
                        lut_metric=input.lut_scaling(),
                        lw=line_width.get(),
                        arrowsize=arrow_size.get()
                        )
                @render.download(label="Download figure", filename="Track visualization.svg")
                def download_tracks_plot():
                    figure = pu.visualize_tracks(
                        df=Spot_stats_df.get(),
                        df2=Track_stats_df.get(),
                        condition=input.condition(), 
                        replicate=input.replicate(), 
                        c_mode=input.color_mode(), 
                        grid=input.grid(),
                        smoothing_index=smoothing_index.get(), 
                        lut_metric=input.lut_scaling(),
                        lw=line_width.get(),
                        arrowsize=arrow_size.get()
                        )
                    with io.BytesIO() as buf:
                        figure.savefig(buf, format="svg")
                        yield buf.getvalue()
                @render.download(label="Download LUT map", filename="Track visualization LUT map.svg")
                def download_tracks_lut_map():
                    figure = pu.tracks_lut_map(
                        df2=Track_stats_df.get(), 
                        c_mode=input.color_mode(), 
                        lut_metric=input.lut_scaling(), 
                        metrics_dict=dict_Track_metrics
                        )
                    with io.BytesIO() as buf:
                        figure.savefig(buf, format="svg", bbox_inches='tight')
                        yield buf.getvalue()
                
            
            with ui.panel_well():

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
                    
                ui.input_select(
                    'lut_scaling',
                    'LUT scaling metric:',
                    dict_Track_metrics
                )

                ui.input_select(
                    "color_mode",
                    "Color mode:",
                    color_modes,
                    selected='color1'
                    )
                
                ui.input_numeric(
                    "smoothing", 
                    "Smoothing:", 
                    0, 
                    min=1, 
                    max=100)
                
                @reactive.effect
                async def update_smoothing():
                    value = input.smoothing()
                    await asyncio.sleep(2.5)
                    return smoothing_index.set(value)
                
                ui.input_numeric(
                    'line_width',
                    'Line width:',
                    1
                    )
                
                @reactive.effect
                async def update_line_width():
                    value = input.line_width()
                    await asyncio.sleep(2.5)
                    return line_width.set(value)

                ui.input_numeric(
                    'arrow_size',
                    'Arrow size:',
                    6
                    )
                
                @reactive.effect
                async def update_arrow_size():
                    value = input.arrow_size()
                    await asyncio.sleep(2.5)
                    return arrow_size.set(value)

                ui.input_checkbox(
                    "grid", 
                    "grid", 
                    True
                    ) 
                
                
        # ==========================================================================================================================================================================================================================================================================
        # Time series panel

        with ui.nav_panel("Time series"):
            
            cmaps_ = ['Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10']
            interpolations_ = [
                "basis",
                "basis-open",
                "basis-closed",
                "bundle",
                "cardinal",
                "cardinal-open",
                "cardinal-closed",
                "catmull-rom",
                "linear",
                "linear-closed",
                "monotone",
                "natural",
                "step",
                "step-before",
                "step-after"
                ]


            with ui.card():
                @render.plot
                def time_series_plot():
                    chart = pu.poly_fit_n_scatter(
                        df=Time_stats_df.get(),
                        metric=input.ts_metric(), 
                        Metric=dict_Metrics[input.ts_metric()], 
                        condition=input.ts_condition(), 
                        replicate=input.ts_replicate(), 
                        degree=[1], 
                        cmap=input.ts_cmap(),
                    )





            with ui.panel_well():

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
                
                ui.input_select(
                    "ts_metric",
                    "Metric:",
                    dict_Metrics,
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
                    condition = input.condition()
                    dictionary = du.get_cond_repl(Time_stats_df.get())

                    if Time_stats_df.get().empty:
                        replicates = []

                    if condition in dictionary:
                        replicates = dictionary[condition]
                    else:
                        replicates = []

                    ui.update_select(
                        id='ts_replicate',
                        choices=replicates
                        )

                ui.input_select(
                    "ts_cmap",
                    "Color map:",
                    cmaps_,
                    selected='tab10'
                    )
                
                ui.input_select(
                    "ts_interpolation",
                    "Interpolation type:",
                    interpolations_,
                    selected='catmull-rom'
                    )
                
                ui.input_numeric(
                    "ts_degree",
                    "Fitting degree:",
                    0,
                    min=0,
                    max=50
                    )















        with ui.nav_panel("Corelation"):
            with ui.card():


                'yeah'




































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
                return pu.swarm_plot(df, metric, dict_Metrics[metric], show_violin=input.violins(), show_swarm=input.swarm(), show_mean=input.mean(), show_median=input.median(), show_error_bars=input.errorbars(), show_legend=input.legend(), p_testing=input.p_test())
            
            @render.download(label="Download figure", filename="Swarmplot.svg")
            def download_swarmplot():
                metric = input.testing_metric()

                if metric in Track_metrics.get():
                    df = Track_stats_df.get()
                elif df.empty:
                    return plt.figure()
                # else:
                #     return plt.figure()
            
                figure = pu.swarm_plot(df, metric, dict_Metrics[metric], show_violin=input.violins(), show_swarm=input.swarm(), show_mean=input.mean(), show_median=input.median(), show_error_bars=input.errorbars(), show_legend=input.legend(), p_testing=input.p_test())
                with io.BytesIO() as buf:
                    figure.savefig(buf, format="svg")
                    yield buf.getvalue()
            


    with ui.panel_well():

        ui.input_select(  
                "testing_metric",  
                "Test for metric:",  
                dict_Track_metrics 
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