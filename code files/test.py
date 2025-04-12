from altair_viewer import display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import plotly.graph_objects as go
import mpld3
pd.options.mode.chained_assignment = None



_cell = '🦠'

_random = ['👨🏽‍🦽‍➡️', '🌪️', '🚀', '👨‍🦯‍➡️', '🏇🏼', '🛩️', '🚁', '🚂', '✈️', '🛳️', '🪂', '🚡', '💩', '🚕', '🚓', '🛒', '🐌', '💸']

_farm = ['🐂', '🐃', '🐄', '🐏', '🐖', '🐎', '🦃', '🫏', '🐇']

_safari = ['🦍', '🐅', '🐆', '🦒', '🦘', '🦓', '🐂']

_insects = ['🪰', '🦗', '🦟', '🐞', '🐜', '🐝', '🦋', '🐛']

_birds = ['🐦', '🦜', '🦆', '🦅', '🦉', '🦩', '🦚', '🦃']

_forest = ['🦌', '🦫', '🦦', '🦔', '🦇', '🦉', '🦅', '🦆', '🐢', '🐍', '🦎', '🐌']

_aquarium = ['🐠', '🐟', '🐢', '🏊🏻‍♀️', '🐡', '🦈', '🐙', '🐬', '🦭', '🐋', '🪼', '🦑', '🦐']

_scaled = {
    '0-5': '🪦',
    '5-10': '🌳',
    '10-20': '🐌',
    '20-30': '👨🏽‍🦽‍➡️',
    '30-40': '👩🏼‍🦼‍➡️',
    '40-50': '⛵',
    '50-60': '🚎',
    '60-70': '🐆',
    '70-80': '🌪️',
    '80-90': '🚀',
    '90-100': '🦸🏼'
}

_trains = {
    '0-20': '🚂',
    '20-40': '🚝',
    '40-60': '🚈',
    '60-80': '🚄',
    '80-100': '🚅'
    }






Track_stats = pd.read_csv(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\Track_stats.csv')
Spot_stats = pd.read_csv(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\Spot stats.csv')

color_modes = [
    'random colors',
    'random greys',
    'only-one-color',
    'greyscale LUT', 
    'jet LUT', 
    'brg LUT', 
    'hot LUT', 
    'gnuplot LUT', 
    'viridis LUT', 
    'rainbow LUT', 
    'turbo LUT', 
    'nipy-spectral LUT', 
    'gist-ncar LUT'
    ]



def _generate_random_color():
    """
    Generate a random color in hexadecimal format.

    """

    r = np.random.randint(0, 255)   # Red LED intensity
    g = np.random.randint(0, 255)   # Green LED intensity
    b = np.random.randint(0, 255)   # Blue LED intensity

    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def _generate_random_grey():
    """
    Generate a random grey color in hexadecimal format.

    """

    n = np.random.randint(0, 240)  # All LED intensities

    return '#{:02x}{:02x}{:02x}'.format(n, n, n)


def _make_q_cmap(elements, cmap):
    """
    Generate a qualitative colormap for a given list of elements.

    """

    n = len(elements)   # Number of elements in the dictionary
    if n == 0:          # Return an empty list if there are no elements
        return []       
    
    cmap = plt.get_cmap(cmap)                                   # Get the colormap
    colors = [mcolors.to_hex(cmap(i / n)) for i in range(n)]    # Generate a color for each element

    return colors


def _get_cmap(c_mode):
    """
    Get a colormap according to the selected color mode.

    """

    if c_mode == 'greyscale LUT':
        return plt.cm.gist_yarg
    elif c_mode == 'jet LUT':
        return plt.cm.jet
    elif c_mode == 'brg LUT':
        return plt.cm.brg
    elif c_mode == 'hot LUT':
        return plt.cm.hot
    elif c_mode == 'gnuplot LUT':
        return plt.cm.gnuplot
    elif c_mode == 'viridis LUT':
        return plt.cm.viridis
    elif c_mode == 'rainbow LUT':
        return plt.cm.rainbow
    elif c_mode == 'turbo LUT':
        return plt.cm.turbo
    elif c_mode == 'nipy_spectral LUT':
        return plt.cm.nipy_spectral
    elif c_mode == 'gist_ncar LUT':
        return plt.cm.gist_ncar
    else:
        return None


def _assign_marker(value, markers):
    """
    Qualitatively map a metric's percentile value to a symbol.

    """

    lut = []    # Initialize a list to store the ranges and corresponding symbols

    for key, val in markers.items():                # Iterate through the markers dictionary
        low, high = map(float, key.split('-'))      # Split the key into low and high values
        lut.append((low, high, val))                # Append the range and symbol to the list

    for low, high, symbol in lut:               # Return the symbol for the range that contains the given value
        if low <= value < high:                  # Check if the value falls within the range
            return symbol
    
    return list(markers.items())[-1][-1]            # Return the last symbol for thr 100th percentile (which is not included in the ranges)


def _get_markers(markers):
    """
    Get the markers according to the selected mode.

    """

    if markers == 'cell':
        return _cell
    elif markers == 'scaled':
        return _scaled
    elif markers == 'trains':
        return _trains
    elif markers == 'random':
        return _random
    elif markers == 'farm':
        return _farm
    elif markers == 'safari':
        return _safari
    elif markers == 'insects':
        return _insects
    elif markers == 'birds':
        return _birds
    elif markers == 'forest':
        return _forest
    elif markers == 'aquarium':
        return _aquarium



tracks = {
    "TRACK_ID": "Track ID",
    "CONDITION": "Condition",
    "REPLICATE": "Replicate",
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


def visualize_normalized_tracks(df, c_mode='random colors', only_one_color='black', lw=0.5, grid=True, backround='light', tooltip_face_color='w', tooltip_size=8, tooltip_face_alpha=0.85, tooltip_outline_width=0.75, tooltip_color='match', lut_metric='NET_DISTANCE'):

    # First sort the data and get groups of tracks.
    df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'], inplace=True)
    grouped = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])
    
    unique_tracks = df[['CONDITION', 'REPLICATE', 'TRACK_ID']].drop_duplicates().reset_index(drop=True)
    # For the random modes, pre-assign a color per track.
    if c_mode in ['random colors']:
        track_colors = [_generate_random_color() for _ in range(len(unique_tracks))]
    elif c_mode in ['random greys']:
        track_colors = [_generate_random_grey() for _ in range(len(unique_tracks))]
    else:
        track_colors = [None] * len(unique_tracks)  # Colors will be assigned via the LUT
    
    color_map_direct = dict(zip(unique_tracks['TRACK_ID'], track_colors))
    df['COLOR'] = df['TRACK_ID'].map(color_map_direct)
    
    # Normalize the positions for each track (shift tracks to start at 0,0)
    for (cond, repl, track_id), group in grouped:
        start_x = group['POSITION_X'].iloc[0]
        start_y = group['POSITION_Y'].iloc[0]
        df.loc[group.index, 'POSITION_X'] -= start_x
        df.loc[group.index, 'POSITION_Y'] -= start_y

    # Convert to polar coordinates.
    df['r'] = np.sqrt(df['POSITION_X']**2 + df['POSITION_Y']**2)
    df['theta'] = np.arctan2(df['POSITION_Y'], df['POSITION_X'])
    
    fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
    y_max = df['r'].max() * 1.1

    ax.set_title('Normalized Tracks')
    ax.set_ylim(0, y_max)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(grid)
    
    # If using a colormap based on a LUT metric, pre-compute aggregated values
    # Here we use the mean of the lut_metric per track. You can adjust the aggregation as needed.
    if c_mode not in ['random colors', 'random greys', 'only-one'] and lut_metric is not None:
        track_metric = df.groupby('TRACK_ID')[lut_metric].mean()
        metric_min = track_metric.min()
        metric_max = track_metric.max()
    else:
        track_metric = None

    # Plot all tracks
    for (cond, repl, track_id), group in grouped:
        # First, handle the modes that specify a direct color.
        if c_mode == 'random colors':
            color = group['COLOR'].iloc[0]
        elif c_mode == 'random greys':
            color = group['COLOR'].iloc[0]
        elif c_mode == 'only-one':
            color = only_one_color
        else:
            colormap = _get_cmap(c_mode)

            # If no explicit color was assigned and we have a colormap, then use LUT mapping.
            if colormap is not None and track_metric is not None:
                # Get the aggregated metric value for the track.
                val = track_metric.get(track_id, 0)
                # Normalize to [0, 1] (protect against division by zero)
                if metric_max > metric_min:
                    norm_val = (val - metric_min) / (metric_max - metric_min)
                else:
                    norm_val = 0.5
                color = colormap(norm_val)
            else:
                # Fallback if something goes wrong
                color = 'black'

        # Plot the track using computed color.
        ax.plot(group['theta'], group['r'], lw=lw, color=color)
    
    if backround == 'light':
        x_grid_color = 'grey'
        y_grid_color = 'lightgrey'
        ax.set_facecolor('white')
    elif backround == 'dark':
        x_grid_color = 'lightgrey'
        y_grid_color = 'grey'
        ax.set_facecolor('darkgrey')

    # Style the polar grid.
    for i, line in enumerate(ax.get_xgridlines()):
        if i % 2 == 0:
            line.set_linestyle('--')
            line.set_color(x_grid_color)
            line.set_linewidth(0.5)

    for line in ax.get_ygridlines():
        line.set_linestyle('-.')
        line.set_color(y_grid_color)
        line.set_linewidth(0.5)

    ax.text(0, df['r'].max() * 1.2, f'{int(round(y_max, -1))} µm',
            ha='center', va='center', fontsize=9, color='black')

    return plt.gcf()

# Example usage:
# visualize_normalized_tracks(
#     Spot_stats,
#     c_mode='color2',  # change to any of: 'greyscale','color1','color2',... or 'random colors', etc.
#     lw=0.5,
#     lut_metric='NET_DISTANCE'
# )
        



# ===================================================================================================================================================================================================================	


# MATPLOTLIB INTERACTIVE PLOT WITH A TOOLTIP - no html :(

def visualize_normalized_tracksl(df, df2, c_mode='', only_one_color='black', lw=0.5, grid=True, backround='light', tooltip_face_color='w', tooltip_size=8, tooltip_face_alpha=0.85, tooltip_outline_width=0.75, tooltip_color='match', lut_metric='NET_DISTANCE'):

    # Sort the data by specified columns
    df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'], inplace=True)
    grouped = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])
    
    unique_tracks = df[['CONDITION', 'REPLICATE', 'TRACK_ID']].drop_duplicates().reset_index(drop=True)
    # If using random colors or greys, assign colors beforehand
    if c_mode in ['random colors']:
        track_colors = [_generate_random_color() for _ in range(len(unique_tracks))]
    elif c_mode in ['random greys']:
        track_colors = [_generate_random_grey() for _ in range(len(unique_tracks))]
    else:
        track_colors = [None] * len(unique_tracks)  # Colors will be assigned via LUT mapping if needed
    
    color_map_direct = dict(zip(unique_tracks['TRACK_ID'], track_colors))
    df['COLOR'] = df['TRACK_ID'].map(color_map_direct)
    
    # Normalize each track to start at (0,0)
    for (cond, repl, track_id), group in grouped:
        start_x = group['POSITION_X'].iloc[0]
        start_y = group['POSITION_Y'].iloc[0]
        df.loc[group.index, 'POSITION_X'] -= start_x
        df.loc[group.index, 'POSITION_Y'] -= start_y

    # Convert to polar coordinates
    df['r'] = np.sqrt(df['POSITION_X']**2 + df['POSITION_Y']**2)
    df['theta'] = np.arctan2(df['POSITION_Y'], df['POSITION_X'])
    
    fig, ax = plt.subplots(figsize=(12.5, 9.5), subplot_kw={'projection': 'polar'})
    y_max = df['r'].max() * 1.1

    ax.set_title('Normalized Tracks')
    ax.set_ylim(0, y_max)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(grid)
    
    # Prepare for annotation (for interactive hover)
    lines_info = []

    # If using LUT mapping based on a metric, pre-calculate aggregated values per track.
    if c_mode not in ['random colors', 'random greys', 'only-one'] and lut_metric is not None:
        track_metric = df.groupby('TRACK_ID')[lut_metric].mean()
        metric_min = track_metric.min()
        metric_max = track_metric.max()
    else:
        track_metric = None

    # Plot tracks, record line information and assign color via direct mode or LUT mapping.
    for (cond, repl, track_id), group in grouped:
        # Determine the color for this track.
        if c_mode == 'random colors':
            color = group['COLOR'].iloc[0]
        elif c_mode == 'random greys':
            color = group['COLOR'].iloc[0]
        elif c_mode == 'only-one':
            color = only_one_color
        else:
            colormap = _get_cmap(c_mode)

            # Use the LUT mapping if colormap available and track_metric computed
            if colormap is not None and track_metric is not None:
                val = track_metric.get(track_id, 0)
                if metric_max > metric_min:
                    norm_val = (val - metric_min) / (metric_max - metric_min)
                else:
                    norm_val = 0.5
                color = colormap(norm_val)
            else:
                color = 'black'
        
        line, = ax.plot(group['theta'], group['r'], lw=lw, color=color)
        lines_info.append({
            'condition': cond,
            'replicate': repl,
            'track_id': track_id,
            'color': color,
            'line': line
        })
    
    # Define helper functions for interactive annotation.
    def update_annot(info, event):
        text = f"Condition: {info['condition']}\nReplicate: {info['replicate']}\nTrack {info['track_id']}"
        annot.set_text(text)
        annot.xy = (event.xdata, event.ydata)
        # If tooltip_color is set to 'match', use the track color, otherwise use the provided tooltip_color.
        edge_color = info['color'] if tooltip_color == 'match' else tooltip_color
        annot.get_bbox_patch().set_edgecolor(edge_color)
        annot.get_bbox_patch().set_alpha(tooltip_face_alpha)

    def hover(event):
        if event.inaxes == ax:
            for info in lines_info:
                cont, _ = info['line'].contains(event)
                if cont:
                    update_annot(info, event)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    # Create annotation object for tooltips
    annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                        bbox=dict(fc=tooltip_face_color, lw=tooltip_outline_width),
                        fontsize=tooltip_size)
    annot.set_visible(False)

    # Connect the hover event
    fig.canvas.mpl_connect("motion_notify_event", hover)

    # Set grid and background styles.
    if backround == 'light':
        x_grid_color = 'grey'
        y_grid_color = 'lightgrey'
        ax.set_facecolor('white')
    elif backround == 'dark':
        x_grid_color = 'lightgrey'
        y_grid_color = 'grey'
        ax.set_facecolor('darkgrey')

    for i, line in enumerate(ax.get_xgridlines()):
        if i % 2 == 0:
            line.set_linestyle('--')
            line.set_color(x_grid_color)
            line.set_linewidth(0.5)

    for line in ax.get_ygridlines():
        line.set_linestyle('-.')
        line.set_color(y_grid_color)
        line.set_linewidth(0.5)

    ax.text(0, df['r'].max() * 1.2, f'{int(round(y_max, -1))} µm',
            ha='center', va='center', fontsize=9, color='black')
    
    mpld3.save_html(fig, 'interactive_fig.html') #save to html here

    plt.show()

# # Example usage:
# visualize_normalized_tracksl(
#     Spot_stats,
#     Track_stats,
#     c_mode='random greys',  # Try other modes such as 'greyscale','color1','color2', etc.
#     lw=0.5,
#     lut_metric='NET_DISTANCE'
# )





        
        

# ===================================================================================================================================================================================================================

# PLOTLY INTERACTIVE PLOT WITH A TOOLTIP - html :), but ugly :(

def visualize_normalized_tracks_interactive(df, c_mode='random colors', only_one_color='black', lw=0.5, grid=True,
                                              backround='light', tooltip_size=12, lut_metric='NET_DISTANCE'):
    # Sort and group data
    df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'], inplace=True)
    grouped = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])
    
    # Prepare unique track IDs and assign colors for direct color modes
    unique_tracks = df[['CONDITION', 'REPLICATE', 'TRACK_ID']].drop_duplicates().reset_index(drop=True)
    if c_mode in ['random colors']:
        track_colors = [_generate_random_color() for _ in range(len(unique_tracks))]
    elif c_mode in ['random greys']:
        track_colors = [_generate_random_grey() for _ in range(len(unique_tracks))]
    else:
        track_colors = [None] * len(unique_tracks)  # Colors will be assigned via LUT mapping if applicable

    # Save the direct mapping in a new column if needed.
    color_map_direct = dict(zip(unique_tracks['TRACK_ID'], track_colors))
    df['COLOR'] = df['TRACK_ID'].map(color_map_direct)
    
    # Compute the LUT metric aggregation if needed
    if c_mode not in ['random colors', 'random greys', 'only-one']:
        # Use mean as aggregation. Adjust if needed.
        track_metric = df.groupby('TRACK_ID')[lut_metric].mean()
        metric_min = track_metric.min()
        metric_max = track_metric.max()
        colormap = _get_cmap(c_mode)
    else:
        track_metric = None

    # Normalize each track's positions to start at (0, 0)
    for (cond, repl, track_id), group in grouped:
        start_x = group['POSITION_X'].iloc[0]
        start_y = group['POSITION_Y'].iloc[0]
        df.loc[group.index, 'POSITION_X'] -= start_x
        df.loc[group.index, 'POSITION_Y'] -= start_y

    # Convert to polar coordinates (theta in radians & degrees)
    df['r'] = np.sqrt(df['POSITION_X']**2 + df['POSITION_Y']**2)
    df['theta'] = np.arctan2(df['POSITION_Y'], df['POSITION_X'])
    df['theta_deg'] = np.degrees(df['theta'])
    
    # Determine maximum radius for layout
    y_max = df['r'].max()
    y_max_r = y_max * 1.12
    y_max_a = y_max * 1.1

    # Create Plotly figure
    fig = go.Figure()

    # Loop over tracks and add a polar trace for each
    for (cond, repl, track_id), group in df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID']):
        # Determine the color for this track
        if c_mode == 'random colors':
            color = group['COLOR'].iloc[0]
        elif c_mode == 'random greys':
            color = group['COLOR'].iloc[0]
        elif c_mode == 'only-one':
            color = only_one_color
        else:
            # LUT mapping: use aggregated value and colormap if available.
            if colormap is not None and track_metric is not None:
                val = track_metric.get(track_id, 0)
                if metric_max > metric_min:
                    norm_val = (val - metric_min) / (metric_max - metric_min)
                else:
                    norm_val = 0.5
                # Convert the RGBA output of the colormap to hex
                color = mcolors.to_hex(colormap(norm_val))
            else:
                color = 'black'

        hover_text = (f"Condition: {cond}<br>"
                      f"Replicate: {repl}<br>"
                      f"Track: {track_id}")
        
        fig.add_trace(go.Scatterpolar(
            r=group['r'],
            theta=group['theta_deg'],
            mode='lines',
            line=dict(color=color, width=lw * 2),  # Adjust line width if needed
            name=f"Track {track_id}",
            hovertemplate=hover_text + "<extra></extra>"
        ))

    # Update layout
    fig.update_layout(
        title="Normalized Tracks",
        title_x=0.5,
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                range=[0, y_max_r],
                gridcolor='lightgrey',
                showticklabels=False,
                ticks=''
            ),
            angularaxis=dict(
                gridcolor='grey',
                showticklabels=False,
                ticks=''
            ),
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Add annotation for maximum radius, if desired.
    fig.add_annotation(
        text=f'{int(round(y_max_a, -1))} µm',
        x=0.74, y=0.5, xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=tooltip_size, color="black")
    )

    # Write the figure to an HTML file (adjust the path as needed)
    # fig.write_html(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\code files\cache_\normalized_tracks_radial.html', auto_open=True)

    return fig  # Return the figure object for further manipulation if needed

# # Example usage:
# visualize_normalized_tracks_interactive(
#     Spot_stats,
#     c_mode='random colors',  # Use a colormap mode e.g. 'color1' (jet)
#     lw=0.5,
#     lut_metric='NET_DISTANCE'
#     )






# Main plotting function
def Visualize_normalized_tracks_plotly(Spots_df, Tracks_df, condition='all', replicate='all', 
                                        c_mode='random colors', only_one_color='black', 
                                        lut_scaling_metric='NET_DISTANCE', let_me_look_at_these=None, 
                                        background='dark', smoothing_index=6, lw=1, marker_size=5, 
                                        end_track_markers=False, markers='circle-open', 
                                        I_just_wanna_be_normal=True, metric_dictionary=None):
    
    let_me_look_at_these = list(let_me_look_at_these)
    if 'level_0' in Tracks_df.columns:
        Tracks_df.drop(columns=['level_0'], inplace=True)
    
    Tracks_df.reset_index(drop=False, inplace=True)

    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except (ValueError, TypeError):
            pass
        try:
            replicate = int(replicate)
        except (ValueError, TypeError):
            pass

    if condition == 'all':
        Spots_df = Spots_df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])
    elif condition != 'all' and replicate == 'all':
        Spots_df = Spots_df[Spots_df['CONDITION'] == condition].sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])
    elif condition != 'all' and replicate != 'all':
        Spots_df = Spots_df[(Spots_df['CONDITION'] == condition) & (Spots_df['REPLICATE'] == replicate)].sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])

    # Set colors based on chosen mode
    if c_mode in ['random colors', 'random greys', 'only-one-color']:
        colormap = None
        if c_mode == 'random colors':
            track_colors = [_generate_random_color() for _ in range(len(Tracks_df))]
        elif c_mode == 'random greys':
            track_colors = [_generate_random_grey() for _ in range(len(Tracks_df))]
        else:
            track_colors = [only_one_color for _ in range(len(Tracks_df))]
        
        color_map_direct = dict(zip(Tracks_df['TRACK_ID'], track_colors))
        Tracks_df['COLOR'] = Tracks_df['TRACK_ID'].map(color_map_direct)
    else:
        colormap = _get_cmap(c_mode)
        metric_min = Spots_df[lut_scaling_metric].min()
        metric_max = Spots_df[lut_scaling_metric].max()

    min_track_length = Tracks_df['TRACK_LENGTH'].min()
    max_track_length = Tracks_df['TRACK_LENGTH'].max()

    Spots_grouped = Spots_df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])
    Tracks_df.set_index(['CONDITION', 'REPLICATE', 'TRACK_ID'], inplace=True)

    # Normalize each track's positions to start at (0, 0)
    for (cond, repl, track), group_df in Spots_grouped:
        start_x = group_df['POSITION_X'].iloc[0]
        start_y = group_df['POSITION_Y'].iloc[0]
        Spots_df.loc[group_df.index, 'POSITION_X'] -= start_x
        Spots_df.loc[group_df.index, 'POSITION_Y'] -= start_y

    # Convert to polar coordinates (theta in radians & degrees)
    Spots_df['r'] = np.sqrt(Spots_df['POSITION_X']**2 + Spots_df['POSITION_Y']**2)
    Spots_df['theta'] = np.arctan2(Spots_df['POSITION_Y'], Spots_df['POSITION_X'])
    Spots_df['theta_deg'] = np.degrees(Spots_df['theta'])
    
    # Determine maximum radius for layout
    y_max = Spots_df['r'].max()
    y_max_r = y_max * 1.12
    y_max_a = y_max * 1.1

    # Create Plotly polar figure
    fig = go.Figure()

    # Re-group here, since Spots_df has been modified
    Spots_grouped = Spots_df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])

    # Loop over tracks and add a polar trace for each
    for (cond, repl, track), group_df in Spots_grouped:

        track_row = Tracks_df.loc[(cond, repl, track)]
        track_row['CONDITION'] = cond
        track_row['REPLICATE'] = repl
        track_row['TRACK_ID'] = track

        if colormap is not None:
            # Normalize the metric for color mapping
            norm = plt.Normalize(metric_min, metric_max)
            color = colormap(norm(track_row[lut_scaling_metric]))
            group_df['COLOR'] = color if isinstance(color, str) else [mcolors.to_hex(color)] * len(group_df)
        elif c_mode in ['random colors', 'random greys']:
            group_df['COLOR'] = track_row['COLOR']
        elif c_mode == 'only-one-color':
            group_df['COLOR'] = only_one_color

        # Apply smoothing if required
        if isinstance(smoothing_index, (int, float)) and smoothing_index > 1:
            group_df['POSITION_X'] = group_df['POSITION_X'].rolling(window=smoothing_index, min_periods=1).mean()
            group_df['POSITION_Y'] = group_df['POSITION_Y'].rolling(window=smoothing_index, min_periods=1).mean()
        else:
            group_df['POSITION_X'] = group_df['POSITION_X']
            group_df['POSITION_Y'] = group_df['POSITION_Y']

        # Create hover text if extra metrics are provided
        hover_text = ""
        if len(let_me_look_at_these) > 0:
            hover_dict = {}
            for metric in let_me_look_at_these:
                hover_dict[metric_dictionary[metric]] = track_row[metric]
            hover_text = "<br>".join(f"{key}: {value}" for key, value in hover_dict.items())

        # Add the polar trace
        fig.add_trace(go.Scatterpolar(
            r=group_df['r'],
            theta=group_df['theta_deg'],
            mode='lines',
            line=dict(color=group_df['COLOR'].iloc[0], width=lw * 2),
            name=f"Track {track}",
            hovertemplate=hover_text + "<extra></extra>"
        ))
        
        # Optionally add markers for the end of each track within the polar coordinate system
        if end_track_markers:
            if I_just_wanna_be_normal:
                fig.add_trace(go.Scatterpolar(
                    r=[group_df['r'].iloc[-1]],
                    theta=[group_df['theta_deg'].iloc[-1]],
                    mode='markers',
                    marker=dict(
                        symbol=markers,
                        size=marker_size,
                        color=group_df['COLOR'].iloc[0],
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                if markers == 'cell':
                    marker = _cell  # Ensure you define _cell if needed
                elif markers in ['scaled', 'trains']:
                    markers_ = _get_markers(markers)
                    percentile_value = ((track_row['TRACK_LENGTH'] - min_track_length) / (max_track_length - min_track_length)) * 100
                    marker = _assign_marker(percentile_value, markers_)
                elif markers in ['random','farm','safari','insects','birds','forest','aquarium']:
                    markers_ = _get_markers(markers)
                    marker = np.random.choice(markers_)
                else:
                    marker = ''

                fig.add_trace(go.Scatterpolar(
                    r=[group_df['r'].iloc[-1]],
                    theta=[group_df['theta_deg'].iloc[-1]],
                    mode='text',
                    text=marker,
                    textposition='middle center',
                    textfont=dict(size=marker_size),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Update layout for a polar plot
    fig.update_layout(
        title="Normalized Tracks",
        title_x=0.5,
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                range=[0, y_max_r],
                gridcolor='lightgrey',
                showticklabels=False,
                ticks=''
            ),
            angularaxis=dict(
                gridcolor='grey',
                showticklabels=False,
                ticks=''
            ),
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Add annotation for maximum radius if desired
    fig.add_annotation(
        text=f'{int(round(y_max_a, -1))} µm',
        x=0.74, y=0.5, xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=12, color="black")
    )

    return fig

    # Write to file and optionally auto open in a browser
    fig.write_html(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\code files\cache_\normalized_tracks_radial.html', auto_open=True)

# Example call (replace Spot_stats, Track_stats, and tracks with your actual data)
Visualize_normalized_tracks_plotly(
    Spots_df=Spot_stats,            # Your DataFrame with spot data
    Tracks_df=Track_stats,          # Your DataFrame with track data
    condition='all',
    replicate='all',
    c_mode='random colors',         # Color mode: e.g., 'random colors'
    only_one_color='black',
    lw=0.5,
    marker_size=5,
    end_track_markers=False,
    markers='circle-open',
    I_just_wanna_be_normal=True,
    metric_dictionary=tracks,       # Your dictionary mapping metrics
    let_me_look_at_these=['TRACK_ID', 'REPLICATE', 'CONDITION', 'NET_DISTANCE', 'TRACK_LENGTH', 'CONFINEMENT_RATIO']
)