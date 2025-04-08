import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import plotly.graph_objects as go

Track_stats = pd.read_csv(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\Track_stats.csv')
Spot_stats = pd.read_csv(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\Spot_stats.csv')

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



def get_colormap(c_mode):
    if c_mode == 'greyscale':
        return plt.cm.gist_yarg
    elif c_mode == 'color1':
        return plt.cm.jet
    elif c_mode == 'color2':
        return plt.cm.brg
    elif c_mode == 'color3':
        return plt.cm.hot
    elif c_mode == 'color4':
        return plt.cm.gnuplot
    elif c_mode == 'color5':
        return plt.cm.viridis
    elif c_mode == 'color6':
        return plt.cm.rainbow
    elif c_mode == 'color7':
        return plt.cm.turbo
    elif c_mode == 'color8':
        return plt.cm.nipy_spectral
    elif c_mode == 'color9':
        return plt.cm.gist_ncar
    else:
        return None


def generate_random_color():
    r = np.random.randint(0, 255)  # Random value for Red
    g = np.random.randint(0, 255)  # Random value for Green
    b = np.random.randint(0, 255)  # Random value for Blue
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def generate_random_grey():
    n = np.random.randint(0, 240)  # Random value for Grey
    return '#{:02x}{:02x}{:02x}'.format(n, n, n)




def visualize_normalized_tracks(df, c_mode='random colors', only_one_color='black', lw=0.5, grid=True, backround='light', tooltip_face_color='w', tooltip_size=8, tooltip_face_alpha=0.85, tooltip_outline_width=0.75, tooltip_color='match', lut_metric='NET_DISTANCE'):

    # First sort the data and get groups of tracks.
    df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'], inplace=True)
    grouped = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])
    
    unique_tracks = df[['CONDITION', 'REPLICATE', 'TRACK_ID']].drop_duplicates().reset_index(drop=True)
    # For the random modes, pre-assign a color per track.
    if c_mode in ['random colors']:
        track_colors = [generate_random_color() for _ in range(len(unique_tracks))]
    elif c_mode in ['random greys']:
        track_colors = [generate_random_grey() for _ in range(len(unique_tracks))]
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
            colormap = get_colormap(c_mode)

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
        track_colors = [generate_random_color() for _ in range(len(unique_tracks))]
    elif c_mode in ['random greys']:
        track_colors = [generate_random_grey() for _ in range(len(unique_tracks))]
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
            colormap = get_colormap(c_mode)

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

    plt.show()

# Example usage:
visualize_normalized_tracksl(
    Spot_stats,
    Track_stats,
    c_mode='random greys',  # Try other modes such as 'greyscale','color1','color2', etc.
    lw=0.5,
    lut_metric='NET_DISTANCE'
)





        
        

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
        track_colors = [generate_random_color() for _ in range(len(unique_tracks))]
    elif c_mode in ['random greys']:
        track_colors = [generate_random_grey() for _ in range(len(unique_tracks))]
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
        colormap = get_colormap(c_mode)
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
    fig.write_html(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\code files\cache_\normalized_tracks_radial.html', auto_open=False)

# Example usage:
visualize_normalized_tracks_interactive(
    Spot_stats,
    c_mode='color1',  # Use a colormap mode e.g. 'color1' (jet)
    lw=0.5,
    lut_metric='NET_DISTANCE'
    )
