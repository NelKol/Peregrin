import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

Track_stats = pd.read_csv(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\Track_stats.csv')
Spot_stats = pd.read_csv(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\Spot_stats.csv')

def generate_random_color():
    r = np.random.randint(0, 255)  # Random value for Red
    g = np.random.randint(0, 255)  # Random value for Green
    b = np.random.randint(0, 255)  # Random value for Blue
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)
def generate_random_grey():
    n = np.random.randint(0, 255)  # Random value for Grey
    return '#{:02x}{:02x}{:02x}'.format(n, n, n)

def visualize_normalized_tracks_radial(df, df2, color_scale='color', c_mode='random', only_one_color='black', lut_mode=False, lw=0.5, grid=True, backround='light', tooltip_face_color='w', tooltip_size=8, tooltip_face_alpha=0.85, tooltip_outline_width=0.75, tooltip_color='match', lut_metric='NET_DISTANCE'):
    df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'], inplace=True)
    grouped = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])

    unique_tracks = df[['CONDITION', 'REPLICATE', 'TRACK_ID']].drop_duplicates().reset_index(drop=True)
    track_colors = [generate_random_color() for _ in range(len(unique_tracks))] if color_scale == 'color' else [generate_random_grey() for _ in range(len(unique_tracks))]
    color_map = dict(zip(unique_tracks['TRACK_ID'], track_colors))
    df['COLOR'] = df['TRACK_ID'].map(color_map)

    # Normalize each track to its starting point
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

    lines_info = []

    # Plot all tracks
    for (cond, repl, track_id), group in grouped:
        if color_scale == 'color':
            if lut_mode:
                pass
            elif lut_mode == False:
                if c_mode == 'random':
                    color = group['COLOR'].iloc[0]
                elif c_mode == 'only_one':
                    color = only_one_color
        elif color_scale == 'bnw':
            if lut_mode:
                pass
            elif lut_mode == False:
                if c_mode == 'random':
                    color = group['COLOR'].iloc[0]
                elif c_mode == 'only_one':
                    color = only_one_color


        line, = ax.plot(group['theta'], group['r'], lw=lw, color=color)
        # lines_info.append({
        #     'condition': cond,
        #     'replicate': repl,
        #     'track_id': track_id,
        #     'color': color,
        #     'line': line,
        # })


    # def update_annot(info, event):
    #     text = f"Condition: {info['condition']}\nReplicate: {info['replicate']}\nTrack {info['track_id']}"
    #     annot.set_text(text)
    #     annot.xy = (event.xdata, event.ydata)
    #     annot.get_bbox_patch().set_edgecolor(info['color'] if tooltip_color == 'match' else tooltip_color)
    #     annot.get_bbox_patch().set_alpha(tooltip_face_alpha)

    # def hover(event):
    #     if event.inaxes == ax:
    #         for info in lines_info:
    #             cont, _ = info['line'].contains(event)
    #             if cont:
    #                 update_annot(info, event)
    #                 annot.set_visible(True)
    #                 fig.canvas.draw_idle()
    #                 return
    #         if annot.get_visible():
    #             annot.set_visible(False)
    #             fig.canvas.draw_idle()

    # annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
    #                     bbox=dict(fc=tooltip_face_color, lw=tooltip_outline_width), fontsize=tooltip_size)
    # annot.set_visible(False)

    # fig.canvas.mpl_connect("motion_notify_event", hover)

    if backround == 'light':
        x_grid_color = 'grey'
        y_grid_color = 'lightgrey'
        ax.set_facecolor('white')
    elif backround == 'dark':
        x_grid_color = 'lightgrey'
        y_grid_color = 'grey'
        ax.set_facecolor('darkgrey')

    # Style the polar grid
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

# Example usage
visualize_normalized_tracks_radial(
    Spot_stats,
    Track_stats,
    lw=0.5,
    lut_metric='NET_DISTANCE'
)




# Functions to generate colors
def generate_random_color():
    r = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    b = np.random.randint(0, 255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def generate_random_grey():
    n = np.random.randint(0, 255)
    return '#{:02x}{:02x}{:02x}'.format(n, n, n)


def visualize_normalized_tracks_radial(df, df2, color_scale='color', c_mode='random', only_one_color='black', lut_mode=False, lw=0.5, grid=True, backround='light', tooltip_size=12, lut_metric='NET_DISTANCE'):
    # Sort and group
    df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'], inplace=True)
    grouped = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])
    
    # Assign colors for unique tracks
    unique_tracks = df[['CONDITION', 'REPLICATE', 'TRACK_ID']].drop_duplicates().reset_index(drop=True)
    if color_scale == 'color':
        track_colors = [generate_random_color() for _ in range(len(unique_tracks))]
    else:  # bnw mode
        track_colors = [generate_random_grey() for _ in range(len(unique_tracks))]
    color_map = dict(zip(unique_tracks['TRACK_ID'], track_colors))
    df['COLOR'] = df['TRACK_ID'].map(color_map)
    
    # Normalize each track to its starting point
    for (cond, repl, track_id), group in grouped:
        start_x = group['POSITION_X'].iloc[0]
        start_y = group['POSITION_Y'].iloc[0]
        df.loc[group.index, 'POSITION_X'] -= start_x
        df.loc[group.index, 'POSITION_Y'] -= start_y

    # Convert to polar coordinates
    df['r'] = np.sqrt(df['POSITION_X']**2 + df['POSITION_Y']**2)
    df['theta'] = np.arctan2(df['POSITION_Y'], df['POSITION_X'])
    # Convert theta from radians to degrees (Plotly uses degrees by default)
    df['theta_deg'] = np.degrees(df['theta'])
    
    # Determine the maximum radius for setting the radial axis limit
    y_max = df['r'].max()
    y_max_r = y_max * 1.12
    y_max_a = y_max * 1.1
    

    # Create Plotly figure
    fig = go.Figure()

    # Add each track as a separate polar trace with custom hover text
    for (cond, repl, track_id), group in df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID']):
        # Decide trace color
        if color_scale in ['color', 'bnw']:
            if not lut_mode:
                if c_mode == 'random':
                    trace_color = group['COLOR'].iloc[0]
                elif c_mode == 'only_one':
                    trace_color = only_one_color
        else:
            trace_color = only_one_color

        hover_text = (f"Condition: {cond}<br>"
                      f"Replicate: {repl}<br>"
                      f"Track: {track_id}")

        fig.add_trace(go.Scatterpolar(
            r=group['r'],
            theta=group['theta_deg'],
            mode='lines',
            line=dict(color=trace_color, width=lw*2),  # adjust line width if needed
            name=f"Track {track_id}",
            hovertemplate=hover_text + "<extra></extra>"
        ))

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

    # Optionally, add a text annotation for the maximum radius
    fig.add_annotation(
        text=f'{int(round(y_max_a, -1))} µm',
        x=0.74, y=0.5, xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=tooltip_size, color="black")
    )

    fig.write_html(r'C:\Users\modri\Desktop\python\Peregrin\Peregrin\code files\cache_\normalized_tracks_radial.html', auto_open=False)

# Example usage
visualize_normalized_tracks_radial(
    Spot_stats,
    Track_stats,
    lw=0.5,
    lut_metric='NET_DISTANCE'
)
