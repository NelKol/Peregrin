import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
from peregrin.scripts import PlotParams
import seaborn as sns
from itertools import combinations
import altair as alt
import fitz
import os



def get_q_cmap_list(elements, cmap):
    n = len(elements)
    if n == 0:
        return []
    cmap = plt.get_cmap(cmap)
    
    # Generate colors by sampling evenly from the colormap.
    colors = [mcolors.to_hex(cmap(i / n)) for i in range(n)]
    return colors

def generate_random_color():
    r = np.random.randint(0, 255)  # Random value for Red
    g = np.random.randint(0, 255)  # Random value for Green
    b = np.random.randint(0, 255)  # Random value for Blue
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def generate_random_grey():
    n = np.random.randint(0, 240)  # Random value for Grey
    return '#{:02x}{:02x}{:02x}'.format(n, n, n)

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




def migration_directions_with_kde_plus_mean(df, metric, subject, scaling_metric, cmap_normalization_metric, cmap, threshold, title_size2):	

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    df_mean_direction = df[metric]

    # Prepare for KDE plot
    x_kde = np.cos(df_mean_direction)
    y_kde = np.sin(df_mean_direction)
    kde = gaussian_kde([x_kde, y_kde])

    # Define the grid for evaluation
    theta_kde = np.linspace(0, 2 * np.pi, 360)
    x_grid = np.cos(theta_kde)
    y_grid = np.sin(theta_kde)

    # Evaluate the KDE on the grid and normalize
    z_kde = kde.evaluate([x_grid, y_grid])
    z_kde = z_kde / z_kde.max() * 0.5  # Normalize to fit within the radial limit

    # Calculate the mean direction
    mean_direction = np.arctan2(np.mean(y_kde), np.mean(x_kde))

    # Start plotting
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})

    # Plot KDE
    ax.plot(theta_kde, z_kde, label='Circular KDE', color='None', zorder=5)
    ax.fill(theta_kde, z_kde, alpha=0.25, color='#1b5a9e', zorder=5)

    # Directional Arrows
    scaling_max = df[scaling_metric].max()

    # Normalization of the color map
    if cmap_normalization_metric == None:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        normalization_min = df[cmap_normalization_metric].min()
        normalization_max = df[cmap_normalization_metric].max()
        norm = mcolors.Normalize(vmin=normalization_min, vmax=normalization_max)

    # Row itteration
    for _, row in df.iterrows():
        scaling_metrics = row[scaling_metric]
        mean_direction_rad = row[metric]
        arrow_length = scaling_metrics / scaling_max

        if cmap_normalization_metric == None:
            color = cmap(norm(arrow_length))
        else:
            color = cmap(norm(row[cmap_normalization_metric]))

        if arrow_length == 0:
            continue  # Skip if the arrow length is zero

        # Dynamically adjust the head size based on arrow_length
        scaling_factor = 1 / arrow_length if arrow_length != 0 else 1
        head_width = 0.011 * scaling_factor
        head_length = 0.013

        ax.arrow(mean_direction_rad, 0, 0, arrow_length, color=color, linewidth=0.75, 
                head_width=head_width, head_length=head_length, zorder=4)

    # Plot the dashed line in the mean direction
    ax.plot([mean_direction, mean_direction], [0, 1], linestyle='--', color='darkslateblue', alpha=0.93, linewidth=2.5, zorder=6)

    # Hide the polar plot frame (spines) but keep the grid visible
    ax.spines['polar'].set_visible(False)
    # Customize grid lines (if needed)
    ax.grid(True, 'major', color='#C6C6C6', linestyle='-', linewidth=0.5, zorder=0)

    # Access and customize the radial grid lines
    radial_lines = ax.get_xgridlines()
    for i, line in enumerate(radial_lines):
        if i % 2 == 0:  # Customize every other radial grid line
            line.set_linestyle('--')
            line.set_color('#E6E6E6')
            line.set_linewidth(0.5)

    radial_lines = ax.get_ygridlines()
    for i, line in enumerate(radial_lines):
        line.set_linestyle('--')
        line.set_color('#E6E6E6')
        line.set_linewidth(0.5)

    # Customize the appearance of the polar plot
    ax.set_title(f'Mean Direction of Travel\nwith Kernel Density Estimate\n$\it{{{subject}}}$', fontsize=title_size2)
    ax.set_yticklabels([])  # Remove radial labels
    ax.set_xticklabels([])  # Remove angular labels

    return plt.gcf()

def donut(df, ax, outer_radius, inner_radius, kde_bw):
    # Extend the data circularly to account for wrap-around at 0 and 2*pi
    extended_data = np.concatenate([df - 2 * np.pi, df, df + 2 * np.pi])

    # Create a grid of theta values (angles)
    theta_grid = np.linspace(0, 2 * np.pi, 360)  # 360 points over full circle
    
    # Create a grid of radii
    r_grid = np.linspace(inner_radius, outer_radius, 100)  # Radius from inner to outer edge
    
    # Compute KDE values for the extended data
    kde = gaussian_kde(extended_data, bw_method=kde_bw)
    kde_values = kde.evaluate(theta_grid)  # Evaluate KDE on the regular theta grid
    
    # Repeat KDE values across radii to create the heatmap data
    kde_values = np.tile(kde_values, (r_grid.size, 1))
    
    # Normalize KDE values for consistent color mapping
    norm = Normalize(vmin=kde_values.min(), vmax=kde_values.max())
    
    # Create the meshgrid for the polar plot
    theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)
    
    # Remove polar grid lines and labels
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)  # Hide the outer frame

    return theta_mesh, r_mesh, kde_values, norm

def df_gaussian_donut(df, metric, subject, heatmap, weight, threshold, title_size2, label_size, figtext_color, figtext_size):

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    diameter=2
    width_ratio=0.3
    kde_bw=0.1

    df=df[metric]

    # Calculate radius and width from the diameter
    outer_radius = diameter / 2
    width = width_ratio * outer_radius
    inner_radius = outer_radius - width
    
    theta_mesh, r_mesh, kde_values, norm = donut(df, ax, outer_radius, inner_radius, kde_bw)
    
    # Set title and figure text
    ax.set_title(f'Heatmap of Migration Direction\n({subject})', pad=20, ha='center', fontsize=title_size2)
    
    # Add a colorbar
    cbar = plt.colorbar(ax.pcolormesh(theta_mesh, r_mesh, kde_values, shading='gouraud', cmap=heatmap, norm=norm), ax=ax, fraction=0.04, orientation='horizontal', pad=0.1)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)  # Remove outline
    
    # Add min and max labels below the colorbar
    cbar.ax.text(0.05, -0.4, 'min', va='center', ha='center', color='black', transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(0.95, -0.4, 'max', va='center', ha='center', color='black', transform=cbar.ax.transAxes, fontsize=9)

    # Add the density label below the min and max labels
    cbar.set_label('Density', labelpad=10, fontsize=label_size)
    
    # if weight == None:
    #     plt.savefig(f'04a_Plot_donut_heatmap-migration_direction_{subject}{threshold}.png', dpi=300)
    # else:
    #     weight = 'weighted by' + weight
    #     plt.figtext(0.515, 0.01, f'{weight}', ha='center', color=figtext_color, fontsize=figtext_size)
    #     plt.savefig(f'04a_Plot_donut_heatmap-migration_direction_{subject}{weight}{threshold}.png', dpi=300)

    # plt.show()

    return plt.gcf()

    # try to normalize the heatmap colors to the absolute 0 (not min of the kde values) and to the max of the kde values






def track_visuals(df2, c_mode, grid, lut_metric, title_size=12):
    
    fig, ax = plt.subplots(figsize=(13, 10))

    unique_tracks = df2[['CONDITION', 'REPLICATE', 'TRACK_ID', lut_metric]].drop_duplicates()

    lut_norm_df = df2[['TRACK_ID', lut_metric]].drop_duplicates()

    # Normalize the NET_DISTANCE to a 0-1 range
    lut_min = lut_norm_df[lut_metric].min()
    lut_max = lut_norm_df[lut_metric].max()
    norm = plt.Normalize(vmin=lut_min, vmax=lut_max)
    if c_mode == 'greyscale':
        colormap = plt.cm.gist_yarg
        grid_color = 'gainsboro'
        face_color = 'None'
        grid_alpha = 0.5
        if grid:
            grid_lines = '-.'
        else:
            grid_lines = 'None'
    else:
        colormap = get_colormap(c_mode)
            
        grid_color = 'silver'
        face_color = 'darkgrey'
        grid_alpha = 0.75
        if grid:
            grid_lines = '-.'
        else:
            grid_lines = 'None'

    # Create a dictionary to store the color for each track based on its confinement ratio
    track_colors = {}

    for track_ids_all in unique_tracks['TRACK_ID'].drop_duplicates():
        ratio = lut_norm_df[lut_norm_df['TRACK_ID'] == track_ids_all][lut_metric].values[0]
        track_colors[f'all all {track_ids_all}'] = colormap(norm(ratio))

    for condition in unique_tracks['CONDITION'].drop_duplicates():
        condition_tracks = unique_tracks[unique_tracks['CONDITION'] == condition]
        for track_ids_conditions in condition_tracks['TRACK_ID'].drop_duplicates():
            ratio = lut_norm_df[lut_norm_df['TRACK_ID'] == track_ids_conditions][lut_metric].values[0]
            track_colors[f'{condition} all {track_ids_conditions}'] = colormap(norm(ratio))
        
        for replicate in condition_tracks['REPLICATE'].drop_duplicates():
            replicate_tracks = condition_tracks[condition_tracks['REPLICATE'] == replicate]
            for track_id_replicates in replicate_tracks['TRACK_ID'].drop_duplicates():
                ratio = lut_norm_df[lut_norm_df['TRACK_ID'] == track_id_replicates][lut_metric].values[0]
                track_colors[f'{condition} {replicate} {track_id_replicates}'] = colormap(norm(ratio))

    # Set up the plot limits
    # Define the desired dimensions in microns
    microns_per_pixel = 0.7381885238402274 # for 10x lens
    x_min, x_max = 0, (1600 * microns_per_pixel)
    y_min, y_max = 0, (1200 * microns_per_pixel)

    ax.set_aspect('1', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Position X [microns]')
    ax.set_ylabel('Position Y [microns]')
    ax.set_title('Track Visualization', fontsize=title_size)
    ax.set_facecolor(face_color)
    ax.grid(True, which='both', axis='both', color=grid_color, linestyle=grid_lines, linewidth=1, alpha=grid_alpha)

    # Manually set the major tick locations and labels
    x_ticks_major = np.arange(x_min, x_max, 200)  # Adjust the step size as needed
    y_ticks_major = np.arange(y_min, y_max, 200)  # Adjust the step size as needed
    ax.set_xticks(x_ticks_major)
    ax.set_yticks(y_ticks_major)
    ax.set_xticklabels([f'{tick:.0f}' for tick in x_ticks_major])
    ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks_major])

    # Enable minor ticks and set their locations
    ax.minorticks_on()
    x_ticks_minor = np.arange(x_min, x_max, 50)  # Minor ticks every 50 microns
    y_ticks_minor = np.arange(y_min, y_max, 50)  # Minor ticks every 50 microns
    ax.set_xticks(x_ticks_minor, minor=True)
    ax.set_yticks(y_ticks_minor, minor=True)
    ax.tick_params(axis='both', which='major', labelsize=8)

    return fig, ax, unique_tracks, track_colors, norm, colormap

def visualize_tracks(df, df2, condition='all', replicate='all', c_mode='color1', grid=True, smoothing_index=0, lut_metric='NET_DISTANCE', lw=1, arrowsize=6):  # smoothened tracks visualization

    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    if condition == 'all':
        df = df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])
    elif condition != 'all' and replicate == 'all':
        df = df[df['CONDITION'] == condition].sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])
    elif condition != 'all' and replicate != 'all':
        df = df[(df['CONDITION'] == condition) & (df['REPLICATE'] == replicate)].sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])


    # Using the  track_visuals function
    fig_visuals, ax_visuals, unique_tracks, track_colors_visuals, norm_visuals, colormap_visuals = track_visuals(df2, c_mode, grid, lut_metric)

    # Plot the full tracks
    for condition in df['CONDITION'].drop_duplicates():
        condition_tracks = df[df['CONDITION'] == condition]
        for replicate in condition_tracks['REPLICATE'].drop_duplicates():
            replicate_tracks = condition_tracks[condition_tracks['REPLICATE'] == replicate]
            for track_id in replicate_tracks['TRACK_ID'].drop_duplicates():
                track_data = replicate_tracks[replicate_tracks['TRACK_ID'] == track_id]
                x = track_data['POSITION_X']
                y = track_data['POSITION_Y']

                # Apply smoothing to the track (if applicable)
                if smoothing_index == None:
                    x_smoothed = x
                    y_smoothed = y
                elif smoothing_index > 0:
                    x_smoothed = x.rolling(window=smoothing_index, min_periods=1).mean()
                    y_smoothed = y.rolling(window=smoothing_index, min_periods=1).mean()
                else:
                    x_smoothed = x
                    y_smoothed = y

                ax_visuals.plot(x_smoothed, y_smoothed, lw=lw, color=track_colors_visuals[f'{condition} {replicate} {track_id}'], label=f'Track {track_id}')
                
                # Get the original color from track_colors_visuals[track_id]
                arrow_color = mcolors.to_rgb(track_colors_visuals[f'{condition} {replicate} {track_id}'])

                if len(x_smoothed) > 1:
                    # Extract the mean direction from df2 for the current track
                    mean_direction_rad = df2[df2[['CONDITION', 'REPLICATE', 'TRACK_ID']] == [condition, replicate, track_id]]['MEAN_DIRECTION_RAD'].values[0]
                    
                    # Use trigonometry to calculate the direction (dx, dy) from the angle
                    dx = np.cos(mean_direction_rad)  # Change in x based on angle
                    dy = np.sin(mean_direction_rad)  # Change in y based on angle
                    
                    # Create an arrow to indicate direction
                    arrow = FancyArrowPatch(
                        posA=(x_smoothed.iloc[-2], y_smoothed.iloc[-2]),  # Start position (second-to-last point)
                        posB=(x_smoothed.iloc[-2] + dx, y_smoothed.iloc[-2] + dy),  # End position based on direction
                        arrowstyle='-|>',  # Style of the arrow (you can adjust the style as needed)
                        color=arrow_color,  # Set the color of the arrow
                        mutation_scale=arrowsize,  # Scale the size of the arrow head (adjust this based on the plot scale)
                        linewidth=1.2,  # Line width for the arrow
                        zorder=30  # Ensure the arrow is drawn on top of the line
                    )

                    # Add the arrow to your plot (if you're using a `matplotlib` figure/axes)
                    plt.gca().add_patch(arrow)

    return plt.gcf()


# ===========================================================================================================================================================================================================================================================================================================

def visualize_normalized_tracks(df, condition='all', replicate='all', c_mode='random colors', only_one_color='black', lw=0.5, grid=True, backround='light', lut_metric='NET_DISTANCE'):

    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    if condition == 'all':
        df = df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])
    elif condition != 'all' and replicate == 'all':
        df = df[df['CONDITION'] == condition].sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])
    elif condition != 'all' and replicate != 'all':
        df = df[(df['CONDITION'] == condition) & (df['REPLICATE'] == replicate)].sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])


    # First sort the data and get groups of tracks.
    # df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'], inplace=True)
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


"""def tracks_lut_map(df2, c_mode='color1', lut_metric='NET_DISTANCE', metrics_dict=None):

    lut_norm_df = df2[['TRACK_ID', lut_metric]].drop_duplicates()

    # Normalize the NET_DISTANCE to a 0-1 range
    lut_min = lut_norm_df[lut_metric].min()
    lut_max = lut_norm_df[lut_metric].max()
    norm = plt.Normalize(vmin=lut_min, vmax=lut_max)


    colormap = get_colormap(c_mode)
    
    # Add a colorbar to show the LUT map
    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    # Create a separate figure for the LUT map (colorbar)
    fig_lut, ax_lut = plt.subplots(figsize=(2, 6))
    ax_lut.axis('off')
    cbar = fig_lut.colorbar(sm, ax=ax_lut, orientation='vertical', extend='both')
    cbar.set_label(metrics_dict[lut_metric], fontsize=10)

    return plt.gcf()"""



# ==========================================================================================================================================================================================================================================================================================================
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def swarm_plot(df, metric, Metric, show_violin=True, show_swarm=True, show_mean=True, show_median=True, show_error_bars=True, show_legend=True, p_testing=False):
    # fig, ax = plt.subplots(figsize=(12.5, 9.5))
    
    # Set the figure size
    plt.figure(figsize=(12.5, 9.5))
    
    # Ensure CONDITION is treated as categorical
    df['CONDITION'] = df['CONDITION'].astype(str)

    swarm_size = 3.15
    swarm_alpha = 0.5

    violin_fill_color = 'whitesmoke'
    violin_edge_color = 'lightgrey'
    violin_alpha = 0.525

    mean_span = 0.275
    median_span = 0.25
    line_width = 1.6

    sns.despine()

    if show_swarm:
        sns.swarmplot(data=df, x='CONDITION', y=metric, hue='REPLICATE', palette='tab10', size=swarm_size, dodge=False, alpha=swarm_alpha, zorder=1, legend=False)
    else:
        pass

    replicate_means = df.groupby(['CONDITION', 'REPLICATE'])[metric].mean().reset_index()
    sns.scatterplot(data=replicate_means, x='CONDITION', y=metric, hue='REPLICATE', palette='tab10', edgecolor='black', s=175, zorder=3, legend=False)

    if show_violin:
        sns.violinplot(data=df, x='CONDITION', y=metric, color=violin_fill_color, edgecolor=violin_edge_color, inner=None, alpha=violin_alpha, zorder=0)
    else:
        pass

    # Calculate mean and median for each condition
    condition_stats = df.groupby('CONDITION')[metric].agg(['mean', 'median']).reset_index()

    # Plot mean and median lines for each condition using seaborn functions
    for i, row in condition_stats.iterrows():
        x_center = i # Adjust x-coordinate to start at the correct condition position
        if show_mean:
            sns.lineplot(x=[x_center - mean_span, x_center + mean_span], y=[row['mean'], row['mean']], color='black', linestyle='-', linewidth=line_width, zorder=4, label='Mean' if i == 0 else "")
        else:
            pass
        if show_median:
            sns.lineplot(x=[x_center - median_span, x_center + median_span], y=[row['median'], row['median']], color='black', linestyle='--', linewidth=line_width, zorder=4, label='Median' if i == 0 else "")
        else:
            pass

    # Calculate mean, median, and standard deviation for each condition
    condition_stats = df.groupby('CONDITION')[metric].agg(['mean', 'median', 'std']).reset_index()

    if show_error_bars:
        # Plot error bars representing mean ± standard deviation for each condition
        for i, row in condition_stats.iterrows():
            x_center = i
            plt.errorbar(x_center, row['mean'], yerr=row['std'], fmt='None', color='black', alpha=0.8,
                        linewidth=1, capsize=11, zorder=5, label='Mean ± SD' if i == 0 else "")
    else:
        pass
    
    if p_testing:
        # P-test
        # Perform pairwise p-tests
        conditions = df['CONDITION'].unique()
        pairs = list(combinations(conditions, 2))
        y_max = df[metric].max()
        y_offset = (y_max * 0.1)  # Offset for p-value annotations
        for i, (cond1, cond2) in enumerate(pairs):
            data1 = df[df['CONDITION'] == cond1][metric]
            data2 = df[df['CONDITION'] == cond2][metric]
            stat, p_value = mannwhitneyu(data1, data2)
            
            # Annotate the plot with the p-value
            x1, x2 = conditions.tolist().index(cond1), conditions.tolist().index(cond2)
            y = y_max + y_offset * (i + 1)
            plt.plot([x1, x1, x2, x2], [y+4.5, y + y_offset / 2.5, y + y_offset / 2.5, y+1.5], lw=1, color='black')
            plt.text((x1 + x2) / 2, y + y_offset / 2, f'p = {round(p_value, 3):.3f}', ha='center', va='bottom', fontsize=10, color='black')
        else:
            pass
        

    plt.title(f"Swarm Plot with Mean and Median Lines for {Metric}")
    plt.xlabel("Condition")
    plt.ylabel(Metric)

    # Create a custom legend entry for the replicates marker.
    replicate_handle = mlines.Line2D([], [], marker='o', color='w',
                                     markerfacecolor=sns.color_palette('tab10')[0],
                                     markeredgecolor='black', markersize=10,
                                     label='Replicates')

    # Get current handles and labels (from Mean, Median, and Error Bars)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the custom replicates handle
    handles.insert(0, replicate_handle)
    labels.insert(0, 'Replicates')
    
    plt.legend(handles=handles, labels=labels,
               title='Legend', title_fontsize='12', fontsize='10',
               loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True)
    
    if show_legend:
        pass
    else:
        plt.legend().remove()

    return plt.gcf()




# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ==============================================================================================================================================================================================================================================================================================

def poly_fit_chart(df, metric, Metric, condition='all', replicate='all', degree=[1], cmap='tab10', fill=False, point_size=75, outline=False, outline_width=1, opacity=0.6, replicates_separately=False, dir=None):

    if outline:
        outline_color = 'black'
    else:
        outline_color = ''
    
    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    df = df.sort_values(by=['CONDITION', 'REPLICATE', 'POSITION_T'])

    if condition == 'all':
        df = df.groupby(['CONDITION', 'POSITION_T']).agg({metric: 'mean'}).reset_index()
        element = 'CONDITION'
        shorthand = 'CONDITION:N'
        what = ''
    elif condition != 'all' and replicate == 'all':
        if replicates_separately:
            df = df[df['CONDITION'] == condition]
            shorthand = 'REPLICATE:N'
            element = 'REPLICATE'
            what = f'for condition {condition}'
        else:
            df = df[df['CONDITION'] == condition].groupby(['CONDITION', 'POSITION_T']).agg({metric: 'mean'}).reset_index()
            shorthand = 'CONDITION:N'
            element = 'CONDITION'
            what = f'for condition {condition}'
    elif condition != 'all' and replicate != 'all':
        df = df[(df['CONDITION'] == condition) & (df['REPLICATE'] == replicate)].sort_values(by=['CONDITION', 'REPLICATE', 'POSITION_T'])
        shorthand = 'REPLICATE:N'
        element = 'REPLICATE'
        what = f'for condition {condition} and replicate {replicate}'   
    

    # Retrieve unique conditions and assign colors from the selected qualitative colormap.
    elements = df[element].unique().tolist()
    colors = get_q_cmap_list(elements, cmap)

    highlight = alt.selection_point(
        on="pointerover", fields=[element], nearest=True
        )
    
    # Create a base chart with the common encodings.
    base = alt.Chart(df).encode(
        x=alt.X('POSITION_T', title='Time position'),
        y=alt.Y(metric, title=Metric),
        color=alt.Color(shorthand, title='Condition', scale=alt.Scale(domain=elements, range=colors))
        )
    
    ignore = 0.1 

    if fill:
        scatter = base.mark_circle(
            size=point_size,
            stroke=outline_color,
            strokeWidth=outline_width,
        ).encode(
            opacity=alt.when(~highlight).then(alt.value(ignore)).otherwise(alt.value(opacity)),
            tooltip=alt.value(None),
        ).add_params(
            highlight
        )
    else:
        # Scatter layer: displays the actual data points.
        scatter = base.mark_point(
            size=point_size, 
            filled=fill, 
            strokeWidth=outline_width,
        ).encode(
            opacity=alt.when(~highlight).then(alt.value(ignore)).otherwise(alt.value(opacity)),
            # tooltip=alt.Tooltip(metric, title=f'Cond: {condition}, Repl: {condition}'),
            tooltip=['POSITION_T', metric, shorthand],
        ).add_params(
            highlight
        )
    
    if degree[0] == 0:
        chart = alt.layer(scatter).properties(
            width=800,
            height=400,
            title=f'{Metric} with Polynomial Fits'
        ).configure_view(
            strokeWidth=1
        )
    else:
        # Build a list of polynomial fit layers, one for each specified degree.
        polynomial_fit = [
            base.transform_regression(
                "POSITION_T", metric,
                method="poly",
                order=order,
                groupby=[element],
                as_=["POSITION_T", str(order)]
            ).mark_line(
            ).transform_fold(
                [str(order)], as_=["degree", metric]
            ).encode(
                x=alt.X('POSITION_T', title='Time position'),
                y=alt.Y(metric, title=Metric),
                color=alt.Color(shorthand, title='Condition', scale=alt.Scale(domain=elements, range=colors)),
                size=alt.when(~highlight).then(alt.value(1.25)).otherwise(alt.value(3))
            )
            for order in degree
            ]
        
        # Layer the scatter points with all polynomial fits.
        chart = alt.layer(scatter, *polynomial_fit).properties(
            width=700,
            height=350,
            title=f'{Metric} with Polynomial Fits for {what}'
        ).configure_view(
            strokeWidth=1
        ).interactive()

    chart.save(dir / 'cache/poly_fit_chart.svg')
    chart.save(dir / 'cache/poly_fit_chart.html')

    return chart
        

# ==============================================================================================================================================================================================================================================================================================

def line_chart(df, metric, Metric, condition='all', replicate='all', cmap='tab10', interpolation=None, show_median=True, replicates_separately=False, dir=None):

    metric_mean = metric
    metric_median = metric.replace('MEAN', 'MEDIAN')

    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    df.sort_values(by=['CONDITION', 'REPLICATE', 'POSITION_T'])

    if condition == 'all':
        df = df.groupby(['CONDITION', 'POSITION_T']).agg({metric_mean: 'mean', metric_median: 'median'}).reset_index()
        shorthand = 'CONDITION:N'
        element = 'CONDITION'
        what = ''
    elif condition != 'all' and replicate == 'all':
        if replicates_separately:
            df = df[df['CONDITION'] == condition]
            shorthand = 'REPLICATE:N'
            element = 'REPLICATE'
            what = f'for condition {condition}'
        else:
            df = df[df['CONDITION'] == condition].groupby(['CONDITION', 'POSITION_T']).agg({metric_mean: 'mean', metric_median: 'median'}).reset_index()
            shorthand = 'CONDITION:N'
            element = 'CONDITION'
            what = f'for condition {condition}'
    elif condition != 'all' and replicate != 'all':
        df = df[(df['CONDITION'] == condition) & (df['REPLICATE'] == replicate)].sort_values(by=['POSITION_T'])
        shorthand = 'REPLICATE:N'
        element = 'REPLICATE'
        what = f'for condition {condition} and replicate {replicate}'   

    if show_median:
        median_opacity = 0.85
        Metric = Metric + ' and median'
    else:
        median_opacity = 0
            
    # Retrieve unique conditions and assign colors from the selected qualitative colormap.
    elements = df[element].unique().tolist()
    colors = get_q_cmap_list(elements, cmap)
    

    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=['POSITION_T'], empty=False)    
    
    color_scale = alt.Scale(domain=elements, range=colors)

    mean_base = alt.Chart(df).encode(
        x=alt.X('POSITION_T', title='Time position'),
        y=alt.Y(metric_mean, title=None),
        color=alt.Color(shorthand, title='Condition', scale=color_scale),
        )
    
    median_base = alt.Chart(df).encode(
        x=alt.X('POSITION_T', title='Time position'),
        y=alt.Y(metric_median, title='Median ' + Metric),
        color=alt.Color(shorthand, title='Condition', scale=color_scale),
        )

    if interpolation is not None:
        mean_line = mean_base.mark_line(interpolate=interpolation)
        median_line = median_base.mark_line(interpolate=interpolation, strokeDash=[4, 3]).encode(
            opacity=alt.value(median_opacity)
            )
        text_median = 0
    else:
        mean_line = mean_base.mark_line()
        median_line = median_base.mark_line(strokeDash=[4, 3]).encode(
            opacity=alt.value(median_opacity)
            )


    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    mean_selectors = mean_base.mark_point().encode(
        opacity=alt.value(0),
        ).add_params(
        nearest
        )
    median_selectors = median_base.mark_point().encode(
        opacity=alt.value(0),
        ).add_params(
        nearest
        )
    
    when_near = alt.when(nearest)

    # Draw points on the line, and highlight based on selection
    mean_points = mean_base.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
        )
    median_points = median_base.mark_point().encode(
        opacity=when_near.then(alt.value(median_opacity)).otherwise(alt.value(0))
        )

    # Draw text labels near the points, and highlight based on selection
    text_mean = mean_line.mark_text(align="left", dx=6, dy=-6).encode(
        color=alt.value('black'),	
        text=when_near.then(metric_mean).otherwise(alt.value(" "))
        )
    text_median = median_line.mark_text(align="left", dx=5, dy=-5).encode(
        color=alt.value('grey'),
        text=when_near.then(metric_median).otherwise(alt.value(" "))
        )

    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color="gray").encode(
        x='POSITION_T',
        ).transform_filter(
        nearest
        )

    chart = alt.layer(
        mean_line, mean_points, text_mean, mean_selectors, rules, median_line, median_points, text_median, median_selectors
        ).properties(
        width=700,
        height=350,
        title=f'{Metric} across time {what}'
        ).configure_view(
        strokeWidth=1
        ).interactive()

    chart.save(dir / 'cache/line_chart.html')
    chart.save(dir / 'cache/line_chart.svg')

    return chart


# ===============================================================================================================================================================================================================================================================================================

def errorband_chart(df, metric, Metric, condition=1, replicate='all', cmap='tab10', interpolation=None, show_mean=True, extent='orig_std', replicates_separately=False, dir=None):

    metric_mean = metric
    metric_std = metric.replace('MEAN', 'STD')
    metric_min = metric.replace('MEAN', 'MIN')
    metric_max = metric.replace('MEAN', 'MAX')

    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    df.sort_values(by=['CONDITION', 'REPLICATE', 'POSITION_T'])

    if condition == 'all':
        df = df.groupby(['CONDITION','POSITION_T']).agg({metric_mean: 'mean', metric_std: 'std', metric_min: 'min', metric_max: 'max'}).reset_index()
        shorthand = 'CONDITION:N'
        element = 'CONDITION'
        element_ = 'Condition'
        what = ''
    elif condition != 'all' and replicate == 'all':
        if replicates_separately:
            df = df[df['CONDITION'] == condition]
            shorthand = 'REPLICATE:N'
            element = 'REPLICATE'
            element_ = 'Replicate'
            what = f'for condition {condition}'
        else:
            df = df[df['CONDITION'] == condition].groupby(['CONDITION', 'POSITION_T']).agg({metric_mean: 'mean', metric_std: 'std', metric_min: 'min', metric_max: 'max'}).reset_index()
            shorthand = 'CONDITION:N'
            element = 'CONDITION'
            element_ = 'Condition'
            what = f'for condition {condition}'
    elif condition != 'all' and replicate != 'all':
        df = df[(df['CONDITION'] == condition) & (df['REPLICATE'] == replicate)].sort_values(by=['CONDITION', 'REPLICATE', 'POSITION_T'])
        shorthand = 'REPLICATE:N'
        element = 'REPLICATE'
        element_ = 'Replicate'
        what = f'for condition {condition} and replicate {replicate}'

    if show_mean:
        mean_opacity = 1
    else:
        mean_opacity = 0

    df['lower'] = df[metric_mean] - df[metric_std]
    df['upper'] = df[metric_mean] + df[metric_std]

    # Retrieve unique conditions and assign colors from the selected qualitative colormap.
    elements = df[element].unique().tolist()
    colors = get_q_cmap_list(elements, cmap)
    color_scale=alt.Scale(domain=elements, range=colors)
    
    if extent == 'min-max':
        band = alt.Chart(df).encode(
            x=alt.X('POSITION_T', title='Time position'),
            y=alt.Y(metric_min),
            y2=alt.Y2(metric_max),
            color=alt.Color(shorthand, title=element_, scale=color_scale),
            tooltip=alt.value(None)
            )
        if interpolation is not None:
            band = band.mark_errorband(interpolate=interpolation)
        else:
            band = band.mark_errorband()

    elif extent == 'orig_std':
        band = alt.Chart(df).encode(
            x=alt.X('POSITION_T', title='Time position'),
            y=alt.Y('upper'),
            y2=alt.Y2('lower'),
            color=alt.Color(shorthand, title=element_, scale=color_scale),   
            opacity=alt.value(0.25),
            tooltip=alt.value(None)
        )
        if interpolation is not None:
            band = band.mark_errorband(interpolate=interpolation)
        else:
            band = band.mark_errorband()

    else:
        band = alt.Chart(df).mark_errorband(extent=extent).encode(
            x=alt.X('POSITION_T', title='Time position'),
            y=alt.Y('upper'),
            y2=alt.Y2('lower'),
            color=alt.Color(shorthand, title=element_, scale=color_scale),
            opacity=alt.value(0.25),
            tooltip=alt.value(None)
        )
        if interpolation is not None:
            band = band.mark_errorband(interpolate=interpolation)  
        else:
            band = band.mark_errorband()

    mean_base = alt.Chart(df).encode(
        x=alt.X('POSITION_T', title='Time position'),
        y=alt.Y(metric_mean, title=None),
        color=alt.Color(shorthand, title=element_, scale=color_scale),
        opacity=alt.value(mean_opacity),
        strokeWidth=alt.value(3),
        tooltip=alt.value(None)
        )

    if interpolation is not None:
        mean_line = mean_base.mark_line(interpolate=interpolation)
    else:
        mean_line = mean_base.mark_line()

    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=['POSITION_T'], empty=False)    
    
    color_scale = alt.Scale(domain=elements, range=colors)

    # Transparent selectors across the chart. This is what tells us
    mean_selectors = mean_base.mark_point().encode(
        opacity=alt.value(0),
        ).add_params(
        nearest
        )
    
    when_near = alt.when(nearest)

    mean_points = mean_base.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)),
        )    

    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color="gray").encode(
        x='POSITION_T',
        ).transform_filter(
        nearest
        )

    # Calculate separate fields for each line of the tooltip.
    tooltip_data = alt.Chart(df).transform_calculate(
        tooltip_line1=f'"{element_}: " + datum.CONDITION',
        tooltip_line2=f'"Time point: " + datum.POSITION_T',
        tooltip_line3=f'"Mean: " + datum["{metric_mean}"]',
        tooltip_line4=f'"Std: " + datum["{metric_std}"]',
        tooltip_line5=f'"Min: " + datum["{metric_min}"]',
        tooltip_line6=f'"Max: " + datum["{metric_max}"]',
        )
    
    if condition == 'all':
        text_opacity = 0
    else:
        text_opacity = 1

    l = -75

    # Create text marks for each line.
    tooltip_line1_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=-30-l,  # adjust vertical offset as needed
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='POSITION_T',
        text=alt.condition(nearest, 'tooltip_line1:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)

    tooltip_line2_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=-15-l,  # adjust vertical offset for the second line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='POSITION_T',
        text=alt.condition(nearest, 'tooltip_line2:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)

    tooltip_line3_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=0-l,  # adjust vertical offset for the third line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='POSITION_T',
        text=alt.condition(nearest, 'tooltip_line3:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)
    
    tooltip_line4_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=15-l,  # adjust vertical offset for the fourth line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='POSITION_T',
        text=alt.condition(nearest, 'tooltip_line4:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)
    
    tooltip_line5_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=30-l,  # adjust vertical offset for the fifth line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='POSITION_T',
        text=alt.condition(nearest, 'tooltip_line5:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)
    
    tooltip_line6_mark = tooltip_data.mark_text(
        align='left',
        dx=6,
        dy=45-l,  # adjust vertical offset for the sixth line
        fontSize=12,
        fontWeight='bold'
        ).encode(
        x='POSITION_T',
        text=alt.condition(nearest, 'tooltip_line6:N', alt.value('')),
        opacity=alt.value(text_opacity)
        ).transform_filter(nearest)

    # Layer the tooltip text marks with your other layers.
    chart = alt.layer(
        band, mean_line, mean_points, mean_selectors, rules,
        tooltip_line1_mark, tooltip_line2_mark, tooltip_line3_mark, tooltip_line4_mark, tooltip_line5_mark, tooltip_line6_mark
        ).properties(
        width=700,
        height=350,
        title=f'{Metric} with its standard deviation across time {what}'
        ).interactive()

    chart.save(dir / 'cache/errorband_chart.html')
    chart.save(dir / 'cache/errorband_chart.svg')

    return chart

