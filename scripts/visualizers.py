import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scripts import config

def add_white_square_markers(ax, num_markers=30, cols=None, rows=None):
    """Add white square markers to a plot."""
    # Use the actual dimensions from the axes if not provided
    if cols is None or rows is None:
        # Get dimensions from the current image being displayed
        ax_images = ax.get_images()
        if ax_images:
            height, width = ax_images[0].get_array().shape
            rows = height
            cols = width
        else:
            # If no image found, use minimal reasonable values
            cols, rows = 100, 100
    
    # Print the dimensions being used for markers
    print(f"Adding white markers using dimensions: {rows} x {cols}")
    
    np.random.seed(42)  # Use a fixed seed for reproducibility
    marker_x = np.random.randint(0, cols, num_markers)
    marker_y = np.random.randint(0, rows, num_markers)
    ax.scatter(marker_x, marker_y, s=20, facecolors='none', edgecolors='white', marker='s')

def setup_plot_layout(rows, cols, header):
    """Set up plot layout with proper dimensions."""
    # Print the dimensions being used for plot layout
    print(f"Setting up plot with dimensions: {rows} x {cols}")
    
    # Get physical dimensions for proper aspect ratio
    # Default physical dimensions as fallback if not in header
    default_physical_size = (2.304, 2.028)  # (width, height)
    physical_size = header.get('physical_size', default_physical_size)
    physical_width = physical_size[0]
    physical_height = physical_size[1]
    
    print(f"Using physical dimensions: {physical_width} × {physical_height}, ratio: {physical_width/physical_height:.4f}")
    
    # Calculate using actual physical dimensions
    # Scale to reasonable image size while maintaining aspect ratio
    scale_factor = 5  # Adjust overall image size
    fig_width = physical_width * scale_factor
    fig_height = physical_height * scale_factor
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(fig_width + 2, fig_height))  # Add width for legend space
    
    # Adjust main plot area, move left side position slightly left, reduce width to leave space for legend
    main_plot = plt.axes([0.05, 0.1, 0.7, 0.85])
    
    # Set default background to black
    fig.patch.set_facecolor('black')
    main_plot.set_facecolor('black')
    
    return fig, main_plot

def configure_plot_axes(ax, cols, rows):
    """Configure plot axes with proper ticks and labels."""
    x_ticks = np.linspace(0, cols-1, 6).astype(int)
    y_ticks = np.linspace(0, rows-1, 10).astype(int)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    
    ax.set_xlabel('X Coordinate', color='white')
    ax.set_ylabel('Y Coordinate', color='white')
    
    # Make tick labels white for better visibility against black background
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

def add_colorbar(fig, im, tick_values):
    """Add a colorbar to the figure."""
    cbar_ax = fig.add_axes([0.15, 0.02, 0.75, 0.02])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=tick_values)
    cbar.set_label('Unit: percent', labelpad=0, color='white')
    cbar.ax.xaxis.set_tick_params(color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

def create_congestion_colormap(data_path=None):
    """Create a standard colormap for congestion visualizations."""
    # 无论数据源如何，都使用统一的颜色谱
    return LinearSegmentedColormap.from_list('congestion', config.COLOR_SPECTRUM, N=256)

def add_professional_colorbar(fig, im, data_matrix, fixed_bins=None):
    """Add a professional colorbar with statistics for each color range in table format.
    
    Args:
        fig: matplotlib figure
        im: image object
        data_matrix: the data matrix
        fixed_bins: optional predefined bins (from max to min) to use instead of dynamically calculated bins
                   Using fixed bins ensures consistent color standards across different partitions
    """
    # Get non-zero values (exclude background)
    non_zero_values = data_matrix[data_matrix > 0]
    if len(non_zero_values) == 0:
        return
    
    # Determine the number and range of bins
    num_bins = 10
    
    if fixed_bins is not None:
        # Use predefined fixed bins
        bins = fixed_bins
        num_bins = len(fixed_bins) - 1
    else:
        # Dynamically calculate bins
        # Get data range
        vmin = np.min(non_zero_values)
        vmax = np.max(non_zero_values)
        
        # Ensure maximum value is slightly larger than actual maximum
        vmax = np.ceil(vmax * 100) / 100
        
        # Create uniform bins - sorted from high to low
        bins = np.linspace(vmin, vmax, num_bins + 1)  # Add +1 to ensure 10 intervals
        bins = np.flip(bins)  # Reverse order, from high to low
    
    # Add legend area - placed at the far right, in table form
    legend_ax = fig.add_axes([0.78, 0.15, 0.2, 0.7])
    legend_ax.set_facecolor('black')
    
    # Remove axis labels and ticks
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    
    # Get color mapping
    cmap = im.get_cmap()
    norm = im.norm
    
    # Table-style layout
    # Calculate height for each bin
    bin_height = 0.7 / num_bins  # Height slightly reduced to add grid lines
    
    # Output total element count for troubleshooting
    total_elements = len(non_zero_values)
    print(f"Total non-zero elements for binning: {total_elements}")
    
    # Check data min and max values
    data_min = np.min(non_zero_values)
    data_max = np.max(non_zero_values)
    print(f"Data range: {data_min:.4f} to {data_max:.4f}")
    
    # Check for values greater than 1.0
    over_one_count = np.sum(non_zero_values > 1.0)
    if over_one_count > 0:
        print(f"Found {over_one_count} values greater than 1.0")
    
    # Handle general case of data exceeding normal range (>1.0)
    max_bin_value = bins[0]  # Get the value of the maximum bin
    has_large_values = over_one_count > 0 and data_max > max_bin_value
    
    # FIX: Rather than checking each bin separately, use a single call to digitize
    # This ensures each element is counted exactly once by determining which bin it belongs to
    bin_indices = np.digitize(non_zero_values, bins) - 1
    
    # Ensure boundary values are assigned to the correct bin
    # We want bin_indices to be in range [0, num_bins-1]
    bin_indices = np.clip(bin_indices, 0, num_bins-1)
    
    # Count elements in each bin
    bin_counts = [np.sum(bin_indices == i) for i in range(num_bins)]
    total_counted = sum(bin_counts)
    
    # For each bin, create a bordered rectangle cell
    for i in range(num_bins):
        # Calculate bin upper and lower limits - note they are sorted high to low
        high = bins[i]
        low = bins[i+1] if i < num_bins-1 else 0  # Use 0 as lower bound for last bin
        
        # Get count for this bin
        count = bin_counts[i]
        
        # Determine label based on bin position
        if i == 0 and has_large_values:
            label = f"beyond {high:.3f}"
        elif i == 0:  # Max bin (first one)
            label = f"beyond {high:.3f}"
        elif i == num_bins-1:  # Last bin
            label = f"{low:.3f} to {high:.3f}"
        else:
            label = f"{low:.3f} to {high:.3f}"
        
        # Calculate rectangle position (arranged top to bottom)
        y_pos = 1.0 - (i+1) * bin_height
        
        # Get color - use middle value or maximum value
        if i == 0:  # Max bin
            if high > 1.0:
                color_val = 1.0  # Limit maximum value to 1.0
            else:
                color_val = high  # Use maximum value
        else:
            color_val = (low + high) / 2  # Use middle value
        
        # Use colormap to get corresponding color, ensure within valid range
        color_val_normalized = min(max(color_val, norm.vmin), norm.vmax)
        color = cmap(norm(color_val_normalized))
        
        # Draw color rectangle cell (left side)
        color_cell = plt.Rectangle((0, y_pos), 0.3, bin_height, 
                             facecolor=color, edgecolor='white', linewidth=1)
        legend_ax.add_patch(color_cell)
        
        # Draw label rectangle cell (middle)
        label_cell = plt.Rectangle((0.3, y_pos), 0.45, bin_height, 
                             facecolor='black', edgecolor='white', linewidth=1)
        legend_ax.add_patch(label_cell)
        
        # Draw count rectangle cell (right side)
        count_cell = plt.Rectangle((0.75, y_pos), 0.25, bin_height, 
                             facecolor='black', edgecolor='white', linewidth=1)
        legend_ax.add_patch(count_cell)
        
        # Add text label (value range)
        legend_ax.text(0.525, y_pos + bin_height/2, label, 
                       va='center', ha='center', color='white', fontsize=7)
        
        # Add count value
        count_str = f"{count}"
        legend_ax.text(0.875, y_pos + bin_height/2, count_str, 
                       va='center', ha='center', color='white', fontsize=7)
    
    # Verify total count is correct
    print(f"Bin counts: {bin_counts}")
    print(f"Total counted elements: {total_counted}/{total_elements}")
    if total_counted != total_elements:
        print(f"WARNING: Element count mismatch! Missing {total_elements - total_counted} elements")
    
    # Add border
    for spine in legend_ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1)

def plot_density_map_continuous(density_matrix, header, output_file='cell_density_spectrum.png'):
    """Create a visualization of density data."""
    rows, cols = density_matrix.shape
    print(f"Plotting cell density matrix with shape: {rows} x {cols}")
    
    # Get statistics for logging
    from scripts.utils import log_matrix_stats
    log_matrix_stats(density_matrix, "Density")
    
    # Setup plot
    fig, main_plot = setup_plot_layout(rows, cols, header)
    
    # Create color map
    custom_cmap = create_congestion_colormap()
    
    # Set default values from config
    vmin_value = config.DENSITY_PLOT_VMIN
    vmax_value = config.DENSITY_PLOT_VMAX
    
    # Only dynamically adjust color scale range when not using standard bins
    if not config.USE_STANDARD_BINS:
        # For irregular shapes, use dynamic min-max
        # Get the non-zero values for better visualization
        non_zero_values = density_matrix[density_matrix > 0]
        if len(non_zero_values) > 0:
            # Use data-driven values for better visualization
            p95 = np.percentile(non_zero_values, 95)
            # Set max based on 95th percentile if significantly different from config
            if abs(p95 - vmax_value) > 0.2:
                vmax_value = min(1.0, p95 * 1.2)  # Allow some headroom but cap at 1.0
    
    # Get physical dimensions for proper aspect ratio
    default_physical_size = (2.304, 2.028)  # (width, height)
    physical_size = header.get('physical_size', default_physical_size)
    physical_width = physical_size[0]
    physical_height = physical_size[1]
    
    # Create a masked array to properly handle zero values (make them transparent)
    masked_data = np.ma.masked_where(density_matrix == 0, density_matrix)
    
    # Use proportional display to maintain true physical dimensions
    im = main_plot.imshow(masked_data, cmap=custom_cmap, aspect='equal',
                         vmin=vmin_value, vmax=vmax_value, 
                         interpolation='none', origin='lower')
    
    # Set ticks and labels
    configure_plot_axes(main_plot, cols, rows)
    
    # Get title directly from source filename, removing .data.gz suffix
    title = 'Cell Density'  # Default title
    source_file = header.get('source_file', '')
    if source_file:
        # Remove .data.gz suffix
        if source_file.endswith('.data.gz'):
            title = source_file[:-8]  # Remove .data.gz
        else:
            title = source_file  # If no such suffix, use full filename
    
    # Set title with white text for dark background
    main_plot.set_title(title, fontsize=12, pad=2, color='white')
    
    # Determine whether to use standard bins
    if config.USE_STANDARD_BINS:
        # Use standardized bin definitions
        add_professional_colorbar(fig, im, density_matrix, 
                                fixed_bins=config.STANDARD_DENSITY_BINS)
    else:
        # Use dynamically calculated bins
        add_professional_colorbar(fig, im, density_matrix)
    
    # Don't use tight_layout, control layout manually
    # plt.tight_layout(rect=[0, 0, 0.80, 1])
    
    # Save image with padding to ensure all elements are visible
    plt.savefig(output_file, dpi=config.DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_congestion_map(congestion_matrix, header, direction, output_filename):
    """Create a visualization of congestion data."""
    rows, cols = congestion_matrix.shape
    print(f"Plotting {direction} congestion matrix with shape: {rows} x {cols}")
    
    # Get statistics for logging
    from scripts.utils import log_matrix_stats
    log_matrix_stats(congestion_matrix, f"{direction} Congestion")
    
    # Setup plot
    fig, main_plot = setup_plot_layout(rows, cols, header)
    
    # Create color map
    custom_cmap = create_congestion_colormap()
    
    # Set appropriate vmin and vmax values based on config
    if direction == "horizontal":
        vmin_value = config.HORIZONTAL_CONGESTION_VMIN
        vmax_value = config.HORIZONTAL_CONGESTION_VMAX
        tick_values = config.HORIZONTAL_CONGESTION_TICKS
        standard_bins = config.STANDARD_HORIZONTAL_CONGESTION_BINS
    else:  # vertical
        vmin_value = config.VERTICAL_CONGESTION_VMIN
        vmax_value = config.VERTICAL_CONGESTION_VMAX
        tick_values = config.VERTICAL_CONGESTION_TICKS
        standard_bins = config.STANDARD_VERTICAL_CONGESTION_BINS
    
    # Only dynamically adjust color scale range when not using standard bins
    if not config.USE_STANDARD_BINS:
        # For irregular shapes, use dynamic min-max
        # Get the percentiles for better visualization
        non_zero_values = congestion_matrix[congestion_matrix > 0]
        if len(non_zero_values) > 0:
            # Use data-driven values for better visualization
            p05 = np.percentile(non_zero_values, 5)
            p95 = np.percentile(non_zero_values, 95)
            # Set min/max based on data percentiles if they differ significantly from config
            if abs(p05 - vmin_value) > 0.1 or abs(p95 - vmax_value) > 0.1:
                vmin_value = max(0, p05)
                vmax_value = min(1.0, p95 * 1.1)  # Allow some headroom
    
    # Get physical dimensions for proper aspect ratio
    default_physical_size = (2.304, 2.028)  # (width, height)
    physical_size = header.get('physical_size', default_physical_size)
    physical_width = physical_size[0]
    physical_height = physical_size[1]
    
    # Create a masked array to properly handle zero values (make them transparent)
    masked_data = np.ma.masked_where(congestion_matrix == 0, congestion_matrix)
    
    # Use proportional display to maintain true physical dimensions
    im = main_plot.imshow(masked_data, cmap=custom_cmap, aspect='equal', 
                        vmin=vmin_value, vmax=vmax_value, interpolation='none', origin='lower')
    
    # Set ticks and labels
    configure_plot_axes(main_plot, cols, rows)
    
    # Get title directly from source filename, removing .data.gz suffix
    title = f'{direction} congestion'  # Default title
    source_file = header.get('source_file', '')
    if source_file:
        # Remove .data.gz suffix
        if source_file.endswith('.data.gz'):
            title = source_file[:-8]  # Remove .data.gz
        else:
            title = source_file  # If no such suffix, use full filename
    
    main_plot.set_title(title, fontsize=12, pad=2, color='white')
    
    # Determine whether to use standard bins
    if config.USE_STANDARD_BINS:
        # Use standardized bin definitions
        add_professional_colorbar(fig, im, congestion_matrix, 
                                 fixed_bins=standard_bins)
    else:
        # Use dynamically calculated bins
        add_professional_colorbar(fig, im, congestion_matrix)
    
    # Don't use tight_layout, control layout manually
    # plt.tight_layout(rect=[0, 0, 0.70, 1])
    
    # Save image with padding to ensure all elements are visible
    plt.savefig(output_filename, dpi=config.DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig) 