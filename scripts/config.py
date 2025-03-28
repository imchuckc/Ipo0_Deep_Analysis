# Base data path - modify this to match your environment
#BASE_DATA_PATH = '/home/scratch.gr150_pnr_trials/gr102g/nvgpu_gr102g/layout/revP4.0/pandr/users/chuckc/test/Ipo0_Deep_Analysis/.irregulaDB'
BASE_DATA_PATH = '/home/scratch.gr150_pnr_trials/gr102g/nvgpu_gr102g/layout/revP4.0/pandr/users/chuckc/test/Ipo0_Deep_Analysis/.DB'
# Visualization settings
FIGURE_WIDTH = 4  # Base figure width in inches (reduced for better aspect ratio)
DPI = 300  # Output image resolution

# Color map specifications
COLOR_SPECTRUM = [
    (0, 0, 0),           # Black for zero/background
    (0, 0, 0.5),         # Dark blue
    (0, 0, 1.0),         # Blue
    (0, 0.5, 1.0),       # Light blue
    (0, 1.0, 1.0),       # Cyan
    (0, 1.0, 0.5),       # Blue-green
    (0, 1.0, 0),         # Green
    (0.5, 1.0, 0),       # Yellow-green
    (1.0, 1.0, 0),       # Yellow
    (1.0, 0.5, 0),       # Orange
    (1.0, 0, 0),         # Red
    (0.5, 0, 0),         # Dark red
]

# Standard fixed bin definitions (sorted from high to low)
# Density map standard bins - range from 0 to 1.0, unified standard
STANDARD_DENSITY_BINS = [
    1.0,    # Maximum boundary
    0.9,    # Deep red
    0.8,    # Red
    0.7,    # Orange-red
    0.6,    # Orange
    0.5,    # Yellow
    0.4,    # Yellow-green
    0.3,    # Green
    0.2,    # Cyan-green
    0.1,    # Light blue
    0.0,    # Minimum boundary
]

# Horizontal congestion standard bins - range from 0 to 1.0, unified standard
STANDARD_HORIZONTAL_CONGESTION_BINS = [
    1.0,    # Maximum boundary
    0.9,    # Deep red
    0.8,    # Red
    0.7,    # Orange-red
    0.6,    # Orange
    0.5,    # Yellow
    0.4,    # Yellow-green
    0.3,    # Green
    0.2,    # Cyan-green
    0.1,    # Light blue
    0.0,    # Minimum boundary
]

# Vertical congestion standard bins - range from 0 to 1.0, unified standard
STANDARD_VERTICAL_CONGESTION_BINS = [
    1.0,    # Maximum boundary
    0.9,    # Deep red
    0.8,    # Red
    0.7,    # Orange-red
    0.6,    # Orange
    0.5,    # Yellow
    0.4,    # Yellow-green
    0.3,    # Green
    0.2,    # Cyan-green
    0.1,    # Light blue
    0.0,    # Minimum boundary
]

# Whether to use standard bins
USE_STANDARD_BINS = True  # Set to True to use unified standard bins

# File patterns for different report types
STAGE_FILE_PATTERNS = {
    # Basic density patterns
    "density": "*.ipo*.{stage}.cell_density.data.gz",
    "pin_density": "*.ipo*.{stage}.pin_density.data.gz",
    "under_utilized_areas": "*.ipo*.{stage}.under_utilized_areas.data.gz",
    
    # Region patterns
    "region_all_cells": "*.ipo*.{stage}.region.all_cells.cell_density.data.gz",
    "region_bounded_cells": "*.ipo*.{stage}.region.bounded_cells.cell_density.data.gz",
    
    # Logic cone density patterns
    "logic_cone_core": "*.ipo*.{stage}.cell_density.logic_cone.core.data.gz",
    "logic_cone_feedthrough": "*.ipo*.{stage}.cell_density.logic_cone.feedthrough.data.gz",
    "logic_cone_input": "*.ipo*.{stage}.cell_density.logic_cone.input.data.gz",
    "logic_cone_output": "*.ipo*.{stage}.cell_density.logic_cone.output.data.gz",
    "logic_cone_retime": "*.ipo*.{stage}.cell_density.logic_cone.retime.data.gz",
    
    # Congestion patterns - TM layers
    "horizontal": "*.ipo*.{stage}.congestion.horizontal.layer_group.TM_layers.data.gz",
    "vertical": "*.ipo*.{stage}.congestion.vertical.layer_group.TM_layers.data.gz",
    
    # Congestion patterns - M layers
    "horizontal_M_layers": "*.ipo*.{stage}.congestion.horizontal.layer_group.M_layers.data.gz",
    "vertical_M_layers": "*.ipo*.{stage}.congestion.vertical.layer_group.M_layers.data.gz",
    
    # Average case congestion patterns
    "horizontal_average": "*.ipo*.{stage}.congestion.horizontal.average_case.data.gz",
    "vertical_average": "*.ipo*.{stage}.congestion.vertical.average_case.data.gz",
    
    # Basic congestion patterns (without layer info)
    "horizontal_basic": "*.ipo*.{stage}.congestion.horizontal.data.gz",
    "vertical_basic": "*.ipo*.{stage}.congestion.vertical.data.gz",
}

# Available processing stages
STAGES = ["place", "route"]

# Visualization parameters
# For density plots
DENSITY_PLOT_VMIN = 0
DENSITY_PLOT_VMAX = 1.0  # 统一设置为1.0以便统一色阶
DENSITY_COLORBAR_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

# For congestion plots - using more dynamic ranges for irregular shapes
HORIZONTAL_CONGESTION_VMIN = 0
HORIZONTAL_CONGESTION_VMAX = 1.0  # 统一设置为1.0以便统一色阶
HORIZONTAL_CONGESTION_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

VERTICAL_CONGESTION_VMIN = 0
VERTICAL_CONGESTION_VMAX = 1.0  # 统一设置为1.0以便统一色阶
VERTICAL_CONGESTION_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Report directory prefix
REPORT_DIR_PREFIX = "llm_reports_"

# White marker settings
WHITE_MARKER_COUNT = 0  # Set to 0 to completely disable white square markers 

# Define available processing stages
# = ["place", "route", "cts", "floorplan", "opt"]  # Add more stages as needed 