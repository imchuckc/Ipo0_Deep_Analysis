# FPGA Layout Report Analyzer

A modular Python tool for analyzing FPGA layout reports, including density and congestion data. 
The tool generates visualizations and statistical analysis summaries.

## Features

- Parse and visualize cell density data
- Parse and visualize horizontal/vertical congestion data
- Generate detailed statistical analysis reports
- Support for different design stages (place, route, etc.)
- Dynamic grid dimensions based on actual input data

## Project Structure

The project has been organized into modular components for better maintainability:

```
├── main.py               # Main entry point
├── scripts/              # All modules are organized in a scripts directory
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration settings and constants
│   ├── data_parsers.py   # Functions for parsing report data files
│   ├── report_processor.py # Processing logic for handling reports
│   ├── utils.py          # Common utility functions
│   └── visualizers.py    # Visualization functions
```

This directory structure makes it easy to:
- Version control the codebase
- Copy the entire scripts directory between projects
- Maintain separate configurations for different environments

## Usage

### Basic Usage

1. Update the `BASE_DATA_PATH` in `scripts/config.py` to point to your report files location
2. Run the script:

```bash
python main.py
```

The script will:
1. Find all relevant report files in the specified directory
2. Process and analyze them
3. Generate visualizations of density and congestion data
4. Create a timestamped directory with all results

### Command-Line Options

You can also override config settings using command-line arguments:

```bash
python main.py --base-path /path/to/reports --prefix custom_prefix_ --stages place route
```

Available options:
- `--base-path`: Override the base path to search for reports
- `--prefix`: Set a custom prefix for the output directory
- `--stages`: Specify which stages to process (space-separated list)

## Key Features

### Dynamic Grid Dimensions

The tool now automatically determines grid dimensions from the actual input data:
1. First, it attempts to read dimensions from the file header
2. If not available, it tries to infer reasonable dimensions from the data size
3. As a last resort, it falls back to default values

This makes the tool adaptable to different partition sizes without requiring manual configuration.

## Customization

You can customize various aspects of the tool by editing the `scripts/config.py` file:

- Modify visualization parameters
- Add new processing stages
- Update file patterns for finding reports

## Adding New Features

To add new types of visualizations or support for new report types:

1. Add new parser functions in `scripts/data_parsers.py`
2. Add new visualization functions in `scripts/visualizers.py`
3. Update the processing logic in `scripts/report_processor.py`
4. Update config settings in `scripts/config.py`

## Version Control and Copying

With the restructured code, you can easily:
- Copy the entire `scripts` directory to create a new version
- Maintain different versions for different projects
- Apply patches or updates to specific modules without affecting others 