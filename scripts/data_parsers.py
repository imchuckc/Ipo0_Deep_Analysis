import numpy as np
import gzip
import re

def parse_density_file(filepath):
    """Parse a gzipped density data file."""
    with gzip.open(filepath, 'rt') as f:
        lines = f.readlines()
    
    header = parse_header(lines[:4])
    
    # Parse density values
    density_data = []
    for line in lines[4:]:
        if line.startswith('#'):
            continue
        values = [float(val) for val in line.strip().split()]
        density_data.extend(values)
    
    # Get grid dimensions from header
    # If we couldn't get it from header, infer from data
    if 'grid_size' not in header:
        # Try to determine grid dimensions from the data
        total_elements = len(density_data)
        print(f"Grid dimensions not found in header, estimating from data length: {total_elements} elements")
        
        if total_elements > 0:
            # Estimate grid dimensions - this is a simplistic approach
            # Assuming roughly square aspect ratio as a starting point
            estimated_side = int(np.sqrt(total_elements))
            
            # Try to find factors close to estimated side
            grid_rows = 0
            grid_cols = 0
            
            # First try to find exact factors
            for i in range(estimated_side, estimated_side // 2, -1):
                if total_elements % i == 0:
                    grid_rows = i
                    grid_cols = total_elements // i
                    break
            
            # If no exact factors, we need a reasonable approximation
            if grid_rows == 0:
                # Use estimated side and adjust
                grid_rows = estimated_side
                grid_cols = total_elements // estimated_side
                print(f"No exact factors found, using approximate dimensions: {grid_rows} x {grid_cols}")
            else:
                print(f"Found exact factors: {grid_rows} x {grid_cols}")
        else:
            # For empty data, use a very small grid
            grid_rows, grid_cols = 10, 10
            print(f"No data found, using minimal grid dimensions: {grid_rows} x {grid_cols}")
    else:
        grid_rows, grid_cols = header['grid_size']
        print(f"Using grid dimensions from header: {grid_rows} x {grid_cols}")
    
    # Set the grid size in header for downstream use
    header['grid_size'] = (grid_rows, grid_cols)
    
    density_matrix = reshape_data_to_grid(density_data, grid_rows, grid_cols)
    return density_matrix, header

def parse_congestion_file(filepath):
    """Parse a congestion data file (either gzipped or uncompressed)."""
    if filepath.endswith('.gz'):
        # Handle gzipped file
        with gzip.open(filepath, 'rt') as f:
            lines = f.readlines()
    else:
        # Handle uncompressed file
        with open(filepath, 'rt') as f:
            lines = f.readlines()
    
    header = parse_header(lines[:4])
    
    # Parse congestion values
    congestion_data = []
    for line in lines[4:]:
        if line.startswith('#'):
            continue
        values = [float(val) for val in line.strip().split()]
        congestion_data.extend(values)
    
    # Get grid dimensions from header
    # If we couldn't get it from header, infer from data
    if 'grid_size' not in header:
        # Try to determine grid dimensions from the data
        total_elements = len(congestion_data)
        print(f"Grid dimensions not found in header, estimating from data length: {total_elements} elements")
        
        if total_elements > 0:
            # Estimate grid dimensions - this is a simplistic approach
            # Assuming roughly square aspect ratio as a starting point
            estimated_side = int(np.sqrt(total_elements))
            
            # Try to find factors close to estimated side
            grid_rows = 0
            grid_cols = 0
            
            # First try to find exact factors
            for i in range(estimated_side, estimated_side // 2, -1):
                if total_elements % i == 0:
                    grid_rows = i
                    grid_cols = total_elements // i
                    break
            
            # If no exact factors, we need a reasonable approximation
            if grid_rows == 0:
                # Use estimated side and adjust
                grid_rows = estimated_side
                grid_cols = total_elements // estimated_side
                print(f"No exact factors found, using approximate dimensions: {grid_rows} x {grid_cols}")
            else:
                print(f"Found exact factors: {grid_rows} x {grid_cols}")
        else:
            # For empty data, use a very small grid
            grid_rows, grid_cols = 10, 10
            print(f"No data found, using minimal grid dimensions: {grid_rows} x {grid_cols}")
    else:
        grid_rows, grid_cols = header['grid_size']
        print(f"Using grid dimensions from header: {grid_rows} x {grid_cols}")
    
    # Set the grid size in header for downstream use
    header['grid_size'] = (grid_rows, grid_cols)
    
    congestion_matrix = reshape_data_to_grid(congestion_data, grid_rows, grid_cols)
    return congestion_matrix, header

def parse_header(header_lines):
    """Parse header information from file."""
    header = {}
    for line in header_lines:
        if '# version' in line:
            header['version'] = line.split(':')[1].strip()
        elif 'grid_info' in line:
            # Extract dimensions from grid_info
            print(f"Header grid line: {line.strip()}")
            # Extract columns and rows using regex
            columns_match = re.search(r'columns (\d+)', line)
            rows_match = re.search(r'rows (\d+)', line)
            size_x_match = re.search(r'size_x (\d+\.\d+)', line)
            size_y_match = re.search(r'size_y (\d+\.\d+)', line)
            
            if columns_match and rows_match:
                grid_cols = int(columns_match.group(1))
                grid_rows = int(rows_match.group(1))
                header['grid_size'] = (grid_rows, grid_cols)
                print(f"Found grid dimensions in header: {grid_rows} x {grid_cols}")
            
            # Also extract physical dimensions for proper aspect ratio
            if size_x_match and size_y_match:
                size_x = float(size_x_match.group(1))
                size_y = float(size_y_match.group(1))
                header['physical_size'] = (size_x, size_y)
                header['aspect_ratio'] = size_x / size_y
                print(f"Physical dimensions: {size_x} x {size_y}, aspect ratio: {header['aspect_ratio']:.4f}")
    return header

def reshape_data_to_grid(data, rows, cols):
    """Reshape data to specified grid dimensions, handling size mismatches."""
    expected_elements = rows * cols
    total_elements = len(data)
    
    if total_elements != expected_elements:
        print(f"WARNING: Data elements count ({total_elements}) doesn't match expected grid size ({expected_elements})")
        # Adjust data to fit expected dimensions
        if total_elements > expected_elements:
            # Truncate if we have too many elements
            data = data[:expected_elements]
            print(f"Truncated data to fit {rows}x{cols} grid")
        else:
            # Pad with zeros if we have too few elements
            data.extend([0] * (expected_elements - total_elements))
            print(f"Padded data with zeros to fit {rows}x{cols} grid")
    
    # Reshape into grid
    try:
        matrix = np.array(data).reshape(rows, cols)
    except ValueError as e:
        print(f"Error reshaping: {e}")
        print(f"Using a zero-filled matrix with specified dimensions")
        matrix = np.zeros((rows, cols))
    
    return matrix 