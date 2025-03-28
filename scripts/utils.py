import numpy as np
import os
import fnmatch
import datetime

def log_matrix_stats(matrix, matrix_type):
    """Log statistics about the matrix for debugging and analysis."""
    # Get value distribution statistics
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    if matrix_type.lower() == "density":
        percentiles = np.percentile(matrix, [5, 25, 50, 75, 95])
        print(f"{matrix_type} percentiles:")
        print(f"  5th: {percentiles[0]:.4f}")
        print(f" 25th: {percentiles[1]:.4f}")
        print(f" 50th (median): {percentiles[2]:.4f}")
        print(f" 75th: {percentiles[3]:.4f}")
        print(f" 95th: {percentiles[4]:.4f}")
    else:
        percentiles = np.percentile(matrix, [1, 5, 25, 50, 75, 95, 99])
        print(f"{matrix_type} percentiles:")
        print(f"  1st: {percentiles[0]:.4f}")
        print(f"  5th: {percentiles[1]:.4f}")
        print(f" 25th: {percentiles[2]:.4f}")
        print(f" 50th (median): {percentiles[3]:.4f}")
        print(f" 75th: {percentiles[4]:.4f}")
        print(f" 95th: {percentiles[5]:.4f}")
        print(f" 99th: {percentiles[6]:.4f}")
    
    # Count non-zero cells
    non_zero_count = np.sum(matrix > 0)
    total_cells = matrix.size
    non_zero_percentage = (non_zero_count / total_cells) * 100
    
    print(f"{matrix_type} range: {min_val:.4f} to {max_val:.4f}")
    print(f"Non-zero {matrix_type.lower()} cells: {non_zero_count} out of {total_cells} ({non_zero_percentage:.2f}%)")
    
    return {
        "min": min_val,
        "max": max_val,
        "percentiles": percentiles,
        "non_zero_percentage": non_zero_percentage
    }

def find_report_files(base_path, stage_name, file_patterns):
    """Find report files matching patterns for a given stage."""
    found_files = {}
    
    for report_type, pattern in file_patterns.items():
        matching_files = []
        print(f"Searching for {report_type} files with pattern: {pattern}")
        
        # Recursively search for matching files
        for root, dirs, files in os.walk(base_path):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    full_path = os.path.join(root, filename)
                    matching_files.append(full_path)
                    print(f"  - Found match: {filename}")
        
        # If multiple matches found, use the most recent file
        if matching_files:
            if len(matching_files) > 1:
                print(f"  Found {len(matching_files)} matches, selecting most recent file")
            
            # Sort by modification time, newest first
            matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            found_files[report_type] = matching_files[0]
            print(f"Selected {report_type} report: {matching_files[0]}")
        else:
            print(f"Warning: No {report_type} report found for stage '{stage_name}'")
    
    return found_files

def create_timestamped_dir(base_dir, prefix="llm_reports_"):
    """Create a timestamped directory for reports."""
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_dir = os.path.join(base_dir, f"{prefix}{timestamp_str}")
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

def write_summary_stats(summary_file, matrix, matrix_type, source_filename, output_filename):
    """Write statistics about a matrix to a summary file."""
    with open(summary_file, 'a') as summary:
        summary.write(f"=== {matrix_type.upper()} ANALYSIS ===\n")
        summary.write(f"Source file: {os.path.basename(source_filename)}\n")
        
        # Collect statistics
        stats = log_matrix_stats(matrix, matrix_type)
        min_val, max_val = stats["min"], stats["max"]
        percentiles = stats["percentiles"]
        non_zero_percentage = stats["non_zero_percentage"]
        
        # Write statistics to summary
        summary.write(f"Minimum Value: {min_val:.6f}\n")
        summary.write(f"Maximum Value: {max_val:.6f}\n")
        summary.write(f"Median Value: {percentiles[2 if matrix_type.lower() == 'density' else 3]:.6f}\n")
        summary.write(f"Non-zero Percentage: {non_zero_percentage:.2f}%\n")
        
        if matrix_type.lower() == "density":
            percentile_labels = ["5th", "25th", "50th (median)", "75th", "95th"]
            percentile_indices = range(5)
        else:
            percentile_labels = ["1st", "5th", "25th", "50th (median)", "75th", "95th", "99th"]
            percentile_indices = range(7)
            
        summary.write("Percentiles:\n")
        for i, label in enumerate(percentile_labels):
            summary.write(f"  {label}: {percentiles[i]:.6f}\n")
        
        summary.write(f"Output visualization: {os.path.basename(output_filename)}\n\n") 