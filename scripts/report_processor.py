import os
import datetime
import numpy as np
from scripts.data_parsers import parse_density_file, parse_congestion_file
from scripts.visualizers import plot_density_map_continuous, plot_congestion_map
from scripts.utils import find_report_files, write_summary_stats
from scripts import config as cfg

def process_reports_by_stage(stage_name, base_path, reports_dir):
    """Process report files for a specified stage and generate visualizations."""
    # Validate stage is in allowed stages
    if stage_name not in cfg.STAGES:
        print(f"Warning: Stage '{stage_name}' not in configured stages {cfg.STAGES}")
    
    # Create stage-specific subdirectory
    stage_dir = os.path.join(reports_dir, f"{stage_name}_reports")
    os.makedirs(stage_dir, exist_ok=True)
    
    # Build filename patterns by substituting stage name into the patterns
    filename_patterns = {}
    for report_type, pattern_template in cfg.STAGE_FILE_PATTERNS.items():
        filename_patterns[report_type] = pattern_template.format(stage=stage_name)
    
    # Find matching files
    found_files = find_report_files(base_path, stage_name, cfg.STAGE_FILE_PATTERNS)
    
    # Process found files
    output_files = []
    
    # Debug info
    print(f"Processing {stage_name} report files: {list(found_files.keys())}")
    
    # Create a summary text file in the stage directory
    summary_file = os.path.join(stage_dir, f"{stage_name}_analysis_summary.txt")
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(summary_file, 'w') as summary:
        # Write content to summary file
        summary.write(f"=== {stage_name.upper()} STAGE ANALYSIS SUMMARY ===\n")
        summary.write(f"Generated on: {timestamp}\n\n")
    
    # First process density data to get shape information
    shape_mask = None
    density_matrix = None
    density_header = None
    
    if "density" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["density"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        density_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} density data...")
        
        # Parse file and get dimensions from the data itself
        density_matrix, density_header = parse_density_file(found_files["density"])
        
        # Add source file information to header
        density_header["source_file"] = os.path.basename(found_files["density"])
        
        # Create a better shape mask - use thresholding to identify valid areas
        # Only consider values > 0 as part of the valid chip area
        shape_mask = np.zeros_like(density_matrix)
        shape_mask[density_matrix > 0] = 1.0
        
        # Save the dimensions for statistics output
        density_shape = density_matrix.shape
        print(f"Density shape: {density_shape}")
        
        write_summary_stats(summary_file, density_matrix, "Density", found_files["density"], density_output)
        
        plot_density_map_continuous(density_matrix, density_header, density_output)
        output_files.append(density_output)
    
    # Process pin density data
    if "pin_density" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["pin_density"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        pin_density_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} pin density data...")
        
        # Parse file to get data
        pin_density_matrix, pd_header = parse_density_file(found_files["pin_density"])
        pd_shape = pin_density_matrix.shape
        print(f"Pin density shape: {pd_shape}")
        
        # Apply shape mask if available
        if shape_mask is not None and pin_density_matrix.shape != shape_mask.shape:
            # Resize the mask to match pin density data
            print(f"Resizing mask from {shape_mask.shape} to {pin_density_matrix.shape}")
            resized_mask = np.zeros(pin_density_matrix.shape, dtype=np.float32)
            
            # Copy the common area
            rows_min = min(shape_mask.shape[0], pin_density_matrix.shape[0])
            cols_min = min(shape_mask.shape[1], pin_density_matrix.shape[1])
            
            # Copy the valid part of the mask
            resized_mask[:rows_min, :cols_min] = shape_mask[:rows_min, :cols_min]
            current_mask = resized_mask
        
        # Add source file information to header
        pd_header["source_file"] = os.path.basename(found_files["pin_density"])
        
        write_summary_stats(summary_file, pin_density_matrix, "Pin Density", 
                           found_files["pin_density"], pin_density_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(pin_density_matrix, pd_header, pin_density_output)
        output_files.append(pin_density_output)
    
    # Process under-utilized areas data
    if "under_utilized" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["under_utilized"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        under_utilized_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} under-utilized areas data...")
        
        # Parse file to get data (use density file parser as the format is similar)
        under_utilized_matrix, uu_header = parse_density_file(found_files["under_utilized"])
        uu_shape = under_utilized_matrix.shape
        print(f"Under-utilized areas shape: {uu_shape}")
        
        # Apply shape mask if available
        if shape_mask is not None and under_utilized_matrix.shape != shape_mask.shape:
            # Resize the mask to match under-utilized data
            print(f"Resizing mask from {shape_mask.shape} to {under_utilized_matrix.shape}")
            resized_mask = np.zeros(under_utilized_matrix.shape, dtype=np.float32)
            
            # Copy the common area
            rows_min = min(shape_mask.shape[0], under_utilized_matrix.shape[0])
            cols_min = min(shape_mask.shape[1], under_utilized_matrix.shape[1])
            
            # Copy the valid part of the mask
            resized_mask[:rows_min, :cols_min] = shape_mask[:rows_min, :cols_min]
            current_mask = resized_mask
        
        # Add source file information to header
        uu_header["source_file"] = os.path.basename(found_files["under_utilized"])
        
        write_summary_stats(summary_file, under_utilized_matrix, "Under-Utilized Areas", 
                           found_files["under_utilized"], under_utilized_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(under_utilized_matrix, uu_header, under_utilized_output)
        output_files.append(under_utilized_output)
    
    # Process logic cone core density data
    if "logic_cone_core" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["logic_cone_core"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        logic_cone_core_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} logic cone core density data...")
        
        # Parse file to get data
        logic_cone_core_matrix, lcc_header = parse_density_file(found_files["logic_cone_core"])
        lcc_shape = logic_cone_core_matrix.shape
        print(f"Logic cone core density shape: {lcc_shape}")
        
        # Add source file information to header
        lcc_header["source_file"] = os.path.basename(found_files["logic_cone_core"])
        
        write_summary_stats(summary_file, logic_cone_core_matrix, "Logic Cone Core Density", 
                           found_files["logic_cone_core"], logic_cone_core_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(logic_cone_core_matrix, lcc_header, logic_cone_core_output)
        output_files.append(logic_cone_core_output)
    
    # Process logic cone feedthrough density data
    if "logic_cone_feedthrough" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["logic_cone_feedthrough"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        logic_cone_ft_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} logic cone feedthrough density data...")
        
        # Parse file to get data
        logic_cone_ft_matrix, lcft_header = parse_density_file(found_files["logic_cone_feedthrough"])
        lcft_shape = logic_cone_ft_matrix.shape
        print(f"Logic cone feedthrough density shape: {lcft_shape}")
        
        # Add source file information to header
        lcft_header["source_file"] = os.path.basename(found_files["logic_cone_feedthrough"])
        
        write_summary_stats(summary_file, logic_cone_ft_matrix, "Logic Cone Feedthrough Density", 
                           found_files["logic_cone_feedthrough"], logic_cone_ft_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(logic_cone_ft_matrix, lcft_header, logic_cone_ft_output)
        output_files.append(logic_cone_ft_output)
    
    # Process logic cone input density data
    if "logic_cone_input" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["logic_cone_input"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        logic_cone_input_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} logic cone input density data...")
        
        # Parse file to get data
        logic_cone_input_matrix, lci_header = parse_density_file(found_files["logic_cone_input"])
        lci_shape = logic_cone_input_matrix.shape
        print(f"Logic cone input density shape: {lci_shape}")
        
        # Add source file information to header
        lci_header["source_file"] = os.path.basename(found_files["logic_cone_input"])
        
        write_summary_stats(summary_file, logic_cone_input_matrix, "Logic Cone Input Density", 
                           found_files["logic_cone_input"], logic_cone_input_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(logic_cone_input_matrix, lci_header, logic_cone_input_output)
        output_files.append(logic_cone_input_output)
    
    # Process logic cone output density data
    if "logic_cone_output" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["logic_cone_output"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        logic_cone_output_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} logic cone output density data...")
        
        # Parse file to get data
        logic_cone_output_matrix, lco_header = parse_density_file(found_files["logic_cone_output"])
        lco_shape = logic_cone_output_matrix.shape
        print(f"Logic cone output density shape: {lco_shape}")
        
        # Add source file information to header
        lco_header["source_file"] = os.path.basename(found_files["logic_cone_output"])
        
        write_summary_stats(summary_file, logic_cone_output_matrix, "Logic Cone Output Density", 
                           found_files["logic_cone_output"], logic_cone_output_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(logic_cone_output_matrix, lco_header, logic_cone_output_output)
        output_files.append(logic_cone_output_output)
    
    # Process logic cone retime density data
    if "logic_cone_retime" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["logic_cone_retime"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        logic_cone_retime_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} logic cone retime density data...")
        
        # Parse file to get data
        logic_cone_retime_matrix, lcr_header = parse_density_file(found_files["logic_cone_retime"])
        lcr_shape = logic_cone_retime_matrix.shape
        print(f"Logic cone retime density shape: {lcr_shape}")
        
        # Add source file information to header
        lcr_header["source_file"] = os.path.basename(found_files["logic_cone_retime"])
        
        write_summary_stats(summary_file, logic_cone_retime_matrix, "Logic Cone Retime Density", 
                           found_files["logic_cone_retime"], logic_cone_retime_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(logic_cone_retime_matrix, lcr_header, logic_cone_retime_output)
        output_files.append(logic_cone_retime_output)
    
    # Process horizontal congestion with proper masking
    if "horizontal" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["horizontal"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        horizontal_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} horizontal congestion data...")
        
        # Parse file and get dimensions from the data itself
        horizontal_matrix, h_header = parse_congestion_file(found_files["horizontal"])
        h_shape = horizontal_matrix.shape
        print(f"Horizontal congestion shape: {h_shape}")
        
        # Apply shape mask if available - use a proper masking approach
        if shape_mask is not None:
            # First check if the dimensions match
            if shape_mask.shape != horizontal_matrix.shape:
                # Need to resize the mask to match the congestion data
                print(f"Resizing mask from {shape_mask.shape} to {horizontal_matrix.shape}")
                resized_mask = np.zeros(horizontal_matrix.shape, dtype=np.float32)
                
                # Copy the common area
                rows_min = min(shape_mask.shape[0], horizontal_matrix.shape[0])
                cols_min = min(shape_mask.shape[1], horizontal_matrix.shape[1])
                
                # Copy the valid part of the mask
                resized_mask[:rows_min, :cols_min] = shape_mask[:rows_min, :cols_min]
                shape_mask = resized_mask
            
            print(f"Applying clean shape mask to horizontal congestion data")
            # Create a clean masked version:
            # 1. Make a copy of the original data
            masked_horizontal = horizontal_matrix.copy()
            # 2. Identify regions outside chip area (mask == 0)
            outside_region = (shape_mask == 0)
            # 3. Set those regions to zero
            masked_horizontal[outside_region] = 0.0
            
            horizontal_matrix = masked_horizontal
        
        write_summary_stats(summary_file, horizontal_matrix, "Horizontal Congestion", 
                           found_files["horizontal"], horizontal_output)
        
        # Add source file information to header
        header_to_use = density_header if density_header else h_header
        header_to_use["source_file"] = os.path.basename(found_files["horizontal"])
        
        plot_congestion_map(horizontal_matrix, header_to_use, "horizontal", horizontal_output)
        output_files.append(horizontal_output)
    
    # Process vertical congestion with proper masking
    if "vertical" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["vertical"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        vertical_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} vertical congestion data...")
        
        # Parse file and get dimensions from the data itself
        vertical_matrix, v_header = parse_congestion_file(found_files["vertical"])
        v_shape = vertical_matrix.shape
        print(f"Vertical congestion shape: {v_shape}")
        
        # Apply shape mask if available
        if shape_mask is not None:
            # First check if the dimensions match
            if shape_mask.shape != vertical_matrix.shape:
                # Need to resize the mask to match the congestion data
                print(f"Resizing mask from {shape_mask.shape} to {vertical_matrix.shape}")
                resized_mask = np.zeros(vertical_matrix.shape, dtype=np.float32)
                
                # Copy the common area
                rows_min = min(shape_mask.shape[0], vertical_matrix.shape[0])
                cols_min = min(shape_mask.shape[1], vertical_matrix.shape[1])
                
                # Copy the valid part of the mask
                resized_mask[:rows_min, :cols_min] = shape_mask[:rows_min, :cols_min]
                shape_mask = resized_mask
            
            print(f"Applying clean shape mask to vertical congestion data")
            # Create a clean masked version:
            # 1. Make a copy of the original data
            masked_vertical = vertical_matrix.copy()
            # 2. Identify regions outside chip area (mask == 0)
            outside_region = (shape_mask == 0)
            # 3. Set those regions to zero
            masked_vertical[outside_region] = 0.0
            
            vertical_matrix = masked_vertical
        
        write_summary_stats(summary_file, vertical_matrix, "Vertical Congestion", 
                           found_files["vertical"], vertical_output)
        
        # Add source file information to header
        header_to_use = density_header if density_header else v_header
        header_to_use["source_file"] = os.path.basename(found_files["vertical"])
        
        plot_congestion_map(vertical_matrix, header_to_use, "vertical", vertical_output)
        output_files.append(vertical_output)
    
    # Process region_all_cells data file
    if "region_all_cells" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["region_all_cells"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        region_all_cells_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} region all cells density data...")
        
        # Parse file to get data
        region_all_cells_matrix, rac_header = parse_density_file(found_files["region_all_cells"])
        rac_shape = region_all_cells_matrix.shape
        print(f"Region all cells density shape: {rac_shape}")
        
        # Add source file information to header
        rac_header["source_file"] = os.path.basename(found_files["region_all_cells"])
        
        write_summary_stats(summary_file, region_all_cells_matrix, "Region All Cells Density", 
                           found_files["region_all_cells"], region_all_cells_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(region_all_cells_matrix, rac_header, region_all_cells_output)
        output_files.append(region_all_cells_output)
    
    # Process region_bounded_cells data file
    if "region_bounded_cells" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["region_bounded_cells"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        region_bounded_cells_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} region bounded cells density data...")
        
        # Parse file to get data
        region_bounded_cells_matrix, rbc_header = parse_density_file(found_files["region_bounded_cells"])
        rbc_shape = region_bounded_cells_matrix.shape
        print(f"Region bounded cells density shape: {rbc_shape}")
        
        # Add source file information to header
        rbc_header["source_file"] = os.path.basename(found_files["region_bounded_cells"])
        
        write_summary_stats(summary_file, region_bounded_cells_matrix, "Region Bounded Cells Density", 
                           found_files["region_bounded_cells"], region_bounded_cells_output)
        
        # Use the same plotting function as density map
        plot_density_map_continuous(region_bounded_cells_matrix, rbc_header, region_bounded_cells_output)
        output_files.append(region_bounded_cells_output)

    # Add the summary file to the output files list
    output_files.append(summary_file)
    
    # Process M layers horizontal congestion data
    if "horizontal_M_layers" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["horizontal_M_layers"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        horizontal_M_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} horizontal M layers congestion data...")
        
        # Parse file to get data
        horizontal_M_matrix, hm_header = parse_congestion_file(found_files["horizontal_M_layers"])
        hm_shape = horizontal_M_matrix.shape
        print(f"Horizontal M layers congestion shape: {hm_shape}")
        
        # Apply shape mask if available
        if shape_mask is not None:
            # Check if dimensions match
            if shape_mask.shape != horizontal_M_matrix.shape:
                # Need to resize the mask to match congestion data
                print(f"Resizing mask from {shape_mask.shape} to {horizontal_M_matrix.shape}")
                resized_mask = np.zeros(horizontal_M_matrix.shape, dtype=np.float32)
                
                # Copy the common area
                rows_min = min(shape_mask.shape[0], horizontal_M_matrix.shape[0])
                cols_min = min(shape_mask.shape[1], horizontal_M_matrix.shape[1])
                
                # Copy the valid part of the mask
                resized_mask[:rows_min, :cols_min] = shape_mask[:rows_min, :cols_min]
                current_mask = resized_mask
            else:
                current_mask = shape_mask
            
            print(f"Applying clean shape mask to horizontal M layers congestion data")
            # Create a clean masked version
            masked_horizontal_M = horizontal_M_matrix.copy()
            outside_region = (current_mask == 0)
            masked_horizontal_M[outside_region] = 0.0
            
            horizontal_M_matrix = masked_horizontal_M
        
        write_summary_stats(summary_file, horizontal_M_matrix, "Horizontal M Layers Congestion", 
                           found_files["horizontal_M_layers"], horizontal_M_output)
        
        # Add source file information to header
        header_to_use = density_header if density_header else hm_header
        header_to_use["source_file"] = os.path.basename(found_files["horizontal_M_layers"])
        
        plot_congestion_map(horizontal_M_matrix, header_to_use, "horizontal", horizontal_M_output)
        output_files.append(horizontal_M_output)
    
    # Process M layers vertical congestion data
    if "vertical_M_layers" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["vertical_M_layers"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        vertical_M_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} vertical M layers congestion data...")
        
        # Parse file to get data
        vertical_M_matrix, vm_header = parse_congestion_file(found_files["vertical_M_layers"])
        vm_shape = vertical_M_matrix.shape
        print(f"Vertical M layers congestion shape: {vm_shape}")
        
        # Apply shape mask if available
        if shape_mask is not None:
            # Check if dimensions match
            if shape_mask.shape != vertical_M_matrix.shape:
                # Need to resize the mask to match congestion data
                print(f"Resizing mask from {shape_mask.shape} to {vertical_M_matrix.shape}")
                resized_mask = np.zeros(vertical_M_matrix.shape, dtype=np.float32)
                
                # Copy the common area
                rows_min = min(shape_mask.shape[0], vertical_M_matrix.shape[0])
                cols_min = min(shape_mask.shape[1], vertical_M_matrix.shape[1])
                
                # Copy the valid part of the mask
                resized_mask[:rows_min, :cols_min] = shape_mask[:rows_min, :cols_min]
                shape_mask = resized_mask
            
            print(f"Applying clean shape mask to vertical M layers congestion data")
            # Create a clean masked version
            masked_vertical_M = vertical_M_matrix.copy()
            outside_region = (shape_mask == 0)
            masked_vertical_M[outside_region] = 0.0
            
            vertical_M_matrix = masked_vertical_M
        
        write_summary_stats(summary_file, vertical_M_matrix, "Vertical M Layers Congestion", 
                           found_files["vertical_M_layers"], vertical_M_output)
        
        # Add source file information to header
        header_to_use = density_header if density_header else vm_header
        header_to_use["source_file"] = os.path.basename(found_files["vertical_M_layers"])
        
        plot_congestion_map(vertical_M_matrix, header_to_use, "vertical", vertical_M_output)
        output_files.append(vertical_M_output)
    
    # Process average case horizontal congestion data
    if "horizontal_average" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["horizontal_average"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        horizontal_avg_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} horizontal average case congestion data...")
        
        # Parse file to get data
        horizontal_avg_matrix, ha_header = parse_congestion_file(found_files["horizontal_average"])
        ha_shape = horizontal_avg_matrix.shape
        print(f"Horizontal average case congestion shape: {ha_shape}")
        
        # Apply shape mask if available
        if shape_mask is not None:
            # Check if dimensions match
            if shape_mask.shape != horizontal_avg_matrix.shape:
                # Need to resize the mask to match congestion data
                print(f"Resizing mask from {shape_mask.shape} to {horizontal_avg_matrix.shape}")
                resized_mask = np.zeros(horizontal_avg_matrix.shape, dtype=np.float32)
                
                # Copy the common area
                rows_min = min(shape_mask.shape[0], horizontal_avg_matrix.shape[0])
                cols_min = min(shape_mask.shape[1], horizontal_avg_matrix.shape[1])
                
                # Copy the valid part of the mask
                resized_mask[:rows_min, :cols_min] = shape_mask[:rows_min, :cols_min]
                current_mask = resized_mask
            else:
                current_mask = shape_mask
            
            print(f"Applying clean shape mask to horizontal average case congestion data")
            # Create a clean masked version
            masked_horizontal_avg = horizontal_avg_matrix.copy()
            outside_region = (current_mask == 0)
            masked_horizontal_avg[outside_region] = 0.0
            
            horizontal_avg_matrix = masked_horizontal_avg
        
        write_summary_stats(summary_file, horizontal_avg_matrix, "Horizontal Average Case Congestion", 
                           found_files["horizontal_average"], horizontal_avg_output)
        
        # Add source file information to header
        header_to_use = density_header if density_header else ha_header
        header_to_use["source_file"] = os.path.basename(found_files["horizontal_average"])
        
        plot_congestion_map(horizontal_avg_matrix, header_to_use, "horizontal", horizontal_avg_output)
        output_files.append(horizontal_avg_output)
    
    # Process average case vertical congestion data
    if "vertical_average" in found_files:
        # Extract base filename from original filename, remove .data.gz suffix
        base_filename = os.path.basename(found_files["vertical_average"])
        if base_filename.endswith(".data.gz"):
            base_filename = base_filename[:-8]  # Remove .data.gz suffix
        
        vertical_avg_output = os.path.join(stage_dir, f"{base_filename}.png")
        print(f"Processing {stage_name} vertical average case congestion data...")
        
        # Parse file to get data
        vertical_avg_matrix, va_header = parse_congestion_file(found_files["vertical_average"])
        va_shape = vertical_avg_matrix.shape
        print(f"Vertical average case congestion shape: {va_shape}")
        
        # Apply shape mask if available
        if shape_mask is not None:
            # Check if dimensions match
            if shape_mask.shape != vertical_avg_matrix.shape:
                # Need to resize the mask to match congestion data
                print(f"Resizing mask from {shape_mask.shape} to {vertical_avg_matrix.shape}")
                resized_mask = np.zeros(vertical_avg_matrix.shape, dtype=np.float32)
                
                # Copy the common area
                rows_min = min(shape_mask.shape[0], vertical_avg_matrix.shape[0])
                cols_min = min(shape_mask.shape[1], vertical_avg_matrix.shape[1])
                
                # Copy the valid part of the mask
                resized_mask[:rows_min, :cols_min] = shape_mask[:rows_min, :cols_min]
                current_mask = resized_mask
            else:
                current_mask = shape_mask
            
            print(f"Applying clean shape mask to vertical average case congestion data")
            # Create a clean masked version
            masked_vertical_avg = vertical_avg_matrix.copy()
            outside_region = (current_mask == 0)
            masked_vertical_avg[outside_region] = 0.0
            
            vertical_avg_matrix = masked_vertical_avg
        
        write_summary_stats(summary_file, vertical_avg_matrix, "Vertical Average Case Congestion", 
                           found_files["vertical_average"], vertical_avg_output)
        
        # Add source file information to header
        header_to_use = density_header if density_header else va_header
        header_to_use["source_file"] = os.path.basename(found_files["vertical_average"])
        
        plot_congestion_map(vertical_avg_matrix, header_to_use, "vertical", vertical_avg_output)
        output_files.append(vertical_avg_output)
    
    # Print summary
    if output_files:
        print(f"Generated {len(output_files)-1} visualizations and 1 summary file for {stage_name} stage")
    else:
        print(f"Failed to generate any visualizations for {stage_name} stage")
    
    return output_files, stage_dir

def create_master_summary(reports_dir, stage_outputs, stages):
    """Create a master summary file for all stages."""
    ipo0_summary_file = os.path.join(reports_dir, "ipo0_analysis_summary.txt")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(ipo0_summary_file, 'w') as f:
        f.write("==== IPO0 ANALYSIS MASTER SUMMARY ====\n")
        f.write(f"Generated on: {timestamp}\n\n")
        
        for stage in stages:
            f.write(f"=== {stage.upper()} STAGE ===\n")
            files = stage_outputs[stage]["files"]
            f.write(f"Generated {len(files)-1} visualizations and 1 summary file\n")
            f.write(f"See detailed results in: {os.path.join(os.path.basename(reports_dir), f'{stage}_reports', f'{stage}_analysis_summary.txt')}\n\n")
    
    print(f"\nMaster summary generated: {ipo0_summary_file}")
    return ipo0_summary_file

def find_report_files(base_path, stage_name, filename_patterns):
    """Find report files for a specific stage using the provided filename patterns."""
    import glob
    import os
    
    report_files = {}
    
    for report_type, pattern_template in filename_patterns.items():
        # Substitute stage name into pattern
        pattern = pattern_template.format(stage=stage_name)
        
        # Find matching files
        matching_files = glob.glob(os.path.join(base_path, pattern))
        
        if matching_files:
            # Use the first matching file
            report_files[report_type] = matching_files[0]
    
    return report_files 