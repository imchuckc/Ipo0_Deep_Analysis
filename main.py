#!/usr/bin/env python3
"""
FPGA Layout Report Analyzer

This tool processes FPGA layout reports (density and congestion data)
and generates visualizations with statistical analysis.
"""
import os
import sys
import argparse
from scripts.utils import create_timestamped_dir
from scripts.report_processor import process_reports_by_stage, create_master_summary
from scripts import config

def main():
    """Main function to process report files and generate visualizations."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='FPGA Layout Report Analyzer')
    parser.add_argument('--base-path', dest='base_path', type=str,
                        help='Base path to search for reports')
    parser.add_argument('--prefix', dest='prefix', type=str,
                        help='Prefix for output directory')
    parser.add_argument('--stages', dest='stages', type=str, nargs='+',
                        help='Stages to process (e.g., place route)')
    
    args = parser.parse_args()
    
    # Define base data path from config or command line
    base_path = args.base_path if args.base_path else config.BASE_DATA_PATH
    
    # Check if path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Base path '{base_path}' not found.")
        print("Please update the BASE_DATA_PATH in config.py or specify with --base-path")
        sys.exit(1)
    
    # Define output directory (same as script directory)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create timestamped reports directory
    prefix = args.prefix if args.prefix else config.REPORT_DIR_PREFIX
    reports_dir = create_timestamped_dir(output_dir, prefix=prefix)
    
    # Process reports for different stages
    stages = args.stages if args.stages else config.STAGES
    stage_outputs = {}
    
    # Print configuration summary
    print("=== FPGA Layout Report Analyzer ===")
    print(f"Base path: {base_path}")
    print(f"Output directory: {reports_dir}")
    print(f"Stages to process: {', '.join(stages)}")
    print("=================================\n")
    
    for stage in stages:
        print(f"\n===== Processing {stage} stage reports =====")
        output_files, stage_dir = process_reports_by_stage(stage, base_path, reports_dir)
        stage_outputs[stage] = {
            "files": output_files,
            "dir": stage_dir
        }
    
    # Create the master summary file
    create_master_summary(reports_dir, stage_outputs, stages)
    print(f"All reports saved to: {reports_dir}")

if __name__ == "__main__":
    main()
