#!/usr/bin/env python3
"""
ADNI FreeSurfer Data Compiler
Extracts cortical thickness (68 regions) and brain volume from FreeSurfer stats files
across all timepoints and subjects.
"""

import os
import re
import pandas as pd
from pathlib import Path


def parse_aparc_stats(stats_file):
    """
    Parse aparc.stats file to extract cortical thickness for each region.
    Returns a dictionary of {region_name: thickness_value}
    """
    thickness_dict = {}
    
    try:
        with open(stats_file, 'r') as f:
            lines = f.readlines()
        
        # Find the data section (after the header)
        data_started = False
        for line in lines:
            # Skip comments and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            
            # Data lines have the structure:
            # StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
            parts = line.split()
            if len(parts) >= 5:
                region_name = parts[0]
                thickness = parts[4]  # ThickAvg is the 5th column (index 4)
                try:
                    thickness_dict[region_name] = float(thickness)
                except ValueError:
                    continue
    
    except Exception as e:
        print(f"Error parsing {stats_file}: {e}")
    
    return thickness_dict


def parse_brainvol_stats(stats_file):
    """
    Parse brainvol.stats to extract total brain volume measures.
    Returns a dictionary of volume measurements.
    """
    vol_dict = {}
    
    try:
        with open(stats_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            # Look for specific volume measures
            # Format: # Measure key, value, units, description
            if line.startswith('# Measure'):
                parts = line.split(',')
                if len(parts) >= 3:
                    # Extract measure name and value
                    measure_info = parts[0].replace('# Measure', '').strip()
                    value_str = parts[1].strip()
                    
                    # Common measures we want
                    if 'BrainSeg' in measure_info or 'BrainSegVol' in measure_info:
                        try:
                            vol_dict['BrainSegVol'] = float(value_str)
                        except ValueError:
                            pass
                    elif 'eTIV' in measure_info or 'EstimatedTotalIntraCranialVol' in measure_info:
                        try:
                            vol_dict['eTIV'] = float(value_str)
                        except ValueError:
                            pass
    
    except Exception as e:
        print(f"Error parsing {stats_file}: {e}")
    
    return vol_dict


def extract_subject_data(subject_path, timepoint):
    """
    Extract all relevant data for a single subject at a given timepoint.
    Returns a dictionary with all measurements.
    """
    data = {
        'subjid': subject_path.name,
        'visit': timepoint
    }
    
    stats_dir = subject_path / 'stats'
    
    if not stats_dir.exists():
        print(f"Warning: stats directory not found for {subject_path}")
        return None
    
    # Parse left hemisphere aparc.stats
    lh_aparc = stats_dir / 'lh.aparc.stats'
    if lh_aparc.exists():
        lh_thickness = parse_aparc_stats(lh_aparc)
        # Add 'lh_' prefix to region names
        for region, thickness in lh_thickness.items():
            data[f'lh_{region}_thickness'] = thickness
    else:
        print(f"Warning: {lh_aparc} not found")
    
    # Parse right hemisphere aparc.stats
    rh_aparc = stats_dir / 'rh.aparc.stats'
    if rh_aparc.exists():
        rh_thickness = parse_aparc_stats(rh_aparc)
        # Add 'rh_' prefix to region names
        for region, thickness in rh_thickness.items():
            data[f'rh_{region}_thickness'] = thickness
    else:
        print(f"Warning: {rh_aparc} not found")
    
    # Parse brainvol.stats
    brainvol = stats_dir / 'brainvol.stats'
    if brainvol.exists():
        vol_data = parse_brainvol_stats(brainvol)
        data.update(vol_data)
    else:
        print(f"Warning: {brainvol} not found")
    
    return data


def compile_adni_data(base_dir, output_csv='adni_compiled.csv'):
    """
    Main function to compile all ADNI data across timepoints and subjects.
    
    Args:
        base_dir: Path to the directory containing timepoint folders (12MO, 24MO, etc.)
        output_csv: Name of output CSV file
    """
    base_path = Path(base_dir)
    
    # Define timepoint folders
    timepoints = ['12MO', '24MO', '36MO', '48MO', '60MO', '72MO', 'over72MO', 'SC']
    
    all_data = []
    
    print("Starting data extraction...")
    
    for timepoint in timepoints:
        timepoint_dir = base_path / timepoint / 'FS_output'
        
        if not timepoint_dir.exists():
            print(f"Warning: {timepoint_dir} does not exist, skipping...")
            continue
        
        print(f"\nProcessing timepoint: {timepoint}")
        
        # Get all subject directories
        subject_dirs = [d for d in timepoint_dir.iterdir() if d.is_dir()]
        
        print(f"  Found {len(subject_dirs)} subjects")
        
        for subject_dir in subject_dirs:
            print(f"  Processing subject: {subject_dir.name}")
            
            subject_data = extract_subject_data(subject_dir, timepoint)
            
            if subject_data:
                all_data.append(subject_data)
    
    # Create DataFrame
    print(f"\n\nCreating DataFrame with {len(all_data)} rows...")
    df = pd.DataFrame(all_data)
    
    # Reorder columns: subjid, visit, then brain volumes, then thickness measurements
    cols = ['subjid', 'visit']
    
    # Add brain volume columns
    vol_cols = [col for col in df.columns if 'Vol' in col or 'eTIV' in col]
    cols.extend(sorted(vol_cols))
    
    # Add thickness columns (sorted)
    thickness_cols = [col for col in df.columns if 'thickness' in col]
    cols.extend(sorted(thickness_cols))
    
    # Reorder
    df = df[cols]
    
    # Save to CSV
    output_path = Path(output_csv)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Data compilation complete!")
    print(f"Output saved to: {output_path.absolute()}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumn summary:")
    print(f"  - Subject/Visit info: 2 columns")
    print(f"  - Brain volumes: {len(vol_cols)} columns")
    print(f"  - Thickness measurements: {len(thickness_cols)} columns")
    
    return df


if __name__ == "__main__":

    base_directory = "/data02/bgutman/longADNI_FSrun2025"
    
    print("ADNI FreeSurfer Data Compiler")
    print(f"Base directory: {base_directory}")
    print()
    
    df = compile_adni_data(base_directory, output_csv='adni_longitudinal.csv')
    
    print("\nFirst few rows:")
    print(df.head())
    print("\nData shape:", df.shape)