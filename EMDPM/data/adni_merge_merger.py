#!/usr/bin/env python3

import pandas as pd
import re
import sys

def extract_ptid_from_full_id(full_id):
    """
    Extract PTID from full ADNI ID format like: SITE__PTID__SCANID
    Example: 002_S_0295 from 002__002_S_0295__something
    """
    parts = str(full_id).split('__')
    if len(parts) >= 2:
        return parts[1]
    return full_id

def map_visit_codes(imaging_visit):
    """
    Map imaging timepoint codes to ADNIMERGE VISCODE format
    12MO -> m12, 24MO -> m24, SC -> sc, etc.
    """
    visit_map = {
        'SC': 'sc',
        '12MO': 'm12',
        '24MO': 'm24',
        '36MO': 'm36',
        '48MO': 'm48',
        '60MO': 'm60',
        '72MO': 'm72',
        'over72MO': 'm84'  # Approximation, may need adjustment
    }
    return visit_map.get(imaging_visit, imaging_visit.lower())

def merge_adni_clinical(imaging_csv, adnimerge_csv, output_csv='adni_complete.csv'):
    """
    Merge imaging data with clinical data from ADNIMERGE
    """
    
    print("Loading imaging data...")
    imaging_df = pd.read_csv(imaging_csv)
    print(f"  Loaded {len(imaging_df)} imaging records")
    
    print("\nLoading ADNIMERGE clinical data...")
    clinical_df = pd.read_csv(adnimerge_csv)
    print(f"  Loaded {len(clinical_df)} clinical records")
    
    # Extract PTID from full imaging ID if needed
    if 'subjid' in imaging_df.columns:
        if '__' in str(imaging_df['subjid'].iloc[0]):
            print("\nExtracting PTID from full subject ID...")
            imaging_df['PTID'] = imaging_df['subjid'].apply(extract_ptid_from_full_id)
        else:
            imaging_df['PTID'] = imaging_df['subjid']
    
    # Map visit codes
    if 'visit' in imaging_df.columns:
        print("Mapping visit codes...")
        imaging_df['VISCODE'] = imaging_df['visit'].apply(map_visit_codes)
    
    # Select clinical variables of interest
    clinical_vars = [
        'PTID', 'VISCODE',
        # Time variables
        'Years_bl', 'Month_bl', 'Month', 'EXAMDATE',
        # Demographics
        'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4',
        # Diagnosis
        'DX', 'DX_bl',
        # Cognitive scores
        'MMSE', 'MOCA', 'ADAS11', 'ADAS13', 'CDRSB', 'FAQ',
        # Memory scores
        'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
        # Biomarkers (if available)
        'ABETA', 'TAU', 'PTAU',
        # PET imaging
        'FDG', 'PIB', 'AV45',
        # Functional assessment
        'EcogPtTotal', 'EcogSPTotal',
        # Composite scores
        'mPACCdigit', 'mPACCtrailsB',
        # Baseline values for key measures
        'MMSE_bl', 'MOCA_bl', 'ADAS11_bl', 'CDRSB_bl'
    ]
    
    # Keep only variables that exist in clinical_df
    available_vars = [var for var in clinical_vars if var in clinical_df.columns]
    print(f"\nFound {len(available_vars)} clinical variables in ADNIMERGE")
    
    clinical_subset = clinical_df[available_vars].copy()
    
    # Merge on PTID and VISCODE
    print("\nMerging imaging and clinical data...")
    merged_df = imaging_df.merge(
        clinical_subset,
        on=['PTID', 'VISCODE'],
        how='left'
    )
    
    print(f"  Merged dataset: {len(merged_df)} records")
    
    # Calculate match statistics
    n_matched = merged_df['MMSE'].notna().sum()
    match_rate = (n_matched / len(merged_df)) * 100
    print(f"  Successfully matched clinical data for {n_matched} records ({match_rate:.1f}%)")
    
    # Reorder columns: imaging ID, PTID, visit info, clinical, then brain measures
    id_cols = ['subjid', 'PTID', 'visit', 'VISCODE']
    time_cols = [c for c in merged_df.columns if 'Years_bl' in c or 'Month' in c or 'EXAMDATE' in c]
    demo_cols = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4']
    dx_cols = [c for c in merged_df.columns if 'DX' in c]
    cog_cols = [c for c in merged_df.columns if any(x in c for x in ['MMSE', 'MOCA', 'ADAS', 'CDRSB', 'FAQ', 'RAVLT', 'mPACC', 'Ecog'])]
    bio_cols = [c for c in merged_df.columns if any(x in c for x in ['ABETA', 'TAU', 'FDG', 'PIB', 'AV45'])]
    brain_cols = [c for c in merged_df.columns if 'thickness' in c or 'Vol' in c or 'eTIV' in c]
    
    # Remove duplicates and keep order
    all_ordered_cols = []
    for col_list in [id_cols, time_cols, demo_cols, dx_cols, cog_cols, bio_cols, brain_cols]:
        for col in col_list:
            if col in merged_df.columns and col not in all_ordered_cols:
                all_ordered_cols.append(col)
    
    # Add any remaining columns
    for col in merged_df.columns:
        if col not in all_ordered_cols:
            all_ordered_cols.append(col)
    
    merged_df = merged_df[all_ordered_cols]
    
    # Save
    merged_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved complete dataset to: {output_csv}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("DATA COMPLETENESS SUMMARY")
    print("="*60)
    
    key_vars = ['MMSE', 'MOCA', 'ADAS11', 'CDRSB', 'FAQ', 'Years_bl', 'AGE', 'APOE4']
    print("\nKey clinical variables:")
    for var in key_vars:
        if var in merged_df.columns:
            n_complete = merged_df[var].notna().sum()
            pct_complete = (n_complete / len(merged_df)) * 100
            print(f"  {var:20s}: {n_complete:5d} / {len(merged_df)} ({pct_complete:5.1f}% complete)")
    
    print(f"\nBrain measures: {len(brain_cols)} columns")
    print(f"Total columns: {len(merged_df.columns)}")
    
    return merged_df


if __name__ == "__main__":
    
    print("ADNI Data Merger: Imaging + Clinical")
    print("="*60)
    
    # Get file paths from command line or use defaults
    if len(sys.argv) >= 3:
        imaging_file = sys.argv[1]
        clinical_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else "adni_complete.csv"
    else:
        print("\nUsage: python merge_adni_clinical.py <imaging_csv> <adnimerge_csv> [output_csv]")
        print("\nExample:")
        print("  python merge_adni_clinical.py /path/to/adni_longitudinal.csv /path/to/ADNIMERGE.csv adni_complete.csv")
        print("\nUsing default filenames...")
        imaging_file = "/home/dsemchin/data/adni_longitudinal.csv"
        clinical_file = "/data01/bgutman/ADNI/collections_data/ADNIMERGE.csv"
        output_file = "/home/dsemchin/data/adni_complete.csv"
    
    print(f"\nInput files:")
    print(f"  Imaging: {imaging_file}")
    print(f"  Clinical: {clinical_file}")
    print(f"  Output: {output_file}")
    print()
    
    # Run merger
    df = merge_adni_clinical(imaging_file, clinical_file, output_file)
    
    print("\n✓ Merge complete!")
    print(f"\nPreview of merged data:")
    print(df[['subjid', 'PTID', 'visit', 'VISCODE', 'Years_bl', 'MMSE', 'MOCA', 'AGE']].head(10))