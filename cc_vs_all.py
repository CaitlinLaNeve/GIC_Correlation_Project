import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if input files exist
if not os.path.exists('monitor_line_metrics_by_buffer.csv'):
    raise FileNotFoundError("The file 'monitor_line_metrics_by_buffer.csv' was not found.")
if not os.path.exists('lagged_correlation_pairs.csv'):
    raise FileNotFoundError("The file 'lagged_correlation_pairs.csv' was not found.")

# Load the full monitor metrics
monitor_stats = pd.read_csv('monitor_line_metrics_by_buffer.csv')

# Handle missing data in monitor_stats
monitor_stats = monitor_stats.fillna(0)  # Replace NaN with 0 (or use dropna() if appropriate)

# Validate required columns in monitor_stats
required_columns = ['GICDeviceID'] + [f'Num_NSLines_{buffer}' for buffer in ['500', '1000', '1500']] + \
                   [f'NumTotalLines_{buffer}' for buffer in ['500', '1000', '1500']] + \
                   [f'TotalLineLength_{buffer}' for buffer in ['500', '1000', '1500']] + \
                   [f'VoltageWeightedLength_{buffer}' for buffer in ['500', '1000', '1500']]
if not set(required_columns).issubset(monitor_stats.columns):
    raise ValueError("The file 'monitor_line_metrics_by_buffer.csv' is missing required columns.")

# Load correlation pairs
corr_df = pd.read_csv('lagged_correlation_pairs.csv')

# Handle missing data in corr_df
corr_df = corr_df.dropna(subset=['Device1', 'Device2'])  # Drop rows with missing device IDs

buffers = ['500', '1000', '1500']

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

for buffer in buffers:
    # Subset relevant columns for the buffer
    cols = ['GICDeviceID',
            f'Num_NSLines_{buffer}',
            f'NumTotalLines_{buffer}',
            f'TotalLineLength_{buffer}',
            f'VoltageWeightedLength_{buffer}']
    stats = monitor_stats[cols].copy()
    
    # Merge stats for Device1
    merged = corr_df.merge(stats, left_on='Device1', right_on='GICDeviceID', how='left')
    merged = merged.rename(columns={
        f'Num_NSLines_{buffer}': 'NS1',
        f'NumTotalLines_{buffer}': 'Total1',
        f'TotalLineLength_{buffer}': 'Len1',
        f'VoltageWeightedLength_{buffer}': 'VoltLen1'
    }).drop(columns=['GICDeviceID'])

    # Merge stats for Device2
    merged = merged.merge(stats, left_on='Device2', right_on='GICDeviceID', how='left')
    merged = merged.rename(columns={
        f'Num_NSLines_{buffer}': 'NS2',
        f'NumTotalLines_{buffer}': 'Total2',
        f'TotalLineLength_{buffer}': 'Len2',
        f'VoltageWeightedLength_{buffer}': 'VoltLen2'
    }).drop(columns=['GICDeviceID'])

    # Compute derived stats
    merged[f'Sum_NSLines_{buffer}'] = merged['NS1'] + merged['NS2']
    merged[f'Diff_NSLines_{buffer}'] = abs(merged['NS1'] - merged['NS2'])

    merged[f'Sum_TotalLines_{buffer}'] = merged['Total1'] + merged['Total2']
    merged[f'Diff_TotalLines_{buffer}'] = abs(merged['Total1'] - merged['Total2'])

    merged[f'Sum_LineLength_{buffer}'] = merged['Len1'] + merged['Len2']
    merged[f'Diff_LineLength_{buffer}'] = abs(merged['Len1'] - merged['Len2'])

    merged[f'Sum_VoltageWeightedLength_{buffer}'] = merged['VoltLen1'] + merged['VoltLen2']
    merged[f'Diff_VoltageWeightedLength_{buffer}'] = abs(merged['VoltLen1'] - merged['VoltLen2'])

    # Save to CSV
    out_file = os.path.join(output_dir, f'correlation_with_line_stats_{buffer}m.csv')
    merged.to_csv(out_file, index=False)
    logging.info(f"Saved: {out_file}")
