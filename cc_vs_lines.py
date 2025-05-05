import pandas as pd
import json
import numpy as np

def parse_line_data(line_str):
    try:
        lines = json.loads(line_str.replace("'", '"'))
        valid_lines = [line for line in lines if line.get("Orientation") is not None]
        total = len(valid_lines)
        ns_count = sum(1 for line in valid_lines if is_ns(line["Orientation"]))
        return pd.Series([total, ns_count])
    except Exception:
        return pd.Series([0, 0])

def is_ns(orientation, tol=30):
    """Returns True if the orientation is considered North-South."""
    angle = orientation % 360
    return angle < tol or abs(angle - 180) < tol

def process_buffer_file(filepath):
    df = pd.read_csv(filepath)
    df[['TotalLines', 'NSLines']] = df['LineData'].apply(parse_line_data)
    return df[['GICDeviceID', 'TotalLines', 'NSLines']]

#buffer_500 = process_buffer_file("C:/664/gic/export/monitorlinedata500.csv")
#buffer_1000 = process_buffer_file("C:/664/gic/export/monitorlinedata1000.csv")
buffer_1500 = process_buffer_file("C:/664/gic/monitorlinedata1500.csv")

corr_df = pd.read_csv("C:/664/gic/lagged_correlation_pairs.csv")

def join_buffer_stats(corr_df, buffer_df, suffix):
    merged = corr_df.copy()
    merged = merged.merge(buffer_df, left_on='Device1', right_on='GICDeviceID', how='left')
    merged = merged.rename(columns={'TotalLines': f'Device1_TotalLines_{suffix}', 'NSLines': f'Device1_NSLines_{suffix}'})
    merged = merged.drop(columns='GICDeviceID')
    merged = merged.merge(buffer_df, left_on='Device2', right_on='GICDeviceID', how='left')
    merged = merged.rename(columns={'TotalLines': f'Device2_TotalLines_{suffix}', 'NSLines': f'Device2_NSLines_{suffix}'})
    merged = merged.drop(columns='GICDeviceID')
    return merged

#corr_with_500 = join_buffer_stats(corr_df, buffer_500, "500m")
#corr_with_1000 = join_buffer_stats(corr_df, buffer_1000, "1000m")
corr_with_1500 = join_buffer_stats(corr_df, buffer_1500, "1500m")

#corr_with_500.to_csv("correlation_with_line_stats_500m.csv", index=False)
#corr_with_1000.to_csv("correlation_with_line_stats_1000m.csv", index=False)
corr_with_1500.to_csv("correlation_with_line_stats_1500m.csv", index=False)
