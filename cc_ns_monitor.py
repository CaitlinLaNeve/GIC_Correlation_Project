import pandas as pd
import ast
import numpy as np

def is_ns_orientation(angle, tolerance=15):
    if angle is None:
        return False
    angle = float(angle) % 360
    return (abs(angle - 0) <= tolerance) or (abs(angle - 180) <= tolerance)

def process_monitor_linedata(csv_path, buffer_name):
    df = pd.read_csv(csv_path)

    results = []
    for _, row in df.iterrows():
        device_id = row['GICDeviceID']
        line_data_str = row['LineData']

        try:
            lines = ast.literal_eval(line_data_str) if pd.notna(line_data_str) else []
        except:
            lines = []

        ns_lines = 0
        total_length = 0
        voltage_weighted_length = 0

        for line in lines:
            orientation = line.get('Orientation')
            length = line.get('Length', 0) or 0
            voltage = line.get('Voltage', 0) or 0

            if is_ns_orientation(orientation):
                ns_lines += 1

            total_length += length
            voltage_weighted_length += length * voltage

        results.append({
            'GICDeviceID': device_id,
            f'Num_NSLines_{buffer_name}': ns_lines,
            f'TotalLineLength_{buffer_name}': total_length,
            f'VoltageWeightedLength_{buffer_name}': voltage_weighted_length
        })

    return pd.DataFrame(results)
