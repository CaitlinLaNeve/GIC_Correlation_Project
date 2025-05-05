import os
import zipfile
import pandas as pd

# Paths
input_dir = r'C:/664/gic/1Done'
output_dir = r'C:/664/gic/ExtractedGIC'
monitor_location_csv = r'C:/664/gic/monitorlocation.csv'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load device location data
location_df = pd.read_csv(monitor_location_csv)
location_dict = location_df.set_index('GICDeviceID')[['Latitude', 'Longitude']].to_dict('index')

# Iterate over zip files
for filename in os.listdir(input_dir):
    if filename.endswith('.zip') and filename.startswith('2024E04_'):
        zip_path = os.path.join(input_dir, filename)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Assume there's only one CSV file in each ZIP
            for zip_info in zip_ref.infolist():
                with zip_ref.open(zip_info) as file:
                    df = pd.read_csv(file)

                    # Skip if required columns are not present
                    if not {'GICDeviceID', 'SampleDateTime', 'GICMeasured'}.issubset(df.columns):
                        print(f"Skipping {filename} — missing required columns.")
                        continue

                    gic_id = df['GICDeviceID'].iloc[0]
                    if gic_id not in location_dict:
                        print(f"Skipping {filename} — GICDeviceID {gic_id} not found in location file.")
                        continue

                    # Add Latitude and Longitude columns
                    lat, lon = location_dict[gic_id]['Latitude'], location_dict[gic_id]['Longitude']
                    df.insert(1, 'Latitude', lat)
                    df.insert(2, 'Longitude', lon)

                    # Save file with just the GICDeviceID as filename
                    output_file = os.path.join(output_dir, f"{gic_id}.csv")
                    df.to_csv(output_file, index=False)
                    print(f"Saved {output_file}")
