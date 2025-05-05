import os
import pandas as pd
import glob
import re


csv_files_path = "C:/664/gic/TVA/TVA_Measurements/*.csv"
reference_data_path = "C:/664/gic/monitorlocation.csv"

print("Reading reference data...")
reference_df = pd.read_csv(reference_data_path)

# Print column names exactly as they appear
print("Columns in reference data (with exact spacing):")
print([f"'{col}'" for col in reference_df.columns.tolist()])

# Clean up column names by stripping whitespace
reference_df.columns = reference_df.columns.str.strip()
print("Columns after stripping whitespace:")
print(reference_df.columns.tolist())

# Convert GICDeviceID to string
reference_df['GICDeviceID'] = reference_df['GICDeviceID'].astype(str)

print(f"Reference data contains {len(reference_df)} records")

# Create a dictionary for quick lookup of device information
device_info = {}
for _, row in reference_df.iterrows():
    device_id = str(row['GICDeviceID']).strip()  # Convert to string and remove whitespace
    device_info[device_id] = {
        'Latitude': row['Latitude'],
        'Longitude': row['Longitude']
    }

print(f"Created lookup for {len(device_info)} devices")

# Process each CSV file
csv_files = glob.glob(csv_files_path)
print(f"Found {len(csv_files)} CSV files to process.")

processed_count = 0
skipped_count = 0

for file_path in csv_files:
    # Extract the device ID from the filename using regex
    filename = os.path.basename(file_path)
    
    # Extract GICDeviceID from filename pattern 2024E04_XXXXX
    match = re.search(r'2024E04_(\d+)', filename)
    if match:
        device_id = match.group(1).strip()
    else:
        # Fall back to original method if pattern doesn't match
        device_id = filename.split('.')[0].strip()
    
    print(f"Processing file: {filename} (extracted device ID: {device_id})")
    
    # Skip if device ID not found in reference data
    if device_id not in device_info:
        print(f"  Warning: Device ID {device_id} not found in reference data. Skipping file.")
        skipped_count += 1
        continue
    
    try:
        # Read the CSV file - these files already have headers
        df = pd.read_csv(file_path)
        
        # Add location columns from reference data
        df['Latitude'] = device_info[device_id]['Latitude']
        df['Longitude'] = device_info[device_id]['Longitude']
        
        # Attempt to reorder columns if standard names exist
        try:
            if all(col in df.columns for col in ['GICDeviceID', 'SampleDateTime', 'GICMeasured']):
                df = df[['GICDeviceID', 'Latitude', 'Longitude', 'SampleDateTime', 'GICMeasured']]
        except Exception as e:
            print(f"  Could not reorder columns: {e}. Continuing with original order plus lat/long.")
        
        # Save the updated file with just the device ID as filename
        output_path = os.path.join(os.path.dirname(file_path), f"{device_id}.csv")
        df.to_csv(output_path, index=False)
        print(f"  Successfully processed. Saved to: {device_id}.csv")
        processed_count += 1
        
    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        skipped_count += 1

print(f"Processing complete! Successfully processed {processed_count} files, skipped {skipped_count} files.")