import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("gic_processing.log"),
                              logging.StreamHandler()])

# Paths
input_dir = Path('C:/664/gic/ExtractedGIC')
location_csv = Path('C:/664/gic/monitorlocation.csv')
output_dir = Path('C:/664/gic/results')
output_dir.mkdir(exist_ok=True)  # Create output directory if it doesn't exist

output_corr_csv = output_dir / 'correlation_pairs.csv'
output_lagged_csv = output_dir / 'lagged_correlation_pairs.csv'
output_plot = output_dir / 'high_correlation_links.png'

resample_interval = '1min'  # 1-minute
timestamp_format = "%m/%d/%Y %I:%M:%S %p"

# --- Part 1: Load and resample data ---
device_series = {}
expected_columns = ['SampleDateTime', 'GICMeasured']

logging.info(f"Processing files from {input_dir}")
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        path = os.path.join(input_dir, filename)
        try:
            # Extract device ID from filename
            gic_id = Path(filename).stem
            logging.info(f"Processing device {gic_id} from {filename}")
            
            df = pd.read_csv(path)
            
            # Check if expected columns exist
            for col in expected_columns:
                if col not in df.columns:
                    logging.warning(f"Column '{col}' not found in {filename}")
                    raise KeyError(f"Missing required column: {col}")
            
            # Explicitly parse timestamps
            df['SampleDateTime'] = pd.to_datetime(df['SampleDateTime'], format=timestamp_format, errors='coerce')
            df = df.dropna(subset=['SampleDateTime'])  # Drop rows with unparseable times
            
            if df.empty:
                logging.warning(f"No valid data in {filename} after parsing timestamps")
                continue
                
            # Select and process the data
            df = df[expected_columns].dropna()
            df = df.set_index('SampleDateTime').sort_index()
            
            # Check if we have valid data
            if df.empty:
                logging.warning(f"No valid data for device {gic_id} after filtering")
                continue
                
            resampled = df.resample(resample_interval).mean()
            device_series[gic_id] = resampled['GICMeasured']
            logging.info(f"Successfully processed {filename}: {len(resampled)} datapoints")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            continue

# Check if we have any data
if not device_series:
    logging.error("No valid data was processed. Exiting.")
    exit(1)

# Combine all devices
logging.info(f"Combining data from {len(device_series)} devices")
combined_df = pd.concat(device_series, axis=1)
combined_df.columns.name = 'GICDeviceID'
combined_df = combined_df.dropna(how='all')
logging.info(f"Combined data shape: {combined_df.shape}")

# Calculate correlation matrix
correlation_matrix = combined_df.corr()
logging.info(f"Calculated correlation matrix of shape {correlation_matrix.shape}")

# Check for missing values in correlation matrix
nan_count = correlation_matrix.isna().sum().sum()
if nan_count > 0:
    logging.warning(f"Found {nan_count} NaN values in correlation matrix. This could indicate insufficient overlapping data.")

# --- Part 2: Save correlation pairs ---
correlation_pairs = []

for dev1, dev2 in combinations(correlation_matrix.columns, 2):
    corr_value = correlation_matrix.loc[dev1, dev2]
    if not pd.isna(corr_value):  # Skip NaN values
        correlation_pairs.append((dev1, dev2, corr_value))

corr_df = pd.DataFrame(correlation_pairs, columns=['Device1', 'Device2', 'Correlation'])
corr_df = corr_df.sort_values(by='Correlation', ascending=False)
corr_df.to_csv(output_corr_csv, index=False)
logging.info(f"Saved {len(correlation_pairs)} correlation pairs to {output_corr_csv}")

# --- Part 3: Calculate time-lagged correlations ---
max_lag = 60  # max lag in minutes
lagged_results = []

logging.info(f"Calculating time-lagged correlations with max lag of {max_lag} minutes")
pair_count = 0
skipped_count = 0
error_count = 0

for dev1, dev2 in combinations(device_series.keys(), 2):
    pair_count += 1
    if pair_count % 100 == 0:
        logging.info(f"Processed {pair_count} device pairs for lagged correlation")
        
    series1 = device_series[dev1]
    series2 = device_series[dev2]

    # Join on time index (inner join drops any unmatched timestamps)
    pair_df = pd.concat([series1, series2], axis=1, join='inner')
    pair_df.columns = ['s1', 's2']
    pair_df.dropna(inplace=True)

    # Require at least 30 overlapping data points
    if len(pair_df) < 30:
        logging.debug(f"Skipping pair {dev1}-{dev2} due to insufficient data points ({len(pair_df)} points)")
        skipped_count += 1
        continue
    
    # Calculate regular Pearson correlation first
    pearson_corr = pair_df['s1'].corr(pair_df['s2'])
    
    try:
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        for lag in lags:
            if lag == 0:
                # Already calculated this
                correlations.append(pearson_corr)
                continue
                
            shifted = pair_df['s2'].shift(lag)
            corr = pair_df['s1'].corr(shifted)
            correlations.append(corr)

        # Check if all correlations are NaN
        if all(pd.isna(c) for c in correlations):
            logging.debug(f"All correlations are NaN for pair {dev1}-{dev2}")
            error_count += 1
            continue
            
        correlations_array = np.array(correlations)
        valid_correlations = ~np.isnan(correlations_array)
        
        if not any(valid_correlations):
            logging.debug(f"No valid correlations for pair {dev1}-{dev2}")
            error_count += 1
            continue
            
        best_corr_idx = np.nanargmax(np.abs(correlations_array))  # Find max by absolute value
        best_corr = correlations_array[best_corr_idx]
        best_lag = lags[best_corr_idx]

        lagged_results.append({
            'Device1': dev1,
            'Device2': dev2,
            'PearsonCorrelation': pearson_corr,
            'MaxLaggedCorrelation': best_corr,
            'LagMinutes': best_lag
        })
    except Exception as e:
        logging.debug(f"Error calculating lagged correlation for pair {dev1}-{dev2}: {str(e)}")
        error_count += 1
        continue

logging.info(f"Lagged correlation statistics: processed {pair_count} pairs, skipped {skipped_count}, errors {error_count}")
# Save lagged correlation results
if lagged_results:
    lagged_df = pd.DataFrame(lagged_results)
    lagged_df.to_csv(output_lagged_csv, index=False)
    logging.info(f"Saved {len(lagged_results)} lagged correlation pairs to {output_lagged_csv}")
else:
    logging.warning("No valid lagged correlation results to save")

# --- Part 4: Create correlation visualizations ---
try:
    # Plot heatmap of correlation matrix
    plt.figure(figsize=(12, 10))
    try:
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Create mask for upper triangle
        
        # Adjust annotation settings based on matrix size
        annot = True
        fontsize = 10
        if correlation_matrix.shape[0] > 15:  # For large matrices
            annot = False
            logging.info("Matrix too large for annotations, displaying heatmap without values")
        elif correlation_matrix.shape[0] > 10:  # For medium matrices
            fontsize = 8
            
        sns.heatmap(correlation_matrix, 
                    annot=annot, 
                    fmt=".2f",
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1,
                    linewidths=0.5,
                    mask=mask,  # Only show lower triangle
                    annot_kws={"size": fontsize})
        
        plt.title('GIC Monitor Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300)
        logging.info(f"Saved correlation heatmap to {output_dir / 'correlation_heatmap.png'}")
    except Exception as e:
        logging.error(f"Error creating correlation heatmap: {str(e)}")
    
    # Save a CSV of the correlation matrix for reference
    correlation_matrix.to_csv(output_dir / 'correlation_matrix.csv')
    logging.info(f"Saved correlation matrix to {output_dir / 'correlation_matrix.csv'}")
    
    # Create a histogram of correlation values
    try:
        plt.figure(figsize=(10, 6))
        corr_values = correlation_matrix.values[~np.isnan(correlation_matrix.values) & (correlation_matrix.values != 1.0)]
        plt.hist(corr_values, bins=50, alpha=0.7, color='steelblue')
        plt.title('Distribution of Correlation Values')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'correlation_distribution.png', dpi=300)
        logging.info(f"Saved correlation distribution histogram to {output_dir / 'correlation_distribution.png'}")
    except Exception as e:
        logging.error(f"Error creating correlation distribution: {str(e)}")

    # Try to create the map visualization if geopandas is available
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Point
        
        # Load monitor locations
        logging.info(f"Loading monitor locations from {location_csv}")
        try:
            loc_df = pd.read_csv(location_csv)
            
            # Check if location data has an ID column that matches our device IDs
            id_cols = ['GICDeviceID', 'DeviceID', 'MonitorID', 'ID']
            loc_id_col = None
            for col in id_cols:
                if col in loc_df.columns:
                    loc_id_col = col
                    break
                    
            if loc_id_col is None:
                logging.error("No device ID column found in the location CSV")
                raise ValueError("Location CSV must contain a device ID column")
            
            # Make sure we don't have duplicate IDs in the location data    
            if loc_df[loc_id_col].duplicated().any():
                logging.warning(f"Duplicate device IDs found in location CSV: {loc_df[loc_id_col][loc_df[loc_id_col].duplicated()].tolist()}")
                logging.warning("Keeping only the first occurrence of each device ID")
                loc_df = loc_df.drop_duplicates(subset=[loc_id_col])
            
            # Check if any device IDs from our data are missing in the location data
            device_ids_set = set(device_series.keys())
            location_ids_set = set(loc_df[loc_id_col])
            missing_ids = device_ids_set - location_ids_set
            if missing_ids:
                logging.warning(f"{len(missing_ids)} device IDs are missing location data")
                logging.debug(f"Missing IDs: {missing_ids}")
                
            loc_df = loc_df.set_index(loc_id_col)
            
            # Ensure required columns exist
            if not {'Longitude', 'Latitude'}.issubset(loc_df.columns):
                logging.error("Location CSV missing coordinate columns")
                raise ValueError("The location CSV must contain 'Longitude' and 'Latitude' columns")

            # Check for invalid coordinates
            if loc_df['Longitude'].isna().any() or loc_df['Latitude'].isna().any():
                logging.warning("Some location entries have missing coordinates")
                loc_df = loc_df.dropna(subset=['Longitude', 'Latitude'])
            
            # Basic validation of coordinates
            if (loc_df['Longitude'] < -180).any() or (loc_df['Longitude'] > 180).any() or \
               (loc_df['Latitude'] < -90).any() or (loc_df['Latitude'] > 90).any():
                logging.warning("Some coordinates appear to be invalid - check your location data")
            
            # Create geometry column
            gdf_nodes = gpd.GeoDataFrame(
                loc_df,
                geometry=gpd.points_from_xy(loc_df['Longitude'], loc_df['Latitude']),
                crs='EPSG:4326'
            )
            
            # Filter top correlated pairs
            thresholds = [0.9, 0.8, 0.7]  # Try different thresholds
            
            for threshold in thresholds:
                strong_links = corr_df[corr_df['Correlation'] >= threshold]
                
                if strong_links.empty:
                    logging.warning(f"No links found with correlation ≥ {threshold}")
                    continue
                else:
                    logging.info(f"Found {len(strong_links)} links with correlation ≥ {threshold}")
                    break
            
            if strong_links.empty:
                logging.warning("No strong correlation links found at any threshold")
            else:
                # Build GeoDataFrame for links
                lines = []
                for _, row in strong_links.iterrows():
                    id1, id2 = row['Device1'], row['Device2']
                    if id1 in gdf_nodes.index and id2 in gdf_nodes.index:
                        try:
                            pt1 = gdf_nodes.loc[id1].geometry
                            pt2 = gdf_nodes.loc[id2].geometry
                            
                            # Validate points
                            if not isinstance(pt1, Point) or not isinstance(pt2, Point):
                                logging.warning(f"Invalid geometry for link {id1}-{id2}")
                                continue
                                
                            line = LineString([pt1, pt2])
                            lines.append({'Device1': id1, 'Device2': id2, 'Correlation': row['Correlation'], 'geometry': line})
                        except Exception as e:
                            logging.debug(f"Error creating line for {id1}-{id2}: {str(e)}")
                            continue

                if lines:
                    gdf_lines = gpd.GeoDataFrame(lines, crs='EPSG:4326')
                    
                    try:
                        usa = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                        usa = usa[usa['continent'] == 'North America']
                        bounds = usa.total_bounds
                    except Exception:
                        bounds = [-125, 24, -66, 50]  # [xmin, ymin, xmax, ymax]
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    try:
                        usa.plot(ax=ax, color='lightgray', edgecolor='white')
                    except Exception:
                        ax.set_facecolor('whitesmoke')
                    
                    # Plot points and lines
                    gdf_nodes.plot(ax=ax, color='blue', markersize=30, alpha=0.7, label='Monitors')
                    gdf_lines.plot(ax=ax, linewidth=1.5, color='red', alpha=0.6, label=f'Links (corr ≥ {threshold})')
                    
                    plt.title(f"High-Correlation GIC Monitor Links (≥ {threshold})")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_plot, dpi=300)
                    logging.info(f"Saved correlation map to {output_plot}")
                    
                    # Also save a version with device labels
                    fig, ax = plt.subplots(figsize=(14, 12))
                    try:
                        usa.plot(ax=ax, color='lightgray', edgecolor='white')
                    except Exception:
                        ax.set_facecolor('whitesmoke')
                        
                    gdf_nodes.plot(ax=ax, color='blue', markersize=30, alpha=0.7)
                    gdf_lines.plot(ax=ax, linewidth=1.5, color='red', alpha=0.6)
                    
                    # Add labels for devices with connections
                    connected_devices = set()
                    for _, row in gdf_lines.iterrows():
                        connected_devices.add(row['Device1'])
                        connected_devices.add(row['Device2'])
                    
                    for idx in connected_devices:
                        if idx in gdf_nodes.index:
                            point = gdf_nodes.loc[idx]
                            ax.annotate(str(idx), 
                                        (point.geometry.x, point.geometry.y),
                                        xytext=(3, 3),
                                        textcoords="offset points",
                                        fontsize=8,
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                    
                    plt.title(f"High-Correlation GIC Monitor Links with Device Labels (≥ {threshold})")
                    plt.tight_layout()
                    plt.savefig(output_dir / 'correlation_map_labeled.png', dpi=300)
                    logging.info(f"Saved labeled correlation map to {output_dir / 'correlation_map_labeled.png'}")
                    
                else:
                    logging.warning("No valid links could be created for the map")
                    
        except Exception as e:
            logging.error(f"Error loading location data: {str(e)}")
            
    except ImportError:
        logging.warning("geopandas not installed - skipping map visualization")
    except Exception as e:
        logging.error(f"Error creating map visualization: {str(e)}")

except Exception as e:
    logging.error(f"Error creating visualizations: {str(e)}")

logging.info("Processing complete")