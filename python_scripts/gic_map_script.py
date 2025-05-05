import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Point
import logging
from pathlib import Path
import sys

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gic_map.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  
    ]
)

input_dir = Path('C:/664/gic/results')
location_csv = Path('C:/664/gic/monitorlocation.csv')
output_dir = Path('C:/664/gic/results/maps')
output_dir.mkdir(exist_ok=True) 

# Output files
correlation_csv = input_dir / 'C:/664/gic/results/lagged_correlation_pairs.csv'
output_map = output_dir / 'high_correlation_links.png'
output_labeled_map = output_dir / 'correlation_map_labeled.png'

def load_data():
    """Load correlation and location data"""
    logging.info(f"Loading correlation data from {correlation_csv}")
    try:
        corr_df = pd.read_csv(correlation_csv)
        logging.info(f"Loaded {len(corr_df)} correlation pairs")
    except Exception as e:
        logging.error(f"Failed to load correlation data: {e}")
        return None, None
    
    logging.info(f"Loading monitor locations from {location_csv}")
    try:
        loc_df = pd.read_csv(location_csv)
        
        # Check for ID column
        id_cols = ['GICDeviceID', 'DeviceID', 'MonitorID', 'ID']
        loc_id_col = None
        for col in id_cols:
            if col in loc_df.columns:
                loc_id_col = col
                break
                
        if loc_id_col is None:
            logging.error("No device ID column found in the location CSV")
            return corr_df, None
        
        # Remove duplicates if any
        if loc_df[loc_id_col].duplicated().any():
            logging.warning(f"Duplicate device IDs found in location CSV")
            loc_df = loc_df.drop_duplicates(subset=[loc_id_col])
        
        # Check required columns
        if not {'Longitude', 'Latitude'}.issubset(loc_df.columns):
            logging.error("Location CSV missing coordinate columns")
            return corr_df, None
            
        # Set index to device ID
        loc_df = loc_df.set_index(loc_id_col)
        
        # Clean data
        loc_df = loc_df.dropna(subset=['Longitude', 'Latitude'])
        logging.info(f"Loaded {len(loc_df)} monitor locations")
        
        return corr_df, loc_df
    except Exception as e:
        logging.error(f"Failed to load location data: {e}")
        return corr_df, None

def create_map_visualization(corr_df, loc_df, threshold=0.8):
    """Create map visualization of high-correlation links"""
    if corr_df is None or loc_df is None:
        logging.error("Cannot create map - missing data")
        return False
    
    # Filter by threshold
    strong_links = corr_df[corr_df['PearsonCorrelation'] >= threshold]
    logging.info(f"Found {len(strong_links)} links with correlation >= {threshold}")
    
    if strong_links.empty:
        logging.warning(f"No links found with correlation >= {threshold}")
        return False
    
    # Create GeoDataFrame for nodes
    try:
        gdf_nodes = gpd.GeoDataFrame(
            loc_df,
            geometry=gpd.points_from_xy(loc_df['Longitude'], loc_df['Latitude']),
            crs='EPSG:4326'
        )
        
        # Build lines for links
        lines = []
        for _, row in strong_links.iterrows():
            id1, id2 = row['Device1'], row['Device2']
            if id1 in gdf_nodes.index and id2 in gdf_nodes.index:
                try:
                    pt1 = gdf_nodes.loc[id1].geometry
                    pt2 = gdf_nodes.loc[id2].geometry
                    
                    if not isinstance(pt1, Point) or not isinstance(pt2, Point):
                        continue
                        
                    line = LineString([pt1, pt2])
                    lines.append({
                        'Device1': id1, 
                        'Device2': id2, 
                        'Correlation': row['PearsonCorrelation'], 
                        'geometry': line
                    })
                except Exception as e:
                    logging.debug(f"Error creating line for {id1}-{id2}: {str(e)}")
                    continue
        
        if not lines:
            logging.warning("No valid links could be created for the map")
            return False
            
        gdf_lines = gpd.GeoDataFrame(lines, crs='EPSG:4326')
        logging.info(f"Created {len(gdf_lines)} map links")
        
        # Get USA base map if possible
        try:
            usa = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            usa = usa[usa['continent'] == 'North America']
            north_america = True
        except Exception:
            logging.warning("Could not load natural earth data, using basic background")
            north_america = False
        
        # Create standard map
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if north_america:
            usa.plot(ax=ax, color='lightgray', edgecolor='white')
        else:
            ax.set_facecolor('whitesmoke')
        
        # Plot points and lines
        gdf_nodes.plot(ax=ax, color='blue', markersize=30, alpha=0.7, label='Monitors')
        gdf_lines.plot(ax=ax, linewidth=1.5, color='red', alpha=0.6, 
                       label=f'Links (corr >= {threshold})')
        
        plt.title(f"High-Correlation GIC Monitor Links (>= {threshold})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_map, dpi=300)
        logging.info(f"Saved correlation map to {output_map}")
        
        # Create labeled map
        fig, ax = plt.subplots(figsize=(14, 12))
        
        if north_america:
            usa.plot(ax=ax, color='lightgray', edgecolor='white')
        else:
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
        
        plt.title(f"High-Correlation GIC Monitor Links with Device Labels (>= {threshold})")
        plt.tight_layout()
        plt.savefig(output_labeled_map, dpi=300)
        logging.info(f"Saved labeled correlation map to {output_labeled_map}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error creating map visualization: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    logging.info("Starting GIC map visualization")
    
    # Load data
    corr_df, loc_df = load_data()
    
    # Try different correlation thresholds
    for threshold in [0.9, 0.8, 0.7, 0.6]:
        logging.info(f"Attempting to create map with threshold {threshold}")
        if create_map_visualization(corr_df, loc_df, threshold):
            break
    
    logging.info("Map visualization process complete")

if __name__ == "__main__":
    main()