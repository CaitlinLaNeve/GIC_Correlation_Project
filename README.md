# GIC_Correlation_Project
This repository is for my final project for GMU GGS 664. This will contain sample data, python scripts, and plots produced during this study. <br></br>
Folder 'figures' contains the outputs from various scripts such as correlation plots and maps. <br></br>
Folder 'python scripts' contains 10 scripts:<br></br>
  ----addlocation.py adds location columns from monitorlocations.csv to each monitor measurement file<br></br>
  ----categorical_voltage.py <br></br>
  ----cc_analysis_plots.py contains both Pearson and Lagged correlation plot calculation<br></br>
  ----cc_ns_monitors.py adds orientation, quantity of north-south running transmission lines to the correlation coefficient file<br></br>
  ----cc_vs_all.py extract all information from the ArcPro exports (Spatial Join for all buffer distances), add to correlation file<br></br>
  ----correlation_pipeline.py calculates the correlation coefficients for all monitor pairs<br></br>
  ----extract_monitor.py extracts the target file from all 397 zipped files downloaded from NERC, adds monitor location data and GICDeviceID<br></br>
  ----gic_map_script.py script for creating map of monitors and links for pairs with over 0.9 correlation<br></br>
  ----ratio.py calculates the ratio of lines to compare between monitors<br></br>
2024E04_10708.csv is a sample data of a downloaded monitor from NERC<br></br>
monitorlocation.csv is the file containing lat/long, GICDeviceID for all monitors<br></br>
All other csv files in this folder are various outputs from the scripts and were used the most in this project. Lagged_correlation_pairs also includes Pearson correlation.<br></br>
