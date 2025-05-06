# GIC_Correlation_Project
This repository is for my final project for GMU GGS 664. This will contain sample data, python scripts, and plots produced during this study.
Folder 'figures' contains the outputs from various scripts such as correlation plots and maps.
Folder 'python scripts' contains 10 scripts:
  addlocation.py adds location columns from monitorlocations.csv to each monitor measurement file
  categorical_voltage.py 
  cc_analysis_plots.py contains both Pearson and Lagged correlation plot calculation
  cc_ns_monitors.py adds orientation, quantity of north-south running transmission lines to the correlation coefficient file
  cc_vs_all.py extract all information from the ArcPro exports (Spatial Join for all buffer distances), add to correlation file
  correlation_pipeline.py calculates the correlation coefficients for all monitor pairs
  extract_monitor.py extracts the target file from all 397 zipped files downloaded from NERC, adds monitor location data and GICDeviceID
  gic_map_script.py script for creating map of monitors and links for pairs with over 0.9 correlation
  ratio.py calculates the ratio of lines to compare between monitors
2024E04_10708.csv is a sample data of a downloaded monitor from NERC
monitorlocation.csv is the file containing lat/long, GICDeviceID for all monitors
All other csv files in this folder are various outputs from the scripts and were used the most in this project. Lagged_correlation_pairs also includes Pearson correlation.
