import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Define a function to read the correlation data
def read_correlation_file(file_path):
    # Read the CSV data
    data = pd.read_csv(file_path)

    # Check if the columns we need are present
    if 'Device1_NSLines_1500m' in data.columns and 'Device2_NSLines_1500m' in data.columns:
        # Calculate the missing columns (Sum_NSLines and Diff_NSLines)
        data['Sum_NSLines'] = data['Device1_NSLines_1500m'] + data['Device2_NSLines_1500m']
        data['Diff_NSLines'] = data['Device1_NSLines_1500m'] - data['Device2_NSLines_1500m']
    
    return data

# Define a function to compute Pearson's correlation and p-value
def compute_pearson_correlation(x, y):
    return pearsonr(x, y)

# Define a function to plot the scatterplot
def plot_scatter(x, y, x_label, y_label, file_name):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs. {y_label}")
    plt.savefig(file_name)
    plt.close()

# Define a function to analyze each file (500m, 1000m, 1500m)
def analyze_file(file_path, buffer_size):
    # Read the data
    data = read_correlation_file(file_path)
    
    # Extract the relevant columns
    lagged_correlation = data['MaxLaggedCorrelation']
    pearson_correlation = data['PearsonCorrelation']
    sum_nslines = data['Sum_NSLines']
    diff_nslines = data['Diff_NSLines']
    
    # Compute Pearson's r and p-values for each relationship
    r_lagged_sum, p_lagged_sum = compute_pearson_correlation(lagged_correlation, sum_nslines)
    r_lagged_diff, p_lagged_diff = compute_pearson_correlation(lagged_correlation, diff_nslines)
    r_pearson_sum, p_pearson_sum = compute_pearson_correlation(pearson_correlation, sum_nslines)
    r_pearson_diff, p_pearson_diff = compute_pearson_correlation(pearson_correlation, diff_nslines)
    
    # Plot scatterplots for the relationships
    plot_scatter(lagged_correlation, sum_nslines, 'Lagged Correlation', 'Sum_NSLines', f"lagged_sum_{buffer_size}.png")
    plot_scatter(lagged_correlation, diff_nslines, 'Lagged Correlation', 'Diff_NSLines', f"lagged_diff_{buffer_size}.png")
    plot_scatter(pearson_correlation, sum_nslines, 'Pearson Correlation', 'Sum_NSLines', f"pearson_sum_{buffer_size}.png")
    plot_scatter(pearson_correlation, diff_nslines, 'Pearson Correlation', 'Diff_NSLines', f"pearson_diff_{buffer_size}.png")
    
    # Save a summary of the Pearson correlation and p-values
    summary = {
        'Buffer Size': buffer_size,
        'r_lagged_sum': r_lagged_sum, 'p_lagged_sum': p_lagged_sum,
        'r_lagged_diff': r_lagged_diff, 'p_lagged_diff': p_lagged_diff,
        'r_pearson_sum': r_pearson_sum, 'p_pearson_sum': p_pearson_sum,
        'r_pearson_diff': r_pearson_diff, 'p_pearson_diff': p_pearson_diff
    }
    
    return summary

# Main analysis function
def run_analysis():
    # Define file paths for each buffer size
    files = {
        '500m': 'C:/664/gic/correlation_with_line_stats_500m.csv',
        '1000m': 'C:/664/gic/correlation_with_line_stats_1000m.csv',
        '1500m': 'C:/664/gic/correlation_with_line_stats_1500m.csv'
    }
    
    # List to hold all the summary data
    summary_data = []
    
    # Analyze each file
    for buffer_size, file_path in files.items():
        summary = analyze_file(file_path, buffer_size)
        summary_data.append(summary)
    
    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary_data)
    
    # Save the summary to a CSV file
    summary_df.to_csv('summary.csv', index=False)

# Run the analysis
run_analysis()
