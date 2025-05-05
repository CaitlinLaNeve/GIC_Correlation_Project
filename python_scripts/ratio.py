import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the correlation data
df_corr = pd.read_csv("C:/664/gic/monitor_correlations.csv")  

buffers = [500, 1000, 1500]
n_highlight = 20

for buffer in buffers:
    lines_site1_col = f"1_lines_{buffer}"
    lines_site2_col = f"2_lines_{buffer}"
    ratio_col = f"lines_ratio_{buffer}m"

    # Calculate the ratio of lines
    df_corr[ratio_col] = df_corr.apply(
        lambda row: (min(row[lines_site1_col], row[lines_site2_col]) + 1) / (max(row[lines_site1_col], row[lines_site2_col]) + 1),
        axis=1
    )

    # Sort by correlation coefficient
    df_sorted = df_corr.sort_values(by='cc')
    top_n = df_sorted.tail(n_highlight)
    bottom_n = df_sorted.head(n_highlight)

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_corr[ratio_col], y=df_corr['cc'], color='blue', alpha=0.6)
    sns.scatterplot(x=top_n[ratio_col], y=top_n['cc'], color='green', s=60, label=f'Top {n_highlight} CC')
    sns.scatterplot(x=bottom_n[ratio_col], y=bottom_n['cc'], color='red', s=60, label=f'Bottom {n_highlight} CC')

    plt.xlabel(f"Ratio of Number of Lines ({buffer}m Buffer) (min+1)/(max+1)")
    plt.ylabel("Correlation Coefficient (cc)")
    plt.title(f"Correlation Coefficient vs. Ratio of Lines ({buffer}m Buffer)")
    plt.legend()
    plt.grid(True)
    plt.show()