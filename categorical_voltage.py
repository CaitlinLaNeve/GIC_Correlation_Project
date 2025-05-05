import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the correlation data
df_corr = pd.read_csv("C:/664/gic/monitor_correlations.csv")  # Replace with your actual file path

# Convert 'volt_diff(kV)' to numeric, coercing errors to NaN
df_corr['volt_diff(kV)'] = pd.to_numeric(df_corr['volt_diff(kV)'], errors='coerce')

# Drop rows where 'volt_diff(kV)' is NaN after conversion
df_corr.dropna(subset=['volt_diff(kV)'], inplace=True)

# Define voltage difference thresholds
small_diff_threshold = 50
n_highlight = 20

# Create voltage difference categories
df_corr['voltage_similarity'] = 'Large Difference'
df_corr.loc[abs(df_corr['volt_diff(kV)']) == 0, 'voltage_similarity'] = 'Same Voltage'
df_corr.loc[(abs(df_corr['volt_diff(kV)']) > 0) & (abs(df_corr['volt_diff(kV)']) <= small_diff_threshold), 'voltage_similarity'] = 'Small Difference'

# Sort by correlation coefficient
df_sorted = df_corr.sort_values(by='cc')
top_n = df_sorted.tail(n_highlight)
bottom_n = df_sorted.head(n_highlight)

# Create the box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df_corr['voltage_similarity'], y=df_corr['cc'], order=['Same Voltage', 'Small Difference', 'Large Difference'])
plt.xlabel("Voltage Similarity Category (based on |volt_diff(kV)|)")
plt.ylabel("Correlation Coefficient (cc)")
plt.title("Correlation Coefficient vs. Voltage Similarity")
plt.grid(True)
plt.show()

# Print the distribution of top and bottom CC pairs across voltage categories
print("\nDistribution of Top 20 CC Pairs by Voltage Similarity:")
print(top_n['voltage_similarity'].value_counts())

print("\nDistribution of Bottom 20 CC Pairs by Voltage Similarity:")
print(bottom_n['voltage_similarity'].value_counts())