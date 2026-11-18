
import pandas as pd

# Load the network traffic dataset
df = pd.read_csv("data/sample_network_traffic.csv")

# Display the complete dataset
print("\n--- NETWORK TRAFFIC DATA ---")
print(df)

# Display the first five rows
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# Display the dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Display column names and data types
print("\n--- DATASET INFORMATION ---")
df.info()
