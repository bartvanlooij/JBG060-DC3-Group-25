import os
import pandas as pd

# Create an empty list to store the dataframes
dfs = []

# Loop through all the files in the "data" folder
for file in os.listdir("data"):
    # Check if the file name contains "full_dataset_with_sentiment_"
    if "full_dataset_with_sentiment_" in file:
        if "combined" in file:
            continue
        # Read the CSV file into a dataframe and append it to the list of dataframes
        df = pd.read_csv(os.path.join("data", file), index_col=0)
        dfs.append(df)

# Concatenate all the dataframes in the list
combined_df = pd.concat(dfs)

# Print the combined dataframe
combined_df.to_csv("data/full_dataset_with_sentiment_sum_combined.csv", index=True)

