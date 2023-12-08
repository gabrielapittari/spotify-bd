import streamlit as st

st.title("This is the data visualization code")
st.text("We completed data exploration and visualization before running the ML model to choose which features to use")

code = '''import pandas as pd

df = pd.read_csv('spotify_data.csv')

dtype_mapping = {
    "artist_name": "string",
    "track_name": "string",
    "track_id": "string",
    "popularity": "int",
    "year": "int",
    "genre": "string",
    "danceability": "float",
    "energy": "float",
    "key": "int",
    "loudness": "float",
    "mode": "int",
    "speechiness": "float",
    "acousticness": "float",
    "instrumentalness": "float",
    "liveness": "float",
    "valence": "float",
    "tempo": "float",
    "duration_ms": "float",
    "time_signature": "int"
}

df = df.astype(dtype_mapping)

print(df.head(5))

column_names = df.columns.tolist()
print(column_names)

df.info()

# Cast the columns
df = df.astype(dtype_mapping)

# Print the DataFrame's information to check the data types
df.info()


# Get unique values from the 'genre' column
genreRows = df['genre'].unique()

# Print the unique genre values
print(genreRows)


# Group by 'genre' and count the occurrences
genre_counts = df.groupby('genre').size()

# Display the result
print(genre_counts)


total_rows = df.shape[0]
print("Total rows:", total_rows)


# Iterate over each column and print rows where the column value is null
for column in df.columns:
    print('Analyzing', column)
    null_rows = df[df[column].isna()]
    print(null_rows)


print(df.isna().sum())

# Group by 'track_name' and 'artist_name', and count occurrences
track_artist_counts = df.groupby(['track_name', 'artist_name']).size().reset_index(name='count')

# Filter for counts greater than one
repeated_tracks_artists = track_artist_counts[track_artist_counts['count'] > 1]

# Display the result
print(repeated_tracks_artists)

# Drop duplicates based on 'artist_name' and 'track_name'
df_unique = df.drop_duplicates(subset=['artist_name', 'track_name'])

# Display the first few rows of the resulting DataFrame
print(df_unique.head())

def iqr_outlier_treatment(dataframe, columns, factor=1.5):
    """
    Detects and treats outliers using IQR for multiple variables in a Pandas DataFrame.

    :param dataframe: The input Pandas DataFrame
    :param columns: A list of columns to apply IQR outlier treatment
    :param factor: The IQR factor to use for detecting outliers (default is 1.5)
    :return: The processed DataFrame with outliers treated
    """
    for column in columns:
        # Calculate Q1 and Q3
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1

        # Define the upper and lower bounds for outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Filter out the outliers
        dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

    return dataframe

# Filter out rows where 'tempo' equals 0
df_filtered = df[df['tempo'] > 0]

# Calculate the quartiles and IQR
bounds = df_filtered['tempo'].quantile([0.25, 0.75])
IQR = bounds[0.75] - bounds[0.25]
lower_bound = bounds[0.25] - 1.5 * IQR
upper_bound = bounds[0.75] + 1.5 * IQR

# Remove outliers
df_no_outliers = df_filtered[(df_filtered['tempo'] >= lower_bound) & (df_filtered['tempo'] <= upper_bound)]

# Calculate percentile rank
df_no_outliers['percentile_rank'] = df_no_outliers['tempo'].rank(pct=True)

# Function to categorize based on quartiles
def categorize_tempo(x):
    if x < 0.25:
        return 'G1'
    elif x < 0.5:
        return 'G2'
    elif x < 0.75:
        return 'G3'
    else:
        return 'G4'

# Assign tempo group
df_no_outliers['tempo_group'] = df_no_outliers['percentile_rank'].apply(categorize_tempo)

# Display the results
print(df_no_outliers.head())

integerColumns = ["popularity", "year", "key", "mode", "duration_ms", "time_signature"]
floatColumns = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

print("Integer Columns:", integerColumns)
print("Float Columns:", floatColumns)

numeric_columns = integerColumns + floatColumns

# Displaying the combined list
print("Numeric Columns:", numeric_columns)

df_outlier_treatment = iqr_outlier_treatment(df, numeric_columns, factor=1.5)

# Joining the original DataFrame with the outlier-treated DataFrame
# Assuming df and df_outlier_treatment are Pandas DataFrames
outliers_df = df.join(df_outlier_treatment[numeric_columns], rsuffix="_treated")


# import matplotlib.pyplot as plt

# for column in numeric_columns:
   treated = column + "_treated"
   print("Outliers for ", column)
   plt.figure()
  outliers_df.boxplot(column= [column, treated], grid=False, figsize=(6,3))
   plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Define your string columns
stringColumns = ['artist_name', 'track_name', 'track_id', 'genre']

# Drop non-numeric columns
numeric_df = df.drop(columns=stringColumns)

# Compute the correlation matrix
corr_mat_df = numeric_df.corr(method='pearson')

# Visualize the correlation matrix
plt.figure(figsize=(16, 5))
sns.heatmap(corr_mat_df, xticklabels=corr_mat_df.columns, yticklabels=corr_mat_df.columns, cmap="Greens", annot=True)
plt.show()
'''
st.code(code, language='python')