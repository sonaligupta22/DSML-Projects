#!/usr/bin/env python
# coding: utf-8

# In[114]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# ### Problem Statement:
# 
# Delhivery, India's leading logistics player, seeks to optimize its operations by leveraging data-driven insights. The company aims to clean, manipulate, and analyze its data to extract valuable features, detect outliers, and perform hypothesis testing. The primary objective is to improve operational efficiency and enhance customer satisfaction through informed decision-making.
# 
# ### Exploratory Data Analysis (EDA):
# 
# The EDA process involves handling missing values, profiling the dataset, creating new features, grouping and aggregating data, detecting outliers, and performing hypothesis testing. Visualization and interpretation of the data will provide actionable insights for optimizing logistics operations and driving business growth.
# 
# ### Conclusion:
# 
# Delhivery's data analysis aims to extract actionable insights to optimize logistics operations and improve customer satisfaction. Through effective data cleaning, manipulation, and analysis, the company can make informed decisions and drive business growth.

# In[115]:


df = pd.read_csv("delhivery_data.csv")
df.head(10)


# In[116]:


sample_df = df.sample(n=5, random_state=1) 
sample_df


# In[117]:


df.rename(columns={'factor': 'factor_ratio', 'segment_factor': 'segment_factor_ratio'}, inplace=True)


# ### Converting time columns into pandas datetime.

# In[118]:


df['od_start_time'] = pd.to_datetime(df['od_start_time'])
df['od_end_time'] = pd.to_datetime(df['od_end_time'])
df['cutoff_timestamp'] = pd.to_datetime(df['cutoff_timestamp'])
df['trip_creation_time'] = pd.to_datetime(df['cutoff_timestamp'])


# ### Source_name and destination_name have an unequal number of data counts, let do more analysis below.
# **Filling Missing Values using Mode**
# 
# Given the discrepancy in the number of non-null values between the source_name and destination_name columns, we have 144574 non-null values for source_name and 144606 non-null values for destination_name out of 144867 total entries. With such a small number of missing values compared to the total dataset size, imputation may be a reasonable approach to handle the missing values in this case. Let's proceed with imputing the missing values using the mode (most frequent value) of each respective colum-

# In[119]:


df.shape


# In[120]:


print(len(df['destination_name'].unique()))
print(len(df['source_name'].unique()))


# In[121]:


source_name_mode = df['source_name'].mode()[0]
df['source_name'].fillna(source_name_mode, inplace = True)


# In[122]:


destination_name_mode = df['destination_name'].mode()[0]
df['destination_name'].fillna(destination_name_mode, inplace = True)


# In[123]:


cat_col = [col for col in df.columns if df[col].dtype == 'object']
print("CATEGORICAL COLUMNS= ", cat_col)
print("TOTAL = ",len(cat_col), end="\n\n")
num_col = [col for col in df.columns if df[col].dtype != 'object']
print("NUMERICAl COLUMNS= ", num_col)
print("TOTAL = ",len(num_col))


# ### After filling the null values with the mode, our dataset now contains no null values, allowing us to proceed with further analysis.

# ## Grouping by segment
# Segmenting the data set based on  trip_uuid, source_center & destination_center

# In[124]:


df['trip_segment'] = df['trip_uuid'].astype(str) + '_' + df['source_center'] + '_' + df['destination_center']

# Calculate cumulative sums for the segments
df['segment_actual_time_sum'] = df.groupby('trip_segment')['segment_actual_time'].cumsum()
df['segment_osrm_distance_sum'] = df.groupby('trip_segment')['segment_osrm_distance'].cumsum()
df['segment_osrm_time_sum'] = df.groupby('trip_segment')['segment_osrm_time'].cumsum()
df


# ## Aggregating at segment level
# 
# Aggregating the dataset based on the segment created above "trip_segment". Then we will create a dictionary based on the essential columns in the dataset.

# In[125]:


# Create the aggregation dictionary
create_segment_dict = {
    'trip_uuid' : 'first',
    'od_start_time': 'first',  # Keep the first od_start_time
    'od_end_time': 'last',  # Keep the last od_end_time
    'osrm_time': 'sum',  # Sum of osrm_time
    'osrm_distance': 'sum',  # Sum of osrm_distance
    'segment_actual_time_sum': 'last',  # Sum of segment_actual_time
    'segment_osrm_time_sum': 'last',  # Sum of segment_osrm_time
    'segment_osrm_distance_sum':'last',
    'destination_name' : 'first',
    'source_name' : 'first',
    'trip_creation_time' : 'first',
    "route_type" : 'first',
    'actual_time' : sum,
      'segment_actual_time' :sum,
    'segment_osrm_distance' : sum,
    'segment_osrm_time' : sum,
}

# Group by the segment key and aggregate using the dictionary
aggregated_df = df.groupby('trip_segment').agg(create_segment_dict).reset_index()

# # Sort the resulting DataFrame
# aggregated_df = aggregated_df.sort_values(by=['trip_segment', 'od_end_time'], ascending=[False, True]).reset_index(drop=True)

# # Display the resulting DataFrameaggregated_df
aggregated_df


# ## Extracting features

# In[126]:


#Calculating the od start & end time difference and rounding it off for user readability
aggregated_df['od_time_diff_hour'] = np.round((aggregated_df['od_end_time'] - aggregated_df['od_start_time']).dt.total_seconds()/3600,3)
aggregated_df['od_time_diff_hour']

#Dropping the od time columns
aggregated_df.drop(['od_end_time', 'od_start_time'], axis=1) 
aggregated_df.head(2)


# In[127]:


#Splitting the destination name and extracting the place_code and city
aggregated_df['destination_place_code'] = aggregated_df['destination_name'].str.split(' ', expand=True)[0]
aggregated_df['destination_city'] = aggregated_df['destination_name'].str.split(' ', expand=True)[1]

#Splitting the source name and extracting the place_code and city
aggregated_df['source_place_code'] = aggregated_df['source_name'].str.split(' ', expand=True)[0]
aggregated_df['source_city'] = aggregated_df['source_name'].str.split(' ', expand=True)[1]

#Removing the () from the city name
aggregated_df['destination_city'] = aggregated_df['destination_city'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)
aggregated_df['source_city'] = aggregated_df['source_city'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)
aggregated_df.head()

#Dropping the destination & source name
aggregated_df.drop(['source_name', 'destination_name'], axis=1, inplace=True) 

aggregated_df.head()


# In[128]:


aggregated_df['trip_creation_date'] = aggregated_df['trip_creation_time'].dt.date
aggregated_df['trip_creation_time'] = aggregated_df['trip_creation_time'].dt.time
aggregated_df.head()
# print("Total columns:", len(aggregated_df.index))


# In[143]:


create_trip_dict = {
    'osrm_time' : sum,
    'osrm_distance' : sum,
    'segment_actual_time_sum' : sum,
    'od_time_diff_hour' : sum,
    'od_start_time' : 'first',
    'od_end_time' : 'last',
    'route_type' : 'first',
    'actual_time' : sum,
    'segment_actual_time' :sum,
    'segment_osrm_distance' : sum,
    'segment_osrm_time' : sum,
}
trip_summary_df = aggregated_df.groupby('trip_uuid').agg(create_trip_dict).reset_index()
trip_summary_df.rename(columns={'od_start_time': 'trip_start_time', 'od_end_time': 'trip_end_time'}, inplace=True)
trip_summary_df


# In[130]:


data = trip_summary_df[['osrm_time', 'osrm_distance']]
sns.boxplot(data=data)


# In[131]:


data = trip_summary_df[['segment_actual_time_sum']]
sns.boxplot(data=data)


# In[132]:


data = trip_summary_df[['od_time_diff_hour']]
sns.boxplot(data=data)


# ### Identifying Outliers using IQR - Before Log Transformation

# In[133]:


#Identifying Outliers using IQR - Before Log Transformation
for i in ['osrm_time', 'osrm_distance', 'segment_actual_time_sum', 'od_time_diff_hour']:
    Q1 = np.percentile(trip_summary_df[i], 25)
    Q3 = np.percentile(trip_summary_df[i], 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = [x for x in trip_summary_df[i] if x < lower_bound or x > upper_bound]

    print("Lower bound for outliers(", i, ") :", lower_bound)
    print("Upper bound for outliers(", i, ") :", upper_bound)
    print("Outliers Length(", i, ") :", len(outliers))


# ### Handling outliers using log transformation

# In[134]:


# Define the columns for which you want to apply log transformation
numeric_columns = ['osrm_time', 'osrm_distance', 'segment_actual_time_sum', 'od_time_diff_hour']

# Function to apply log transformation
def log_transform(column):
    # Shift the data if there are non-positive values
    shift = 0
    if (column <= 0).any():
        shift = np.abs(np.min(column)) + 1
        column = column + shift
    # Apply log transformation
    column = np.log(column)
    return column, shift

# Apply log transformation for each numeric column
shift_values = {}
for column in numeric_columns:
    trip_summary_df[column], shift = log_transform(trip_summary_df[column])
    shift_values[column] = shift

# Verify the results
for column in numeric_columns:
    print(f"Shift value for {column}: {shift_values[column]}")
    print(f"Transformed {column} statistics:\n{trip_summary_df[column].describe()}")

# Display the resulting DataFrame
print(trip_summary_df.head())


# ### Identifying Outliers using IQR - After Log Transformation

# In[135]:


#Identifying Outliers using IQR - After Log Transformation
for i in ['osrm_time', 'osrm_distance', 'segment_actual_time_sum', 'od_time_diff_hour']:
    Q1 = np.percentile(trip_summary_df[i], 25)
    Q3 = np.percentile(trip_summary_df[i], 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = [x for x in trip_summary_df[i] if x < lower_bound or x > upper_bound]

    print("Lower bound for outliers(", i, ") :", lower_bound)
    print("Upper bound for outliers(", i, ") :", upper_bound)
    print("Outliers Length(", i, ") :", len(outliers))


# ### Apply one-hot encoding on route_type

# In[136]:


import pandas as pd

# Display the unique values in the 'route_type' column
print("Unique values in 'route_type':", trip_summary_df['route_type'].unique())

# Apply one-hot encoding
trip_summary_df = pd.get_dummies(trip_summary_df, columns=['route_type'], prefix='route')
trip_summary_df.head(3)


# ### Column Normalization

# In[137]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Select the columns to be scaled
numerical_features = ['osrm_time', 'osrm_distance', 'segment_actual_time_sum', 'od_time_diff_hour']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numerical features
trip_summary_df[numerical_features] = scaler.fit_transform(trip_summary_df[numerical_features])

trip_summary_df.head(5)


# ### Correlation Heatmap

# In[142]:


plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# ### Strong Positive Correlations:
# 
# The variables start_scan_to_end_scan, actual_distance_to_destination, actual_time, osrm_time, osrm_distance, segment_actual_time_sum, segment_osrm_distance_sum, and segment_osrm_time_sum all have strong positive correlations with each other. This suggests that as one of these variables increases, the others tend to increase as well.
# 
# 
# ### Weak Positive Correlations:
# 
# The variables is_cutoff and cutoff_factor have weak positive correlations with most other variables in the matrix.

# # Recommendations
# 
# 1. **Focus on Busiest Corridor**: Allocate more resources and marketing efforts to the busiest corridor identified in the data to capitalize on the high demand and increase profitability.
# 
# 2. **Optimize Routes**: Analyze the average distance and time taken between corridors to identify opportunities for route optimization. This can help reduce transportation costs and improve efficiency.
# 
# 3. **Improve Delivery Time**: Identify factors contributing to longer delivery times, such as traffic congestion or inefficient routes, and implement strategies to reduce them. This could include using real-time traffic data or adjusting delivery schedules.
# 
# 5. **Expand Service Coverage**: Explore opportunities to expand service coverage to areas with high demand or underserved regions identified in the data. This can help capture additional market share and increase revenue.
# 
# 6. **Streamline Operations**: Streamline internal processes and workflows to improve overall efficiency and reduce operating costs. This could involve investing in technology solutions or implementing training programs for staff.This will also lead to more customer satisfaction
# 
# 
# By focusing on these actionable items, the business can enhance its operations, improve customer satisfaction, and drive growth in the identified corridors.
