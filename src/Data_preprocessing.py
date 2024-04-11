import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class Data_preprocessing: 
   def __init__(self, path): 
        """
        Constructor for the Data_preprocessing class
        """
        self.path = path
        self.data = None
   
   def load_data_from_csv(self):
        """
        Loads the data from a csv file
        Returns:
            data: dataframe loaded
        """
        self.data = pd.read_csv(self.path)
        return self.data

   def fill_missing_with_mode(data, columns):  
      """
      Replaces missing values in specified columns with the mode of each column.
      Returns:
      - data: The modified dataframe with missing values replaced by each column's mode.
      """
      for col in columns:
         mode_val = data[col].mode()[0]  
         data[col].fillna(mode_val, inplace=True) 
      return data
   
   def fill_missing_with_mean(data, columns):
      """
      Replaces missing values in specified columns with the mean of each column.
      
      Args:
      - data: The dataframe containing the data.
      - columns: A list of column names where missing values should be handled.
      
      Returns:
      - data: The modified dataframe with missing values replaced by each column's mean.
      """
      for col in columns:
         mean_val = data[col].mean()
         data[col].fillna(mean_val, inplace=True) 
      return data
   
   def fill_missing_with_median(data, columns):
      """
      Replaces missing values in specified columns with the median of each column.
      Returns:
      - data: The modified dataframe with missing values replaced by each column's median.
      """
      for col in columns:
         median_val = data[col].median()  
         data[col].fillna(median_val, inplace=True) 
      return data
   
   def remove_outliers(data):
      """
      Removes outliers from the dataset using the Interquartile Range (IQR) method.
      
      Args:
      - data: The dataframe containing the data.
      
      Returns:
      - data_filtered: The dataframe with outliers removed.
      """
      Q1 = data.quantile(0.25)  
      Q3 = data.quantile(0.75) 
      IQR = Q3 - Q1 
      
      # Filter out rows with outliers
      data_filtered = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
      
      return data_filtered

   