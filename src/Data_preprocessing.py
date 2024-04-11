import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


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
   
   def encode_categorical_binary(data, columns):
      """
      Encodes categorical variables in the dataframe as binary values (0 and 1).
      
      Args:
      - data: The dataframe containing the data.
      - columns: A list of column names where categorical variables should be encoded.
      
      Returns:
      - data: The modified dataframe with categorical variables encoded as binary values.
      """
      for col in columns:
         data[col].replace({'No': 0, 'Yes': 1}, inplace=True)
      return data
   
   def split_date_columns(data, date_column):
      """
      Splits a date column into separate 'year', 'month', and 'day' columns.
      Removes the original date column.
      
      Args:
      - data: The dataframe containing the data.
      - date_column: The name of the column containing the date to be split.
      
      Returns:
      - data: The modified dataframe with separate date columns.
      """
      data[['year', 'month', 'day']] = data[date_column].str.split('-', expand=True)
      data = data.drop([date_column], axis=1)
      return data
   
   from sklearn.preprocessing import LabelEncoder

   def encode_categorical_label(data, columns):
      """
      Encodes categorical variables in the dataframe using LabelEncoder.
      
      Args:
      - data: The dataframe containing the data.
      - columns: A list of column names to be encoded.
      
      Returns:
      - data: The modified dataframe with categorical variables encoded.
      """
      label_encoder = LabelEncoder()
      for col in columns:
         unique_values = list(set(data[col].unique()))
         label_encoder.fit(unique_values)
         data[col] = label_encoder.transform(data[col])
      return data
   
   def Split_data(self, df, target):
      """
      Split data to train and test 

      Args: 
            df: dataframe of all the features 
            target: name of the target

      Returns:
            X_train, X_test, y_train, y_test: train and test features and train and test labels
        """
      features = df.drop(target, axis =1)
      targets = df[target]
      X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state=42)

      return X_train, X_test, y_train, y_test

   def balance_data(X_train, y_train):
      """
      Balances the classes in the dataset using SMOTE (Synthetic Minority Over-sampling Technique).

      Returns:
      - X_balanced: The feature matrix after balancing the classes.
      - y_balanced: The target vector after balancing the classes.
      """
      smote = SMOTE()
      X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
      return X_balanced, y_balanced



   