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
   