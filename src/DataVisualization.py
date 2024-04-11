import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class DataVisualization: 
    def __init__(self, path): 
        pass

    
    def boxplot(self, data,columns): 
        """
        gives the boxplots of many variables

        Args: 
            data: the data we're working with
            columns:  list of the name of columns we want to plot
        
        Returns: 
            None: generates boxplot 
        """
        df = pd.DataFrame(data, columns=columns)

        # Plot boxplots for each column
        plt.figure(figsize=(7,5))
        df.boxplot(grid=True)
        if columns == ['MinTemp', 'MaxTemp','Temp9am', 'Temp3pm']:
            plt.title('Boxplot of temperature Variables')
        elif columns == ['WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']:
            plt.title('Boxplot of WindSpeed Variables')
        elif columns == ['Humidity9am', 'Humidity3pm']: 
            plt.title('Boxplot of Humidity Variables')
        elif columns== ['Pressure9am', 'Pressure3pm']:  
            plt.title('Boxplot of Pressure Variables')
        else: 
            plt.title('Boxplot of Cloud Variables')
        plt.xlabel('Variable')
        plt.ylabel('Value')
        plt.xticks(rotation=45) 

    def relation_with_rainfall(self, column1, column2, data_viz):
        """
        gives 2 lineplot showing the relationship between 2 certain columns and the column Rainfall

        Args: 
            column1:  name of the first column to plot
            column2:  name of the second column to plot
            data_viz: the data we're using for visualization
        
        Returns: 
            None: generates 2 different lineplots next to each other 
        """


        fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')

        titles = ["Relationship between " + column1 + "and Rainfall" + "Relationship between " + column2 +"and Rainfall"]
        x_labels = [column1, column2]

        for ax, title, x_label in zip(axes, titles, x_labels):
            ax.set_title(title, fontsize=21, fontweight='bold', fontfamily='monospace')
            sns.lineplot(data=data_viz, x=x_label, y='Rainfall', ax=ax, color='goldenrod' if x_label == column1 else 'salmon')
            ax.set_ylabel('Rainfall', rotation=0, fontsize=14, fontfamily='monospace')
            ax.set_yticklabels('')
            ax.tick_params(axis='y', length=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

        plt.tight_layout()
        plt.show()

    
    def scatter_plot(self, column1, column2, data_viz):
        """
        gives scatterplot showing the relationship between 2 certain columns and the column Rainfall

        Args: 
            column1:  name of the first column to plot
            column2:  name of the second column to plot
            data_viz: the data we're using for visualization
        
        Returns: 
            None: generates an animated scatterplot  
        """
    
        fig = px.scatter(data_viz, x=column1, y=column2, color="Rainfall", title="Relationship between " + column1 + "," + column2 + "and Rainfall")
        fig.update_layout(width=800, height=600)
        fig.show()

    
