"""
Docstring
"""

import urllib.request
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AgrosClass:
    """
    A class that downloads a CSV file from a given URL and saves it in a local directory,
    and then loads it into a pandas DataFrame.
    """
    def __init__(self, url = 'https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv'):
        """
        Initializes the DataDownloader instance.

        Parameters:
        url (str): The URL from which to download the CSV file.
        """
        self.url = url
        directory = 'downloads'
        current_directory = os.getcwd()
        downloads_directory = os.path.join(current_directory, directory)

        if not os.path.exists(downloads_directory):
            os.makedirs(downloads_directory)
        self.directory = downloads_directory
        self.df_agros = self.download_and_read_file()

    def download_and_read_file(self):
        """
        Downloads the CSV file from the specified URL and saves it in the `downloads` directory.
        If the file is already present in the directory, it loads the file into a pandas DataFrame.
        Otherwise, it creates the directory, downloads the file, and then loads it into a DataFrame.

        Returns:
        pandas.DataFrame: The loaded pandas DataFrame.
        """
        file_name = "agriculture_dataset.csv"
        file_path = os.path.join(self.directory, file_name)

        if os.path.exists(file_path):
            print(f"{file_name} already exists in {self.directory}")
        else:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(self.url, file_path)
            print(f"{file_name} downloaded to {self.directory}")

        df_agros = pd.read_csv(file_path)
        return df_agros
    
    def countries_list(self):
        """
        Returns all distinct entries in the countries column of the 
        class' dataframe in the form of a list. Removes any occurrence
        of macro-regions or income-based geographical distributions and only
        shows country names.
        
        Parameters
        ---------------
        self
        Refers to the class to which the module belongs to.
        
        Returns
        ---------------
        countries: list
        A list of all distinct countries in the dataframe.
        """
        countries = list(self.df_agros["Entity"].unique())
        non_countries = ["Central Africa", "Central America",
                "Central Asia", "Central Europe",
                 "Developed Asia", "Developed Countries",
                 "East Africa", "Eastern Europe", "Europe",
                 "Former Soviet Union", "High income", 
                 "Horn of Africa", "Latin America and the Caribbean",
                 "Least developed countries", "Low income",
                 "Lower-middle income", "North Africa", "North America",
                 "Northeast Asia", "Northern Europe", "Oceania",
                 "Pacific", "Sahel", "Serbia and Montenegro",
                 "World", "Western Europe", "West Asia",
                 "West Africa", "Western Europe",
                 "Upper-middle income", "Sub-Saharan Africa",
                 "Southern Europe", "Southern Africa", "Southern Asia"]
        
        cleaned_countries = [i for i in countries if i not in non_countries]
        return cleaned_countries
    
    def area_chart (self, country:str, normalize:bool):
        """
        Plots an area chart tracking the evolution of output quantity
        segmented by type (crop, animal and fish) over time. A
        string can be passed to the "country" argument to select the
        country for which data is diplayed, passing None or "World"
        will cause the method to use globally aggregated data. When
        the "normalize" argument is set to "True", the figures are no
        longer displayed in absolute terms, but in percentage of the
        total output.
           
        Parameters
        ---------------
        country: str
        The country for which data is being plotted, passing "World"
        or None will yield globally aggregated data.
        
        normalize: bool
        A boolean that signals whether data should be presented
        as a percentage of total output (when True) or in absolute
        units (when False).
            
        Returns
        ---------------
        Plots the figure requested.
        """
        sns.set_theme()
        output_df = self.df_agros[["Entity",
                                "Year",
                                "output_quantity",
                                "crop_output_quantity",
                                "animal_output_quantity",
                                "fish_output_quantity"]]

        if country in self.countries_list():
            plot_df = output_df[output_df["Entity"]==country]
            if normalize is False:
                plt.stackplot(plot_df["Year"],
                              plot_df["crop_output_quantity"]/10**9,
                              plot_df["animal_output_quantity"]/10**9,
                              plot_df["fish_output_quantity"]/10**9)
                plt.ylabel("Output Quantity by Type (Billions)")
            if normalize is True:
                plt.stackplot(plot_df["Year"],
                              (plot_df["crop_output_quantity"]/plot_df["output_quantity"])*100,
                              (plot_df["animal_output_quantity"]/plot_df["output_quantity"])*100,
                              (plot_df["fish_output_quantity"]/plot_df["output_quantity"])*100)
                plt.ylabel("% of Output by Type")

            plt.xlabel("Year")
            plt.legend(["Crop", "Animal", "Fish"])
            
        elif country in [None, "World"]:
            plot_df = pd.DataFrame()
            plot_df["year_total"] = output_df[["output_quantity", "Year"]]\
            .groupby(["Year"]).sum()/10**9
            plot_df["crop_total"] = output_df[["crop_output_quantity", "Year"]]\
            .groupby(["Year"]).sum()/10**9
            plot_df["animal_total"] = output_df[["animal_output_quantity", "Year"]]\
            .groupby(["Year"]).sum()/10**9
            plot_df["fish_total"] = output_df[["fish_output_quantity", "Year"]]\
            .groupby(["Year"]).sum()/10**9
            if normalize is False:
                plt.stackplot(output_df["Year"].unique(),
                              plot_df["crop_total"],
                              plot_df["crop_total"],
                              plot_df["fish_total"])
                plt.ylabel("Output Quantity by Type (Billions)")
            if normalize is True:
                plt.stackplot(output_df["Year"].unique(),
                              (plot_df["crop_total"]\
                               /plot_df["year_total"])*100,
                              (plot_df["animal_total"]\
                               /plot_df["year_total"])*100,
                              (plot_df["fish_total"]\
                               /plot_df["year_total"])*100)
                plt.ylabel("% of Output by Type")
            plt.xlabel("Year")
            plt.legend(["Crop", "Animal", "Fish"])
        else:
            raise ValueError("Inserted Country is not in Dataset")
    
    def __gapminder__(year):
        """
        This method plots a scatter plot of fertilizer_quantity vs 
        output_quantity with the size of each dot determined
        by the Total factor productivity , for a given year.

        :param year: Year of the harvest. Used for the scatter plot.
        :type year: int
        :raises TypeError: In case year is not an integer.
        """

        if not isinstance(year, int):
            raise TypeError("Year must be an integer.")

        agriculture_df = pd.read_csv("downloads/agriculture_dataset.csv")
        agriculture_filtered_df = agriculture_df[agriculture_df['Year'] == year]

        # Plot the scatter plot
        fig, ax = plt.subplots()
        ax.scatter(agriculture_filtered_df['fertilizer_quantity'], 
                   agriculture_filtered_df['output_quantity'],
                   s=agriculture_filtered_df['tfp'], alpha=0.6)
        ax.set_xlabel('Fertilizer Quantity')
        ax.set_ylabel('Output Quantity')
        ax.set_title(f'Crops Production in {year}')
        plt.show()

#AgrosClass.__gapminder__(2014)
#dd = AgrosClass()
#print(dd.df_agros.head())
