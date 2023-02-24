from matplotlib import pyplot as plt
import urllib.request
import os
import pandas as pd

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
    
    def __gapminder__(year):
        """
        This method plots a scatter plot of fertilizer_quantity vs output_quantity with the size of each dot determined
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
        ax.scatter(agriculture_filtered_df['fertilizer_quantity'], agriculture_filtered_df['output_quantity'],
                s=agriculture_filtered_df['tfp'], alpha=0.6)
        ax.set_xlabel('Fertilizer Quantity')
        ax.set_ylabel('Output Quantity')
        ax.set_title(f'Crops Production in {year}')
        plt.show()

#AgrosClass.__gapminder__(2014)
#dd = AgrosClass()
#print(dd.df_agros.head())