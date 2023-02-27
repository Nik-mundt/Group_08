"""
Defines class "AgrosClass" which downloads the necessary dataset
and reads it as a Pandas DataFrame and retains it as an attribute.
Further, the class has 6 methods which facilitate the analysis
of data for our final Python Notebook.
"""
import urllib.request
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class AgrosClass:
    """
    Downloads a CSV file from a given URL, saves it in
    a local folder, and reads it into a pandas DataFrame
    that is an attribute of the class. The CSV is stored
    in a "downloads" folder in the file's directory. If the
    folder does not exist, the class automatically creates
    it.

    The class downloads the USDA's Agricultural Total Factor
    Productivity dataset by default, but a different URL can be
    passed as an argument if necessary. A set of methods can
    be used to analyze the data visually.

    Attributes
    ----------
    directory:
        Directory of the downloads folder
    df_agros: DataFrame
        DataFrame containing the downloaded CSV

    Methods
    --------
    download_and_read_file()
        Downloads and reads the CSV from URL.
        Creates downloads directory if necessary.
    countries_list()
        Returns a list of all countries in the dataset.
    area_chart()
        Plots an area chart displaying the composition
        of a country's output segmented by output type.
    __gapminder__()
        Generates a scatter plot displaying the relationship
        between fertilizer use and output for a given year.
        Mark size represents factor productivity.
    corr_matrix()
        Calculates and displays a correlation matrix heatmap for
        the columns in the DataFrame containing a specified keyword.
    output_graph()
        Generates a plot of total output over time for a selected
        country or list of countries.
    """
    def __init__(self,
                 url='https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv'):
        """
        Initializes the DataDownloader instance.

        Parameters
        ---------------
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

        Returns
        ---------------
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

    def area_chart(self, country: str, normalize: bool):
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
        None
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
            plot_df = output_df[output_df["Entity"] == country]
            if normalize is False:
                plt.stackplot(plot_df["Year"],
                              plot_df["crop_output_quantity"] / 10 ** 9,
                              plot_df["animal_output_quantity"] / 10 ** 9,
                              plot_df["fish_output_quantity"] / 10 ** 9)
                plt.ylabel("Output Quantity by Type (Billions)")
                plt.title(f"Output Quantity by Output Type Over Time ({country})")

            if normalize is True:
                plt.stackplot(plot_df["Year"],
                              (plot_df["crop_output_quantity"] /plot_df["output_quantity"]) *100,
                              (plot_df["animal_output_quantity"]/plot_df["output_quantity"]) *100,
                              (plot_df["fish_output_quantity"] /plot_df["output_quantity"]) *100)
                plt.ylabel("% of Output by Type")
                plt.title(f"Share of Total Output by Output Type ({country})")

            plt.xlabel("Year")
            plt.legend(["Crop", "Animal", "Fish"])

        elif country in [None, "World"]:
            plot_df = pd.DataFrame()
            plot_df["year_total"] = output_df[["output_quantity", "Year"]] \
                                        .groupby(["Year"]).sum() / 10 ** 9
            plot_df["crop_total"] = output_df[["crop_output_quantity", "Year"]] \
                                        .groupby(["Year"]).sum() / 10 ** 9
            plot_df["animal_total"] = output_df[["animal_output_quantity", "Year"]] \
                                          .groupby(["Year"]).sum() / 10 ** 9
            plot_df["fish_total"] = output_df[["fish_output_quantity", "Year"]] \
                                        .groupby(["Year"]).sum() / 10 ** 9

            if normalize is False:
                plt.stackplot(output_df["Year"].unique(),
                              plot_df["crop_total"],
                              plot_df["crop_total"],
                              plot_df["fish_total"])
                plt.ylabel("Output Quantity by Type (Billions)")
                plt.title("Output Quantity by Output Type Over Time (World)")

            if normalize is True:
                plt.stackplot(output_df["Year"].unique(),
                              (plot_df["crop_total"] \
                               / plot_df["year_total"]) * 100,
                              (plot_df["animal_total"] \
                               / plot_df["year_total"]) * 100,
                              (plot_df["fish_total"] \
                               / plot_df["year_total"]) * 100)
                plt.ylabel("% of Output by Type")
                plt.title("Share of Total Quantity by Output Type Over Time\
                (World)")

            plt.xlabel("Year")
            plt.legend(["Crop", "Animal", "Fish"])

        else:
            raise ValueError("Inserted Country is not in Dataset")

    def __gapminder__(self, year):
        """
        This method plots a scatter plot of fertilizer_quantity vs
        output_quantity with the size of each dot determined
        by the total factor productivity, for a given year.

        Parameters
        ---------------
        year (int): Year of the harvest. Used for the scatter plot.

        Returns
        ---------------
        None
            Displays a gapminder-style scatter plot between fertilizer
            quantity and output quantity.
        """

        if not isinstance(year, int):
            raise TypeError("Year must be an integer.")
        cleaned_countries = self.countries_list()
        agriculture_filtered_df = self.df_agros.loc[(self.df_agros['Year'] == year)
                                                    & self.df_agros["Entity"]\
                                                    .isin(cleaned_countries)]

        # Plot the scatter plot
        fig_plot, ax_plot = plt.subplots()
        ax_plot.scatter(agriculture_filtered_df['fertilizer_quantity'],
                   agriculture_filtered_df['output_quantity'],
                   s=agriculture_filtered_df['tfp'], alpha=0.6)
        ax_plot.set_xlabel('Fertilizer Quantity')
        ax_plot.set_ylabel('Output Quantity')
        ax_plot.set_title(f'Crops Production in {year}')
        fig_plot.show()

    def corr_matrix(self, keyword="quantity"):
        """
        Calculates and displays a correlation matrix heatmap for the columns in
        df_agros that contain the specified keyword in their column name.

        Parameters
        ---------------
        keyword (str): The keyword to search for in column names. Default is "quantity".

        Returns
        ---------------
        None
            Displays a heatmap of the correlation matrix
        """
        # Select columns that contain the keyword
        keyword_cols = [col for col in self.df_agros.columns if keyword in col]
        keyword_df = self.df_agros[keyword_cols]

        # Calculate correlation matrix
        correlation_matrix = keyword_df.corr()
        corr_df = pd.DataFrame(correlation_matrix)

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        corr_heatmap = sns.heatmap(corr_df, cmap="Greens", annot=True,
                                   annot_kws={"size": 7, "color": "black"}, mask=mask)
        corr_heatmap.set_xticklabels(corr_heatmap.get_xticklabels(), fontsize=7)
        corr_heatmap.set_yticklabels(corr_heatmap.get_yticklabels(), fontsize=7)

        plt.show()

    def output_graph(self, countries):
        """
        Receives a list of countries or a single country as input and creates a plot of the
        total output quantity over time for the selected countries.
        Parameters
        ---------------
        countries: str, list of str
            Countries of which a plot is created

        Returns
        ---------------
        None
            Displays a graph of the selected countries
        """
        if isinstance(countries, list):
            for country in countries:
                #check if each input is a string
                if not isinstance(country, str):
                    raise ValueError("Input should be a string or a list of strings")
                #verify that country exists in database (not being case sensitive)
                country_upper = country[0].upper()+country[1:]
                if country_upper not in self.df_agros['Entity'].unique():
                    raise ValueError(country+" is not a valid country")
                #check if country starts with a capital letter
                if country[0].isupper() is False:
                    lowercase = country[0]+country[1:]
                    uppercase = country[0].upper()+country[1:]
                    raise ValueError("All countries should start with a capital letter. You wrote \
                                     "+lowercase+ " instead of "+ uppercase)
            #filter countries
            df_countries = self.df_agros[self.df_agros['Entity'].isin(countries)]
            total_output = df_countries.groupby(['Entity', 'Year'])['output_quantity'] \
                .sum().reset_index()
            # Create the plot for each country
            plt.figure(figsize=(10, 6))
            ax_output = sns.lineplot(x='Year', y='output_quantity', \
                                        hue='Entity', data=total_output)
            ax_output.set(xlabel='Year', ylabel='Output Quantity', title="Total output per Year")
            ax_output.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.show()
        elif isinstance(countries, str):
            if country[0].isupper() is False:
                raise ValueError("All countries should start with a capital letter. You wrote \
                                     "+lowercase+ " instead of "+ uppercase)
            df_country = self.df_agros[self.df_agros['Entity'] == countries]
            total_output = df_country.groupby('Year')['output_quantity'].sum().reset_index()
            # Create the plot for each country
            plt.figure(figsize=(10, 6))
            ax_output = sns.lineplot(x='Year', y='output_quantity', data=total_output)
            ax_output.set(xlabel='Year', ylabel='Output Quantity', title="Total output per Year")
            ax_output.legend([countries], loc='upper left', bbox_to_anchor=(1, 1))
            plt.show()
        else:
            raise ValueError("Input should be a string or a list of strings")
