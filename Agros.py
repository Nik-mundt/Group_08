from matplotlib import pyplot as plt
import pandas as pd

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

__gapminder__(2014)