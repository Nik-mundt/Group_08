# Group_08
## AgrosClass
The AgrosClass is a Python module that provides functionality for analyzing agricultural data and combines it with additional data from Natural Earth.
There are 9 methods available, which range from downloading the data to more complex visualizations and predictions of trends.

## Installation
To use the Project, you can simply clone the repository from GitHub using the following bash statement:


```bash
git clone git@github.com:Nik-mundt/Group_08.git
```

## Usage
To use the AgrosClass, you can import it in your Python code as follows:

```python
from class_code.agros_class import AgrosClass
```
You can then create an instance of the class and use its methods to analyze the agricultural data. Here are the methods available in the AgrosClass:

`__init__(self, url)`
Initializes the AgrosClass instance and downloads the agricultural data CSV file from a specified URL.

`download_and_read_file(self)`
Downloads the CSV file from the specified URL and a zip file from Natural Earth data which includes additional country information. Both files are then saved in the downloads directory. If the files are already present in the directory, it loads both of them into a pandas DataFrame each. Otherwise, it creates the directory, downloads the files, and then loads them into a DataFrame respectively.

`countries_list(self)`
Returns all distinct entries in the countries column of the class' DataFrame in the form of a list. Removes any occurrence of macro-regions or income-based geographical distributions and only shows country names.

`area_chart(self, country, normalize)`
Plots an area chart tracking the evolution of output quantity segmented by type (crop, animal and fish) over time. A string can be passed to the country argument to select the country for which data is displayed, passing None or "World" will cause the method to use globally aggregated data. When the normalize argument is set to True, the figures are no longer displayed in absolute terms, but in percentage of the total output.

`__gapminder__(self, year)`
Plots a scatter plot of fertilizer_quantity vs output_quantity with the size of each dot determined by the total factor productivity, for a given year.

`corr_matrix(self, keyword)`
Calculates and displays a correlation matrix heatmap for the columns in df_agros that contain the specified keyword in their column name.

`output_graph(self, countries)`
Creates a plot of the total output quantity over time for a list of countries or a single country.

`country_cleaning(self, df_agros, df_geo)`
This method takes in two pandas DataFrames, df_agros and df_geo, and merges them together based on the country names. Before the merge any inconsistencies in the country names are corrected.

`predictor(self, countries_list)`
This method takes a list of up to three countries and plots their Total Factor Productivity (TFP) over time, along with a forecast until 2050 for each country. The function returns nothing and simply displays a graph of the selected countries. 

## Example
Here's an example of how to use the AgrosClass:

```python
from class_code.agros_class import AgrosClass

# Create an instance of the AgrosClass
agros = AgrosClass()

# Get a list of all countries in the data
countries = agros.countries_list()
print(countries)

# Plot the output quantity for selected countries over time
agros.output_graph(["United States", "China"])

# Create a correlation matrix heatmap for columns with "quantity" in their name
agros.corr_matrix(keyword="quantity")

# Show and predict the TFP of 3 countries
agros.predictor(["Italy", "Portugal", "Germany"])
```


## License
This project is licensed under the Apache License, Version 2.0. Please see the LICENSE file for more details.
