"""
Part 2
"""
import folium
import geopandas as gpd
from class_code import *

agros = AgrosClass()

agros_df = agros.df_agros
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

merged_df = world.merge(agros_df, how='left', left_on='name', right_on='Entity')

#Creaate a map object for choropleth map
#Set location to your location of interest (latitude and longitude )
map0 = folium.Map(location=[23.9,121.52], zoom_start=7)

#Create choropleth map object with key on TOWNNAME
folium.Choropleth(geo_data = merged_df,#Assign geo_data to your geojson file
    name = "choropleth",
    data = merged_df,#Assign dataset of interest
    columns = ["name","tfp"],#Assign columns in the dataset for plotting
    key_on = 'feature.properties.name',#Assign the key that geojson uses to connect with dataset
    fill_color = 'YlOrRd',
    fill_opacity = 0.7,
    line_opacity = 0.5,
    legend_name = 'Taiwan').add_to(map0)

#Create style_function
style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}

#Create highlight_function
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

#Create popup tooltip object
NIL = folium.features.GeoJson(
    merged_df,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['tfp'],
        aliases=['tfp'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")))

#Add tooltip object to the map
map0.add_child(NIL)
map0.keep_in_front(NIL)
folium.LayerControl().add_to(map0)

map0