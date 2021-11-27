import streamlit as st
from streamlit_folium import folium_static
import folium

def app_view_map():
    # center on Liberty Bell
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)

    # add marker for Liberty Bell
    tooltip = "Liberty Bell"
    folium.Marker(
        [39.949610, -75.150282], popup="Liberty Bell", tooltip=tooltip
    ).add_to(m)

    # call to render Folium map in Streamlit
    return folium_static(m)

#def app_view_map():
#    HtmlFile = open("map.html", 'r', encoding='utf-8')
#    source_code = HtmlFile.read()
#    components.html(source_code, height=1600, width=1600)