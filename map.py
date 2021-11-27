import folium
import random

from folium.map import Icon, Tooltip
# Random Marker Generator
def marker_generator():
    pass

m= folium.Map(location=[1,114], zoom_start=7)

tooltip_Fire ="Fire. Click for more info"
html_embed_yt = """<iframe width="560" height="315" src="https://www.youtube.com/embed/HSsqzzuGTPo?start=2243" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>"""
fireLogo = folium.CustomIcon('fire.png', icon_size=(50, 50))
iframe_YT = folium.IFrame(html=html_embed_yt,height="400px",width="600px")
folium.Marker([1.1, 114.5],
            popup='<strong>Fire 01</strong>',
            tooltip=tooltip_Fire,
            icon=folium.Icon(color='red',icon='fire')).add_to(m)

folium.Marker([1.3, 113.8],
            popup=folium.Popup(iframe_YT, max_width=800),
            tooltip=tooltip_Fire,
            icon=folium.Icon(color='red',icon='fire')).add_to(m)


# Generate Map
m.save('map.html')