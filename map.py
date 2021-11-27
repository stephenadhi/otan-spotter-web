import folium
import random

# Constants
BORNEO_UPPER = 0.78
BORNEO_LOWER = -2.16
BORNEO_LEFT = 115.52
BORNEO_RIGHT = 110.49

INTERVAL_HORIZONTAL = BORNEO_RIGHT - BORNEO_LEFT
INTERVAL_VERTICAL = BORNEO_UPPER - BORNEO_LOWER

from folium.map import Icon, Tooltip
# Random Marker Generator
def marker_generator(foliumMap, tooltipStr, foliumIframe):
    pass
    # number of markers
    n = random.randrange(start=2,stop= 10, step=1)
    print("random markers N= ", n)
    for i in range(n):
        randLat= random.random()*INTERVAL_VERTICAL
        randLon= random.random()*INTERVAL_HORIZONTAL
        randLoc= [BORNEO_LOWER+randLat,BORNEO_LEFT + randLon] 
        
        folium.Marker(randLoc,
            popup=folium.Popup(foliumIframe, max_width=800),
            tooltip=tooltipStr,
            icon=folium.Icon(color='red',icon='fire')).add_to(foliumMap)




# Initialize Map
m= folium.Map(location=[1,114], zoom_start=7)

# Custom Parameters
tooltip_Fire ="Fire. Click for more info"
html_embed_yt = """<iframe width="560" height="315" src="https://www.youtube.com/embed/HSsqzzuGTPo?start=2243" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>"""
fireLogo = folium.CustomIcon('fire.png', icon_size=(50, 50))
iframe_YT = folium.IFrame(html=html_embed_yt,height="400px",width="600px")

# Markers
folium.Marker([1.1, 114.5],
            popup='<strong>Fire 01</strong>',
            tooltip=tooltip_Fire,
            icon=folium.Icon(color='red',icon='fire')).add_to(m)

folium.Marker([1.3, 113.8],
            popup=folium.Popup(iframe_YT, max_width=800),
            tooltip=tooltip_Fire,
            icon=folium.Icon(color='red',icon='fire')).add_to(m)
            
marker_generator(foliumMap=m, tooltipStr=tooltip_Fire,foliumIframe=iframe_YT)

# Generate Map
m.save('map.html')