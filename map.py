import folium
import random

# Constants
BORNEO_UPPER = 0.78
BORNEO_LOWER = -2.16
BORNEO_LEFT = 115.52
BORNEO_RIGHT = 110.49

# Icon Websites


INTERVAL_HORIZONTAL = BORNEO_RIGHT - BORNEO_LEFT
INTERVAL_VERTICAL = BORNEO_UPPER - BORNEO_LOWER

from folium.map import Icon, Tooltip
# Random Marker Generator
def marker_generator(foliumMap, tooltipStr):
    # Fire generator
    # number of markers
    n_fire = random.randrange(start=2,stop= 10, step=1)
    print("random markers N= ", n_fire)
    
    for i in range(n_fire):
        randLat= random.random()*INTERVAL_VERTICAL
        randLon= random.random()*INTERVAL_HORIZONTAL
        randLoc= [BORNEO_LOWER+randLat,BORNEO_LEFT + randLon] 
        randLocFormatted = f'[{randLoc[0]:.3f}, {randLoc[1]:.3f}]'
        
        # radius in Meter
        # Calculator https://www.sensorsone.com/circle-area-to-radius-calculator
        randRadius= random.random()*20000 + 5000 # in meters, minimum 5000, max 25000
        randRadiusFormatted= f'{randRadius:.3f} m'
        html_Fire = f"<h2 id=\"fire-{i+1}\">Fire {i+1}</h2>"+f"<p>Location:{randLocFormatted}</p>"+f"<p>Radius:{randRadiusFormatted}</p>"
                    
        iframe_Fire = folium.IFrame(html=html_Fire, height="200px", width="300px")

        folium.Circle(randLoc,
            radius= randRadius
        ).add_to(foliumMap)
        folium.Marker(randLoc,
            popup=folium.Popup(iframe_Fire, max_width=800),
            tooltip=tooltipStr,
            icon=folium.Icon(color='red',icon='fire')).add_to(foliumMap)

    # Animal Population Markers Generator
    n_population_marker = random.randrange(start=3, stop= 12, step=1)
    for j in range(n_population_marker):
        randLat= random.random()*INTERVAL_VERTICAL
        randLon= random.random()*INTERVAL_HORIZONTAL
        randLoc= [BORNEO_LOWER+randLat,BORNEO_LEFT + randLon]

        randPop = random.randrange(start=4, stop= 16, step=1)
        randGroup = random.randrange(start=4, stop= 64, step=1)
        randLocFormatted = f'[{randLoc[0]:.3f}, {randLoc[1]:.3f}]'
        htmlPopup = f"<h2 id=\"animal-group\">Group ID #{randGroup}</h2><p><strong>Location</strong>: {randLocFormatted}</p><p><strong>Population</strong>: {randPop}</p>"
        folium.Marker(randLoc,
            popup=folium.Popup(html=htmlPopup, max_width= 300),
            tooltip="Animal Group. Click for more details.",
            icon=folium.Icon(color='green', icon='paw', prefix='fa')).add_to(foliumMap)


# Initialize Map
m= folium.Map(location=[1,114], zoom_start=7)

# Custom Parameters
tooltip_Fire ="Fire. Click for more info"
# html_embed_yt = """<iframe width="560" height="315" src="https://www.youtube.com/embed/HSsqzzuGTPo?start=2243" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>"""
html_embed_yt = """<iframe width="560" height="315" src="https://www.youtube.com/embed/x9Met3iro8c" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>"""
fireLogo = folium.CustomIcon('fire.png', icon_size=(50, 50))
iframe_YT = folium.IFrame(html=html_embed_yt,height="400px",width="600px")

# Markers
## Drone
randLat= random.random()*INTERVAL_VERTICAL
randLon= random.random()*INTERVAL_HORIZONTAL
randLoc= [BORNEO_LOWER+randLat,BORNEO_LEFT + randLon]

randPop = random.randrange(start=4, stop= 16, step=1)
randGroup = random.randrange(start=4, stop= 64, step=1)
randLocFormatted = f'[{randLoc[0]:.3f}, {randLoc[1]:.3f}]'
htmlPopupDrone = f"<h2 id=\"Drone\">Drone ID #{randGroup}</h2><p><strong>Location</strong>: {randLocFormatted}</p>"
htmlPopupDrone_video = htmlPopupDrone + html_embed_yt

iframe_YT_Drone = folium.IFrame(html=htmlPopupDrone_video, height="500px", width= "600px")

folium.Marker(randLoc,
    popup=folium.Popup(iframe_YT_Drone, max_width= 800),
    tooltip="Drone. Click for more details.",
    icon=folium.Icon(color='blue', icon='plane', prefix='fa')).add_to(m)

## Fire (Legacy)
# folium.Marker([1.1, 114.5],
#             popup='<strong>Fire 01</strong>',
#             tooltip=tooltip_Fire,
#             icon=folium.Icon(color='red',icon='fire')).add_to(m)

# folium.Marker([1.3, 113.8],
#             popup=folium.Popup(iframe_YT, max_width=800),
#             tooltip=tooltip_Fire,
#             icon=folium.Icon(color='red',icon='fire')).add_to(m)
            
marker_generator(foliumMap=m, tooltipStr=tooltip_Fire)

# Generate Map
m.save('map.html')