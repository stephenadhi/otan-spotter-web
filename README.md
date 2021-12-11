# Otan Spotter Web Interface
- Dashboard
- Live bounding box predictions and object tracking from drone videos
- Rescue and notification alarm
- Deployment: 
- 
# Setup
To install dependencies:

    pip install -r requirements.txt 
 
To run inference, put video test.mp4 in root directory. 

    python detect.py --source test.mp4 --weight models/best0512.pt  --view-img 



