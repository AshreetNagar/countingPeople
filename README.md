A Single Shot Detector with MobileNet architecture was used to take a single shot to detect what is in an image (For generating the region and for detecting the object). These were chosen since SSD is faster than two shot detectors and MobileNet is designed to be ran on resource constrained devices.

The centriod tracker computed the centroid of the bounding boxes (the (x,y) coordinates) which are from the SSD, the tracker will compute the center of an object and there will be a unique ID for every object detected.

The timer to stop the execution can be set in the script as well.

# How to Run with different methods:
Tested on Python 3.11.3
1. Install the required Python dependencies
    ``` pip install -r requirements.txt ```
2. Test the video file by going into the root directory and running. Make a video and call it test_1.mp4
    ``` python people_counter.py --prototxt detector/MobileNetSSD_deploy.prototxt --model detector/MobileNetSSD_deploy.caffemodel --input utils/data/tests/test_1.mp4 ```
3. Test with Webcam by going to ```utils/config.json``` and set ```"url": 0``` and run
    ``` python people_counter.py --prototxt detector/MobileNetSSD_deploy.prototxt --model detector/MobileNetSSD_deploy.caffemodel ```
4. Test with IP Camera by going to ```utils/config.json``` and set ```"url": http://IPHERE/video``` (Put the IP where IPHERE is) and run
    ```python people_counter.py --prototxt detector/MobileNetSSD_deploy.prototxt --model detector/MobileNetSSD_deploy.caffemodel```

# Email
The max number of people limit  can be set in ```utils/config.json``` like ```"Threshold": 10```
To set up the sender email, ```"Email_Send": ""``` and to the receiver, ```"Email_Receive": ""```. The sender password would also need to be set in ```"Email_Password": ""```

# Threading
Threading removes OpenCV's internal buffer which will increase frame rate. Threading also helps if the system cannot process and output the result simutaneously which causes a delay in the stream. To enable threading go to ```utils/config.json``` and set ```"Thread": true```