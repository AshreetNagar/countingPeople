from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from itertools import zip_longest
from utils.mailer import Mailer
from imutils.video import FPS
from utils import thread
import numpy as np
import threading
import argparse
import datetime
import schedule
import logging
import imutils
import time
import dlib
import json
import csv
import cv2

start_time = time.time()

logging.basicConfig(level = logging.INFO, format = "[INFO] %(message)s")
logger = logging.getLogger(__name__)

with open("utils/config.json", "r") as file:
    config = json.load(file)

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())
    return args

def send_mail():
	# Function to send the email alerts
	Mailer().send(config["Email_Receive"])

def log_data(move_in, in_time, move_out, out_time):
	data = [move_in, in_time, move_out, out_time]
	# Transpose the data to align the columns properly
	export_data = zip_longest(*data, fillvalue = '')

	with open('utils/data/logs/counting_data.csv', 'w', newline = '') as myfile:
		wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
		if myfile.tell() == 0:
			wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
			wr.writerows(export_data)

def people_counter():
	args = parse_arguments()
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# If the video path was not supplied, grab a reference to the ip camera
	if not args.get("input", False):
		logger.info("Starting the live stream..")
		vs = VideoStream(config["url"]).start()
		time.sleep(2.0)

	# get a reference to the video file
	else:
		logger.info("Starting the video..")
		vs = cv2.VideoCapture(args["input"])

	writer = None

	W = None
	H = None

	# Instantiate our centroid tracker, then initialize a list to store each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# Initialize the total number of frames processed thus far, along with the total number of objects that have moved either up or down
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	total = []
	move_out = []
	move_in =[]
	out_time = []
	in_time = []

	fps = FPS().start()

	if config["Thread"]:
		vs = thread.ThreadingClass(config["url"])

	while True:
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		if args["input"] is not None and frame is None:
			break

		# Resize the frame to have a maximum width of 500 pixels and convert the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		if W is None or H is None:
			(H, W) = frame.shape[:2]

		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		status = "Waiting"
		rects = []

		if totalFrames % args["skip_frames"] == 0:
			status = "Detecting"
			trackers = []

			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]

				# Filter out weak detections by requiring a minimum
				if confidence > args["confidence"]:
					# Extract the index of the class label from the
					idx = int(detections[0, 0, i, 1])

					if CLASSES[idx] != "person":
						continue

					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					trackers.append(tracker)

		else:
			for tracker in trackers:
				status = "Tracking"

				tracker.update(rgb)
				pos = tracker.get_position()

				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				rects.append((startX, startY, endX, endY))

		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# Use the centroid tracker to associate the old object centroids with the newly computed object centroids
		objects = ct.update(rects)

		for (objectID, centroid) in objects.items():
			to = trackableObjects.get(objectID, None)

			if to is None:
				to = TrackableObject(objectID, centroid)

			else:
				# Find the direction that the object is moving
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				if not to.counted:
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
						move_out.append(totalUp)
						out_time.append(date_time)
						to.counted = True

					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
						move_in.append(totalDown)
						in_time.append(date_time)
						# If the people limit exceeds over threshold, send an email alert
						if sum(total) >= config["Threshold"]:
							cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
								cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							if config["ALERT"]:
								logger.info("Sending email alert..")
								email_thread = threading.Thread(target = send_mail)
								email_thread.daemon = True
								email_thread.start()
								logger.info("Alert sent!")
						to.counted = True
						total = []
						total.append(len(move_in) - len(move_out))


			trackableObjects[objectID] = to

			# Draw ID and centroid of the object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# Construct a tuple of information we will be displaying on the frame
		info_status = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info_total = [
		("Total people inside", ', '.join(map(str, total))),
		]

		for (i, (k, v)) in enumerate(info_status):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info_total):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		if config["Log"]:
			log_data(move_in, in_time, move_out, out_time)

		if writer is not None:
			writer.write(frame)

		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

		totalFrames += 1
		fps.update()

		# Timer
		if config["Timer"]:
			# Automatic timer to stop the live stream (set to 8 hours/28800s)
			end_time = time.time()
			num_seconds = (end_time - start_time)
			if num_seconds > 28800:
				break

	# Stop the timer and display FPS information
	fps.stop()
	logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
	logger.info("Approx. FPS: {:.2f}".format(fps.fps()))

	# Release the camera device/resource
	if config["Thread"]:
		vs.release()

	# Close any open windows
	cv2.destroyAllWindows()

# Initiate the scheduler
if config["Scheduler"]:
	schedule.every().day.at("09:00").do(people_counter)
	while True:
		schedule.run_pending()
else:
	people_counter()
