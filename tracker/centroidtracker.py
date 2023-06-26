from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# Initialize the next object ID used to keep track of mapping an object ID to its centroid and number of frames marked as "disappeared"
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# Max number of frames an object can be marked as "disappeared" until it can be deregistered
		self.maxDisappeared = maxDisappeared

		# Max distance between centroids. It is "disappeared" if the distance is larger than the max
		self.maxDistance = maxDistance

	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# Deregister if max number of frames were given is marked as missing
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# Take input centroids and register them if there are no objects being tracked
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		else:
			# Grab the set of object IDs and centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# Compute distance between pair of object and input centriods
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# Find the smallest value in each row and sort the row based on the min values
			rows = D.min(axis=1).argsort()

			# Same process as above but with columns
			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				if D[row, col] > self.maxDistance:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# If the number of object centroids is greater than or equal to the number of input centroids, check if some  objects disappeared
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# if the number of input centroids is greater than the number of existing object centroids, register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
					
		return self.objects