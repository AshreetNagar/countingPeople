class TrackableObject:
	def __init__(self, objectID, centroid):
		# Store object Id and initialize list of centroids with the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		self.counted = False