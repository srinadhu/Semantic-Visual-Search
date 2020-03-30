import hnswlib
import numpy as np
import time
import pickle
import os

def LoadFeature(feature_path):
	"""
	feature_path: The feature path of the feature

	Returns:
		feature: The output feature
		image_name: The image name as well
	"""

	feature = pickle.load(open(feature_path, "rb"))
	image_name = feature_path.split("/")[-1].split(".")[0]

	return feature, image_name


EMBEDDING_SIZE = 512
num_elements = 1400000

# Declaring index
p = hnswlib.Index(space = 'l2', dim = EMBEDDING_SIZE) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 400, M = 64)

# Element insertion (can be called several times)
for feature_path in os.listdir("./Image_Features/"):
	feature, label = LoadFeature("./Image_Features/" + feature_path)
	p.add_items(feature, label)

# Controlling the recall by setting ef:
p.set_ef(400) # ef should always be > k


p.save_index('index.idx')


