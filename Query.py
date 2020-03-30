import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image

import hnswlib
import numpy as np
import time
import os

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec


def plot_predictions(images):
	gs = gridspec.GridSpec(3, 3)
	fig = plt.figure(figsize=(15, 15))
	gs.update(hspace=0.1, wspace=0.1)
	for i, (gg, image) in enumerate(zip(gs, images)):
		gg2 = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=gg)
		ax = fig.add_subplot(gg2[:,:])
		ax.imshow(image, cmap='Greys_r')
		ax.tick_params(axis='both',       
					   which='both',      
					   bottom='off',      
					   top='off',         
					   left='off',
					   right='off',
					   labelleft='off',
					   labelbottom='off') 
		ax.axes.set_title("result [{}]".format(i))
		if i == 0:
			plt.setp(ax.spines.values(), color='red')
			ax.axes.set_title("SEARCH".format(i))
	plt.show()


def FeatureExtractorModel():
	"""
	Extract the last layer features from ResNet50.
	Intiialize the model accordingly
	"""

	model_ft = models.resnet18(pretrained=True)
	model = torch.nn.Sequential(*list(model_ft.children())[:-1])

	return model

transform = transforms.Compose( [  transforms.Resize((224, 224)), transforms.ToTensor(), 
					   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ] )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FeatureExtractorModel()
model = model.to(device)
model.eval()

print ("model loaded")


EMBEDDING_SIZE = 512
num_elements = 100000

# Reiniting, loading the index
p = hnswlib.Index(space = 'l2', dim = EMBEDDING_SIZE)  # the space can be changed - keeps the data, alters the distance function.

print("\nLoading index from 'index.idx'\n")

# Increase the total capacity (max_elements), so that it will handle the new data
p.load_index("index.idx", max_elements = num_elements)

p.set_ef(400) # ef should always be > k

# Query the elements:
for query in os.listdir("./Test_Images/"):
	start = time.time()
	
	img = Image.open("./Test_Images/" + query).convert("RGB")

	img = transform(img).unsqueeze(0)
	
	img = img.to(device)

	with torch.no_grad():
		feature = model(img)
		feature = feature.numpy().squeeze()

	labels, distances = p.knn_query(feature, k = 10)

	end = time.time()

	print(query, labels, end-start)
	
	images = [plt.imread("./Image/" + str(label) + ".jpg") for label in labels[0]]
	plot_predictions(images)
