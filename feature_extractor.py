import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
import numpy as np

import os
import pickle	


def FeatureExtractorModel():
	"""
	Extract the last layer features from ResNet18.
	Intiialize the model accordingly
	"""

	model_ft = models.resnet18(pretrained=True)
	model = torch.nn.Sequential(*list(model_ft.children())[:-1])

	return model



def FeatureExtractor( input_dir = "./Image/", output_dir = "./Image_Features/" ):
	"""
	input_dir: Directory of input images	
	output_dir: Directory of output images
	"""

	transform = transforms.Compose( [  transforms.Resize((224, 224)), transforms.ToTensor(), 
					   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ] )

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = FeatureExtractorModel()
	model = model.to(device)
	model.eval()

	print ("model loaded")

	for image_path in os.listdir(input_dir):
	
		img = Image.open(input_dir + image_path).convert("RGB")

		img = transform(img).unsqueeze(0)
    
		img = img.to(device)

		with torch.no_grad():
			feature = model(img)
			feature = feature.numpy().squeeze()

		output_file = output_dir + image_path + ".pickle"
		
		with open(output_file , 'wb') as handle:
    			pickle.dump(feature, handle, protocol = pickle.HIGHEST_PROTOCOL)

		print (image_path)

	return 1

if __name__ == "__main__":
	FeatureExtractor()
