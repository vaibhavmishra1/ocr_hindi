____________________________________________README_____________________________________

TOPIC: Optical Character Recognition of  Hindi words written in English without using Dictionary

Student :Vaibhav Mishra ,IIT Jodhpur
Submitted To: Prof. Chetan Arora

Description:
Project is based on using Deep Learning CNN model for character detection and recognition and thus predicting each characters of the words .
The model achieves object detection and recognition using CNN model and does not uses any dictionary for guessing words .The model is trained on EMNIST Dataset.
This helps in case of Hindi words written in English which cannot be predicted using dictionary .

My model works decent on complex environment with clearly visible and seperable text.


1-Dataset:
		I have used EMNIST Dataset which is the extension on the MNIST dataset for characters as well.
		It contains 62 classes with 0-9 digits and A-Z characters in both uppercase and lowercase. 
		Dataset Summary:
		There are six different splits provided in this dataset. A short summary of the dataset is provided below:

		EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
		EMNIST ByMerge: 814,255 characters. 47 unbalanced classes.
		EMNIST Balanced:  131,600 characters. 47 balanced classes.
		EMNIST Letters: 145,600 characters. 26 balanced classes.
		EMNIST Digits: 280,000 characters. 10 balanced classes.
		EMNIST MNIST: 70,000 characters. 10 balanced classes.

		i have trained my model on two different datasets i.e. 
		1-EMNIST ByClass(model.h5 and model.json)
		2-EMNIST Letters(model_small.h5 and model_small.json)

2-Model:
	the Model used is described as below:
		_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	reshape_1 (Reshape)          (None, 28, 28, 1)         0         
	_________________________________________________________________
	conv2d_1 (Conv2D)            (None, 28, 28, 32)        832       
	_________________________________________________________________
	conv2d_2 (Conv2D)            (None, 24, 24, 32)        25632     
	_________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 4608)              0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 512)               2359808   
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 512)               0         
	_________________________________________________________________
	dense_2 (Dense)              (None, 62)                31806     
	=================================================================
	Total params: 2,418,078
	Trainable params: 2,418,078
	Non-trainable params: 0

3-Accuracy:
	epochs-20
	accuracy is 93.36% for EMNIST Letters(model1.h5 and model1.h5) trained on my PC
	 and
	97.84% for EMNIST ByClass: 814,255 characters. 62 unbalanced classes trained on my institute GPU.
	results sheet has been attached for EMNIST Letters names as results.txt



#################################### HOW TO USE ###################################################

	1-train.py:
		run commands -: python train.py
		changing dataset: change dataset by renaming the path to data folder files
			use either 
				1-EMNIST ByClass(model.h5 and model.json)
				2-EMNIST Letters(model1.h5 and model1.h5)

		install libraries tensorflow ,keras ,numpy,mnist, matplotlib

		model is saved in model.json file and weights in model.h5 file
		use these files to predict the characters in an image using the  model
		make necessary changes regarding loading and saving model before running program.

	2-final.py:
		run commands:- python final.py --image images_detect/banner.jpg --east frozen_east_text_detection.pb

		provide image name of which you want the prediction
		images are provided in folder images

		install libraries opencv ,numpy, keras,argparse,matplotlib
		
		program works in 4 steps :
		1-loading the model
		2-preprocessing the image
		3-character detection
		4-character recognition

		download the EMNIST Dataset and model from google drive 


References:
1-https://machinelearningmastery.com/save-load-keras-deep-learning-models/
2-https://www.nist.gov/itl/iad/image-group/emnist-dataset
3-https://arxiv.org/abs/1702.05373v1
4-https://keras.io/datasets/
5-https://www.sciencedirect.com/science/article/pii/S0031320318302590
6-https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
7-https://github.com/ankanbhunia/AttenScriptNetPR

By-Vaibhav Mishra
IIT Jodhpur





