import numpy as np
import boto3
import sys
import shutil
import json
import os
import sys
import urllib2
import uuid
import time
import tensorflow as tf

modelBucketName = os.environ.get('MODEL_BUCKET_NAME', os.path.dirname(os.path.abspath(__file__)))
modelDirectoryName = os.environ.get('MODEL_DIRECTORY_NAME', os.path.dirname(os.path.abspath(__file__)))
inputHeight = int(os.environ.get('INPUT_HEIGHT', os.path.dirname(os.path.abspath(__file__))))
inputWidth = int(os.environ.get('INPUT_WIDTH', os.path.dirname(os.path.abspath(__file__))))
inputMean = int(os.environ.get('INPUT_MEAN', os.path.dirname(os.path.abspath(__file__))))
inputStd = int(os.environ.get('INPUT_STD', os.path.dirname(os.path.abspath(__file__))))
inputName = "import/" + os.environ.get('INPUT_LAYER', os.path.dirname(os.path.abspath(__file__)))
outputName = "import/" + os.environ.get('OUTPUT_LAYER', os.path.dirname(os.path.abspath(__file__)))
minScore = float(os.environ.get('MIN_SCORE', os.path.dirname(os.path.abspath(__file__))))

def loadGraph(modelFile):
	graph = tf.Graph()
	graphDef = tf.GraphDef()

	with open(modelFile, "rb") as f:
		graphDef.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graphDef)

	return graph
	
def loadLabels(labelFile):
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(labelFile).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

#Tensorflow model gets loaded into memory only on container startup
print('Downloading Models from S3...')
s3 = boto3.resource('s3')
bucket = s3.Bucket(modelBucketName);

labels = {}
models = {}
for file in bucket.objects.all():
	if file.key.startswith(modelDirectoryName + 'graph_') and file.key.endswith('.pb'):
		model = file.key.replace(modelDirectoryName + 'graph_', '').replace('.pb', '')
		loadPath = os.path.join(os.sep, 'tmp', 'graph_' + model + '.pb')
		bucket.download_file(file.key, loadPath)
		models[model] = loadGraph(loadPath)
		os.remove(loadPath)
	if file.key.startswith(modelDirectoryName + 'labels_') and file.key.endswith('.txt'):
		model = file.key.replace(modelDirectoryName + 'labels_', '').replace('.txt', '')
		loadPath = os.path.join(os.sep, 'tmp', 'labels_' + model + '.pb')
		bucket.download_file(file.key, loadPath)
		labels[model] = loadLabels(loadPath)
		os.remove(loadPath)

def lambda_handler(event, context):
	if 'ping' in event.keys():
		print('Lambda warmed')
		return {}
	else:
		client = boto3.client('s3')
		bucket = event['bucket']
		key = event['key']
		model = event['model']
		path = '/tmp/{}'.format(uuid.uuid4())
		client.download_file(bucket, key, path)

		return makePredictions(path, models[model], labels[model])
				
def makePredictions(fileName, model, labels):
	t = readTensorFromImageFile(fileName, inputHeight = inputHeight, inputWidth = inputWidth, inputMean = inputMean, inputStd = inputStd)
	
	inputOperation = model.get_operation_by_name(inputName)
	outputOperation = model.get_operation_by_name(outputName)

	with tf.Session(graph=model) as sess:
		start = time.time()
		results = sess.run(outputOperation.outputs[0], {inputOperation.outputs[0]: t})
		end=time.time()
		
	results = np.squeeze(results)

	top = results.argsort()[-5:][::-1]
	print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
  
	predictions = {}
	for i in top:
		if results[i] >= minScore:
			predictions[labels[i].replace(' ', '-')] = round(results[i] * 100, 2)
					
	return predictions
				

def readTensorFromImageFile(fileName, inputHeight=299, inputWidth=299, inputMean=0, inputStd=255):
	inputName = "file_reader"
	outputName = "normalized"
	fileReader = tf.read_file(fileName, inputName)
	if fileName.endswith(".png"):
		imageReader = tf.image.decode_png(fileReader, channels = 3, name = 'png_reader')
	elif fileName.endswith(".gif"):
		imageReader = tf.squeeze(tf.image.decode_gif(fileReader, name = 'gif_reader'))
	elif fileName.endswith(".bmp"):
		imageReader = tf.image.decode_bmp(fileReader, name='bmp_reader')
	else:
		imageReader = tf.image.decode_jpeg(fileReader, channels = 3,name = 'jpeg_reader')
	floatCaster = tf.cast(imageReader, tf.float32)
	dimsExpander = tf.expand_dims(floatCaster, 0);
	resized = tf.image.resize_bilinear(dimsExpander, [inputHeight, inputWidth])
	normalized = tf.divide(tf.subtract(resized, [inputMean]), [inputStd])
	sess = tf.Session()
	result = sess.run(normalized)

	return result