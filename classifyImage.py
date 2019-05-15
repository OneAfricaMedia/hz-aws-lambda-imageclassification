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
modelPrimaryFilename = os.environ.get('MODEL_PRIMARY_FILENAME', os.path.dirname(os.path.abspath(__file__)))
modelPrimaryLoadPath =  os.path.join(os.sep, 'tmp', 'model_primary.pb')
modelSecondaryFilename = os.environ.get('MODEL_SECONDARY_FILENAME', os.path.dirname(os.path.abspath(__file__)))
modelSecondaryLoadPath =  os.path.join(os.sep, 'tmp', 'model_secondary.pb')
labelsPrimaryFilename = os.environ.get('LABELS_PRIMARY_FILENAME', os.path.dirname(os.path.abspath(__file__)))
labelsPrimaryLoadPath =  os.path.join(os.sep, 'tmp', 'labels_primary.txt')
labelsSecondaryFilename = os.environ.get('LABELS_SECONDARY_FILENAME', os.path.dirname(os.path.abspath(__file__)))
labelsSecondaryLoadPath =  os.path.join(os.sep, 'tmp', 'labels_secondary.txt')
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
print('Downloading Model from S3...')
s3 = boto3.resource('s3')
s3.Bucket(modelBucketName).download_file(modelPrimaryFilename, modelPrimaryLoadPath)
s3.Bucket(modelBucketName).download_file(labelsPrimaryFilename, labelsPrimaryLoadPath)
s3.Bucket(modelBucketName).download_file(modelSecondaryFilename, modelSecondaryLoadPath)
s3.Bucket(modelBucketName).download_file(labelsSecondaryFilename, labelsSecondaryLoadPath)

primaryModel = loadGraph(modelPrimaryLoadPath)
primaryLabels = loadLabels(labelsPrimaryLoadPath)
secondaryModel = loadGraph(modelSecondaryLoadPath)
secondaryLabels = loadLabels(labelsSecondaryLoadPath)
    
def lambda_handler(event, context):
	#api handling
	if 'queryStringParameters' in event:
		queryStringParameters = event['queryStringParameters']
		url = queryStringParameters['url']
		path = '/tmp/{}'.format(uuid.uuid4())
		with open(path,'wb') as f:
			f.write(urllib2.urlopen(url).read())
			f.close()
	
		data = {
			"categories": imageClassification(path),
		}
	
		return {
			"statusCode": 200,
			"body": json.dumps(data)
		}
	else:
	#s3 event
		client = boto3.client('s3')
		bucket = event['Records'][0]['s3']['bucket']['name']
		key = event['Records'][0]['s3']['object']['key']
		path = '/tmp/{}'.format(uuid.uuid4())
		client.download_file(bucket, key, path)
		obj = client.get_object(Bucket=bucket, Key=key)
		meta = obj["Metadata"]
		if 'categories' not in meta:
			meta['categories'] = json.dumps(imageClassification(path));
			client.copy_object(Bucket = bucket, Key = key, CopySource = bucket + '/' + key, Metadata = meta, MetadataDirective='REPLACE')
			print('Successfully saved meta data on image')
		else:
			print('Image was previously processed')
			
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
			predictions[labels[i].replace(' ', '-')] = round(results[i] * 100, 2);
      			
	return predictions
				
def imageClassification(fileName):
	primaryPredictions = makePredictions(fileName, primaryModel, primaryLabels)
	
	if len(primaryPredictions) == 0:
		print('No predictions above the min score specified for primary model.')
		
	secondaryPredictions = makePredictions(fileName, secondaryModel, secondaryLabels)	
	if len(secondaryPredictions) == 0:
		print('No predictions above the min score specified for secondary model.')
		
	predictions = primaryPredictions
	predictions.update(secondaryPredictions)
			
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