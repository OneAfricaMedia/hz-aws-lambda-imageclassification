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
import multiprocessing
from multiprocessing import Pipe

runGroupModelWhenScore = float(os.environ.get('RUN_GROUP_MODEL_WHEN_SCORE', os.path.dirname(os.path.abspath(__file__))))

def lambda_handler(event, context):
	if 'ping' in event.keys():
		processes = []
		for i in range(4):
			process = multiprocessing.Process(target=warmLabelImage)
			processes.append(process)
			process.start()
			
		for process in processes:
			process.join()

		print('labelImage Lambda instances warmed')
		return {}
	else:
		client = boto3.client('s3')
		bucket = event['Records'][0]['s3']['bucket']['name']
		key = event['Records'][0]['s3']['object']['key']
		obj = client.get_object(Bucket=bucket, Key=key)
		meta = obj["Metadata"]
		if 'categories' not in meta or ('test' in event.keys() and event['test'] == 'yes'):
			payload = {'bucket': bucket, 'key': key}
			meta['categories'] = json.dumps(imageClassification(payload));
			client.copy_object(Bucket = bucket, Key = key, CopySource = bucket + '/' + key, Metadata = meta, MetadataDirective='REPLACE')
			print('Successfully saved meta data on image')
		else:
			print('Image was previously processed')
							
def imageClassification(payload):
	parentConnections = []
	miscellaneousParentConn, miscellaneousChildConn = Pipe()
	groupsParentConn, groupsChildConn = Pipe()
	parentConnections.append(miscellaneousParentConn)
	parentConnections.append(groupsParentConn)
	predictions = {}

	miscellaneousProcess = multiprocessing.Process(target=labelImage, args=(payload, 'miscellaneous', miscellaneousChildConn))
	groupsProcess = multiprocessing.Process(target=labelImage, args=(payload, 'groups', groupsChildConn))
	miscellaneousProcess.start()
	groupsProcess.start()
	miscellaneousProcess.join()
	groupsProcess.join()
	
	groupsPredictions = []
	for parentConnection in parentConnections:
		result = parentConnection.recv()
		if 'miscellaneous' in result.keys():
			predictions = result['miscellaneous']
		if 'groups' in result.keys():
			groupsPredictions = result['groups']	

	if len(groupsPredictions) > 0:
		processes = []
		groupsParentConnections = []
		for label, confidence in groupsPredictions.iteritems():
			template = "{} - Confidence = {:.2f}%"
			print('Group prediction: ' + template.format(label, confidence))
				
			if confidence >= runGroupModelWhenScore:
				parentConn, childConn = Pipe()
				groupsParentConnections.append(parentConn)
				process = multiprocessing.Process(target=labelImage, args=(payload, label, childConn))
				processes.append(process)
				process.start()
			else:
				print('Skipping ' + label + ' model as it is below the minimum confidence')
		
		for process in processes:
			process.join()
			
		for groupsParentConnection in groupsParentConnections:
			result = groupsParentConnection.recv()
			for label, group_predictions in result.iteritems():
				predictions.update(group_predictions)
							
	return predictions
	
def labelImage(payload, model, conn):
	payload['model'] = model
	lambdaClient = boto3.client('lambda', region_name=boto3.session.Session().region_name)
	start = time.time()
	response = lambdaClient.invoke(FunctionName = 'labelImage', InvocationType = 'RequestResponse', Payload = json.dumps(payload))
	end = time.time()
	print('\nTime taken for labelImage Lamda - Model {}: {:.3f}s\n'.format(model, end-start))
	conn.send({model: json.loads(response['Payload'].read())})
	conn.close()
	
def warmLabelImage():
	lambdaClient = boto3.client('lambda', region_name=boto3.session.Session().region_name)
	payload = {'ping': True}
	lambdaClient.invoke(FunctionName = 'labelImage', InvocationType = 'RequestResponse', Payload = json.dumps(payload))
	