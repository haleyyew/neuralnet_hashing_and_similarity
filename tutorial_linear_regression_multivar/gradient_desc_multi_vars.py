# https://medium.com/@tyreeostevenson/pokemon-stats-and-gradient-descent-for-multiple-variables-c9c077bbf9bd

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def hypothesis(x, theta):
	return np.dot(
			np.transpose(theta),
			x
		)

def gradientDescent(x, y, theta, m, alpha, iterations=1500):
	for iteration in range(iterations):
		for j in range(len(theta)):
			gradient = 0
			for i in range(m):
				gradient += (hypothesis(x[i], theta) - y[i]) * x[i][j]
		gradient *= 1/m
		theta[j] = theta[j] -  (alpha * gradient)
		# print(theta)
	return theta	

def generateZValues(x, theta):
	z_values = []
	for i in range(len(x)):
		z_values.append(hypothesis(x[i], theta))
	return np.asarray(z_values) 

def graph(features, output, theta, figure_name):
	x = []
	y = []
	for feature in features:
		x.append(feature[0])
		y.append(feature[1])
	z = generateZValues(feature, theta)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plt.scatter(x, y, z, c='r', marker='o')

	#Ensure that the next plot doesn't overwrite the first plot
	
	ax = plt.gca()
	ax.hold(True)

	plt.scatter(x, y, output, c='g', marker='d')

	ax.set_xlabel('Total')
	ax.set_ylabel('Special Attack')
	ax.set_zlabel('Catch Rate')
	plt.savefig(figure_name)


def test(test_features, test_output, theta, m):
	error = 0
	pred_output_array = []
	for i in range(m):
		pred_output = hypothesis(test_features[i], theta)
		pred_output_array.append(pred_output)
		error += (pred_output - test_output[i])**2
	print(pred_output_array, test_output)
	return error

if __name__ == '__main__':
	data = pd.read_csv('pokemon_alopez247.csv')

	total = np.asarray(data['Total'])
	special_attack = np.asarray(data['Sp_Atk'])
	catch_rate = np.asarray(data['Catch_Rate'])

	print(total[0:20])
	print(special_attack[0:20])
	print(catch_rate[0:20])

	total = np.array(total[0:20])
	special_attack = np.array(special_attack[0:20])
	catch_rate = np.array(catch_rate[0:20])

	temp = np.asarray([[tot, spec_atk] for tot, spec_atk in zip(total, special_attack)]) # Gets our features
	training_features, test_features = temp[:int(len(temp) * 0.7)], temp[int(len(temp) * 0.3):] # Splits our features in half between training and testing
	temp = np.asarray([rate for rate in catch_rate]) # Gets our output
	training_output, test_output = temp[:int(len(temp) * 0.7)], temp[int(len(temp) *0.3):]  # Splits our outputs in half between training and testing
	theta = np.random.uniform(0.0, 1.0, size=2)
	
	graph(training_features, training_output, theta, 'graphPreFit')

	alpha = 0.0001
	# #print(features[0][0])
	numiter = 700
	model = gradientDescent(training_features, training_output, theta, len(training_output), alpha, numiter)
	print('model', model)

	error = test(test_features, test_output, theta, len(test_features))
	print('error', error)






graph(test_features, test_output, theta, 'graphPostFit')