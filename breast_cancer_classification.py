"""
This file contains code for classifying an SKlearn dataset using the
random forest algorithm - as well as several visualizations for the
data and result

In this exercise - I worked with the sklearn breast cancer dataset. 
"""

import numpy as np
from sklearn.datasets import load_breast_cancer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# load in data set in form of (data, target)
data, classif = load_breast_cancer(return_X_y=True)


def run_default_rf(data, target, verbose=True):
	"""This method runs random forest with default parameters on 
	the data set that is passed into it. The accuracy of the test
	is displayed in the terminal and the random forest trained object
	is returned to the user to be used for later examination and testing.
	"""
	
	iterations = 15
	accuracies = []
	rf_results = []
	for x in range(iterations):
		# instantiate the Random Forest Classifier
		rf = RandomForestRegressor()
	
		# split the data set into training and testing data
		x_train, x_test, y_train, y_test = train_test_split(data, target)

		# fit the model to the data and find the baseline accuracy
		rf.fit(x_train, y_train)
		results = rf.predict(x_test)
		accuracy = 100 - (np.sum(np.fabs(results - y_test))/y_test.shape[0])*100
		accuracies.append(accuracy)
		rf_results.append(rf)
	
	# find the best random forest result to append
	best_ind = accuracies.index(max(accuracies))
	best_rf = rf_results[best_ind]
	
	# find average accuracy of all tests
	overall_acc = np.mean(np.array(accuracies))

	# print accuracy and return the fitted baseline model
	if(verbose):
		print("Accuracy: {0}%".format(str(overall_acc)))
	return (best_rf, overall_acc)


def visualize_feature_importances(rf):
	"""This function plots all feature importances within the data
	set using matplotlib - x axis represents index in the vector of 
	the feature while the y axis represents the feature importance
	"""
	
	# get feature importances from sklearn and a list of indices for scatter plot
	imp_list = rf.feature_importances_
	indices = np.arange(imp_list.shape[0])
	
	# create plot and label it - display plot
	plt.scatter(indices, imp_list)
	plt.xlabel("Feature Index")
	plt.ylabel("Importance")
	plt.title("Feature Importance Visualization")
	plt.show()


def filter_features(rf, data, importance_threshold):
	"""Uses the feature importances of the baseline
	random forest to eliminate featues from the data 
	that are not useful to the algorithm
	
	Return the data set with unimportant features eliminated	

	Parameters:
	rf -- the random forest from which the importances are 
	being drawn
	data -- the classification data
	importance_threshold -- any importance below this threshold
	will be filtered from the data
	"""
	
	importances = rf.feature_importances_
	
	# get list of indices below the importance threshold
	bad_indices = np.where(importances <= importance_threshold)	

	# eliminate above column indices from the data and return new set
	filtered_data = np.delete(data, bad_indices, axis=1)
	return filtered_data
	

def test_importance_thresholds(imp_thresh_list, rf, data, display=True):
	"""This method tests a list of importance thresholds and determines
	which value performs the best for filtering data. 
	
	The average accuracy is plotted for each importance threshold and 
	the maximum performing threshold is returned to the user.
	"""
	
	iterations = 20 # number of times each threshold value is tested
	
	# go through each possible importance threshold and run random
	# forest a certain number of times to get an average accuracy
	accuracies = []
	for imp in imp_thresh_list:
		new_data = filter_features(rf, data, imp)
		avg_accuracy = 0.0
		# run rf several time to obtain an average accuract value
		for x in range(iterations):
			acc_tmp = run_default_rf(new_data, classif, verbose=False)[1]
			avg_accuracy += acc_tmp
		avg_accuracy /= iterations
		accuracies.append(avg_accuracy)
	
	# display graph of the average accuracies for each of the threshold values
	if(display):
		plt.bar(np.arange(len(accuracies)), accuracies, align='center')
		plt.xticks(np.arange(len(accuracies)), imp_thresh_list)
		plt.ylabel("Average Accuracy")
		plt.xlabel("Importance Threshold")
		plt.show() 
	
	# get the importance threshold that corresponds to the highest average accuracy
	best_ind = accuracies.index(max(accuracies))
	return imp_thresh_list[best_ind]

			
			

if __name__ == "__main__":
	"""Run all code within this main body"""
	base_rf = run_default_rf(data, classif)[0]
	visualize_feature_importances(base_rf)
	IMPORTANCE_THRESH_LIST = [.0005, .001, .002, .005, .008, .1, .12, .15, .18, .2]
	# find best threshold and use it to filter features in the data set	 
	IMPORTANCE_THRESH = test_importance_thresholds([.001, .002, .005, .01, .015], base_rf, data)
	filtered_data = filter_features(base_rf, data, IMPORTANCE_THRESH)
	filtered_rf = run_default_rf(filtered_data, classif)[0]
	visualize_feature_importances(filtered_rf)

