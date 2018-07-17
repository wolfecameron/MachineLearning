"""
This file contains code for classifying an SKlearn dataset using the
random forest algorithm - as well as several visualizations for the
data and result

In this exercise - I worked with the sklearn breast cancer dataset. 
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns





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
		rf = RandomForestClassifier(n_estimators = 25, oob_score=True, n_jobs=-1)
	
		# split the data set into training and testing data
		x_train, x_test, y_train, y_test = train_test_split(data, target)

		# fit the model to the data and find the baseline accuracy
		rf.fit(x_train, y_train)
		accuracies.append(rf.oob_score_)
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


def filter_features(data, bad_indices):
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

	# eliminate above column indices from the data and return new set
	filtered_data = np.delete(data, bad_indices, axis=1)
	return filtered_data


def check_null(df):
	"""This method is used to check all columns of the data frame
	for any null values and check the data types of each of the columns
	in the data frame
	"""

	null_info = df[df.isnull().any(axis=1)].count()
	total_nulls = null_info.sum()
	print("{0} null values were found.".format(str(total_nulls)))
	if(total_nulls > 0):	
		print(null_info)
	print("\n\nShowing all data types:\n\n")
	print(df.dtypes)


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


def vis_correlation_map(df):
	"""This method uses a seaborn heatmap to visualize the correlation
	between features in the dataset
	
	This plot shows that features 2, 3, 20, 22, 23 correlate strongly
	with feature 0 and 12 and 13 with feature 10. These features should
	be removed because they present no new information.	
	"""

	fig, ax = plt.subplots()
	corr = df.corr()
	sns.heatmap(corr, annot=True, cmap='hot')
	plt.show()			

			
def vis_output_distribution(df):
	"""This method creates a bar chart visualization for the
	number of observations classified as either a 1 or a 0 to
	see if the data set is balanced or not.
	
	From this it can be seen that there are a greater amount of 1s
	than 0s.
	"""

	classes = df.loc[:, "classification"]
	classes.value_counts().plot(kind='bar')
	plt.show()


def vis_feature_corr(data, ind_1, ind_2, class_):
	"""Function for visualizing the correlations of two
	features in a 2D plane - colors the plot with the
	classification of the data points.

	Parameters:
	ind_1/2 -- the indices of the features in data frame
	that are being plotted
	"""
	
	df = pd.DataFrame(data)
	plt.scatter(df.iloc[:, ind_1], df.iloc[:, ind_2], c=class_)
	plt.title("Showing Plot of Selected Features")
	plt.xlabel("index {0}".format(ind_1))
	plt.ylabel("index {0}".format(ind_2))
	plt.show()
	

def vis_all_feat_corrs(data, class_):
	"""This method uses the method above to visualize the
	feature correlations of all features in the data set 
	using the above method.
	"""
	
	num_feat = data.shape[1]
	for ind_1 in range(num_feat):
		for ind_2 in range(num_feat):
			print("Showing features {0} and {1} ...".format(
				str(ind_1), str(ind_2)))
			vis_feature_corr(data, ind_1, ind_2, class_)

	
def vis_single_feat(data, class_, ind):
	"""Function for visualizing a single feature with
	a graph being created to show the classification of
	the features based on the value of the feature as well
	as a bar graph of the mean values of the features

	This plot allows you to decide if the feature should be used
	or not - features with very similar plots for 1s and 0s do not
	provide much useful information.
	"""
	
	# create graph of classification and feature values	
	plt.figure(100) # display two plots on separate figures
	df = pd.DataFrame(data)
	feat_vals = df.iloc[:, ind]
	plt.scatter(feat_vals, class_)
	plt.title("Plot of Feature {0}".format(str(ind)))
	plt.xlabel("Feature Value")
	plt.ylabel("Classification")
	
	# create bar graph of mean feature values for each classification
	plt.figure(200)
	plt.title("Mean Values of Feature {0}".format(str(ind)))
	plt.xlabel("Classification")
	plt.ylabel("Mean Feature Value")
	mean_df = pd.concat([df.iloc[:, ind], pd.Series(class_)], axis=1)
	mean_df.columns = ["values", "classif"]	
	mean_df.groupby("classif", as_index=False)["values"].mean().loc[:,"values"].plot(kind='bar')
	
	plt.show()


def vis_all_feat(data, class_):
	"""Method that utilizes the above method to visualize
	all features in a data set by themselves iteratively
	"""
	
	for col_ind in range(data.shape[1]):
		print("Viewing Feature #{0}".format(str(col_ind)))
		vis_single_feat(data, class_, col_ind)
	


if __name__ == "__main__":
	"""Run all code within this main body"""

	# load in data set in form of (data, target)
	data, classif = load_breast_cancer(return_X_y=True)
	data_df = pd.DataFrame(data)
	
	# must check data for nulls and bad data types
	check_null(data_df)

	vis_correlation_map(data_df)
	
	# filter strongly correlated features - can see which ones in correlation map
	data = filter_features(data, [2, 3, 20, 22, 23, 12, 13])
	
	#vis_all_feat(data, classif)
	data = filter_features(data, [1, 2, 6, 7, 9, 10, 14, 15])
	print(data.shape)
	
	informed_rf = run_default_rf(data, classif)[0]
	print(informed_rf.feature_importances_)
	
	# take out features with a very low feature importance
	ind = 0
	low_importance = []
	for x in informed_rf.feature_importances_:
		if (x < .01):
			low_importance.append(ind)
		ind += 1

	data = filter_features(data, low_importance)
	vis_all_feat_corrs(data, classif)
		
	'''
	base_rf = run_default_rf(data, classif)[0]
	#visualize_feature_importances(base_rf)
	IMPORTANCE_THRESH_LIST = [.0005, .001, .002, .005, .008, .1, .12, .15, .18, .2]
	
	# find best threshold and use it to filter features in the data set	 
	IMPORTANCE_THRESH = .008 #test_importance_thresholds(IMPORTANCE_THRESH_LIST, base_rf, data)
	filtered_data = filter_features(base_rf, data, IMPORTANCE_THRESH)
	
	# create full data frame with filtered features
	filtered_df = pd.DataFrame(filtered_data)
	filtered_df.loc[:, "classification"] = pd.Series(classif)
	vis_output_distribution(filtered_df)
	

	#filtered_rf = run_default_rf(filtered_data, classif)[0]
	#visualize_feature_importances(filtered_rf)
	#view_correlation_map(filtered_df)
	'''
