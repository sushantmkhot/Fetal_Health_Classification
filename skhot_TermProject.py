# -*- coding: utf-8 -*-
"""
Sushant Khot
Class: CS 677 - Spring 2
Date: 04/29/2021
Term Project
"""



# ==============================
#          Introduction
# ==============================
"""
Context: I decided to choose this dataset having experienced the birth of my premature kid who is doing great today. 
I wanted to examine and study the dataset to understand what analysis we can do on the fetal health data.

Tasks performed:
•	Examine which features have strong correlation with Class label.
•	Perform Classification Analysis and build models using :
    1. KNN Classifier (find optimal k and re-run to get the final accuracy)
    2. Logistic Regression classifier
    3. Naive Bayesian
    4. Decision Tree
    5. Random Forest
    
•	I calculated and discussed Performance Metrics for these classifiers by preparing Confusion Matrix to look at how our prediction models perform.
•	I have Split Data into Training and Testing to verify the models built (50/50).
•	I also tried to visualize the dataset and look for features that can help for classification, outliers, correlation etc.

Some Questions that I tried to answer at the end of Project and analysis:
1. What features in different Visualizations show us trends and help classify a fetus.
2. Compare and build insights on different features and their correlation with the Class Label.
3. Which Model best predicts the Health of a fetus?
4. Compare all classifiers listed above and discuss our findings using confusion matrix.


Data: This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetricians into 3 classes:
•	Normal
•	Suspect
•	Pathological

I have combined "Suspect" and "Pathological" classes into one class called "Abnormal" and Normal will stay as "Normal".
"""



# ==================================
#          Import Libraries
# ==================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn . ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


# =======================================================================
#          Importing the Fetal Heath dataset in pandas dataframe
# =======================================================================

here = os.path.abspath(__file__) # Relative Path code
input_dir = os.path.abspath(os.path.join(here, os.pardir))
df_fetal_health = os.path.join(input_dir, 'fetal_health.csv')

try:
    df_fetal_health = pd.read_csv(df_fetal_health)
    
except Exception as e:
    print(e)
    print('failed to read Fetal health data into Data Frame')
    
    
# We will validate the import of data in our pandas df
print('\n')
print(df_fetal_health.head())


# We will check the info of the data to check if there are any null values.
print('\n')
print(df_fetal_health.info())


# We see that there are 2126 total rows in the data set and none of them are of null value. 
# Hence our dataset for fetal health is pretty clean and does not need any null value fixes.

# We will add the "class" labels to identify the 3 types of fetal health: 1 = "Normal", 2 = "Suspect" and 3 = "Pathological".
# We will combine the labels in 2 groups, "Normal" - these labels are assigned and "Abnormal" - everything else.
df_fetal_health["class"] = df_fetal_health.apply(lambda x: 1 if x["fetal_health"] == 1 else 0, axis = 1)


# Count Plot of Class Labels:
# We will look the count plot to understand how our dataset is distributed in terms of each class value.
sns.countplot(data= df_fetal_health, x ="class")
plt.title('Count Plot of Class Labels')
plt.show()


df_fetal_health[df_fetal_health["class"] == "Normal"].shape[0]
df_fetal_health[df_fetal_health["class"] == "Abnormal"].shape[0]
# It seems our dataset has 1655 "Normal" class data and 471 "Abnormal" class data.



# ================================================
#          Correlation Matrix of features
# ================================================

corr_mat= df_fetal_health.corr()
sorted_corr_mat = corr_mat["fetal_health"].sort_values(ascending=False)
plt.figure(figsize=(15,15))
sns.heatmap(corr_mat, cmap= 'coolwarm', annot=True)
plt.title('Correlation Between features of Fetal Health dataset')
plt.show()


# This shows top 4 features "Prolonged Deceleration", "Abnormal Short Term Variability", "Percentage of time with abnormal long term variability" and "Accelerations" are highly correlated to "Fetal Health", in that order.
# "Histogram number of zeros", "Histogram number of peaks", "Histogram Max" and "Light Decelerations" are least correlated to "Fetal Health", in that order.



# ===============================================================
#          Feature Selection using pearson's correlation
# ===============================================================

# We will use pearson's correlation feature selection method to select the top 10 features for our analysis.
X = df_fetal_health.drop(["fetal_health", "class"], axis = 1).values
Y = df_fetal_health["class"].values

# define feature selection
fs = SelectKBest(score_func = f_regression, k = 10)
# apply feature selection
X_selected = fs.fit_transform(X, Y)

mask = fs.get_support() #list of booleans
new_features = [] # The list of your K best features

feature_names = list(df_fetal_health.drop(["fetal_health", "class"], axis = 1).columns.values)

for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

X_selected = pd.DataFrame(X_selected, columns = new_features)


# The top 10 features selected are as follows:
# baseline value
# accelerations
# uterine_contractions
# prolongued_decelerations
# abnormal_short_term_variability
# mean_value_of_short_term_variability
# percentage_of_time_with_abnormal_long_term_variability
# mean_value_of_long_term_variability
# histogram_width
# histogram_min


# Boxplot for outliers check:
# We will check if there are any outliers in the dataset using box plots.
plt.figure(figsize=(20,15))
sns.boxplot(data = X_selected).set_title('Boxplot of top 10 correlated features')
plt.xticks(rotation=90)
plt.show()


# We do see many features have outliers specifically "Percentage of time with abnormal long term variability", "Mean value of long term variability" and "Mean value of short term variability".
# However since its a medical dataset, the outcome of the CTG report is unlikely to have any data entry.
# We will hence not remove any outliers and continue our analysis on the dataset.



# Regression Plots:
# Let us look at the regression plots for the top 4 most correlated features with fetal_health vs fetal_movement attribute:

# 1. "Prolonged Deceleration" vs. "fetal_movement"
sns.lmplot(data = df_fetal_health, x="prolongued_decelerations", y="fetal_movement", hue="class",legend_out=False)
plt.title('"Prolonged Deceleration" vs. "fetal movement"')
plt.show()

# 2. "Abnormal Short Term Variability" vs. "fetal_movement"
sns.lmplot(data = df_fetal_health, x="abnormal_short_term_variability", y="fetal_movement", hue="class",legend_out=False)
plt.title('"Abnormal Short Term Variability" vs. "fetal movement"')
plt.show()

# 3. "Percentage of time with abnormal long term variability" vs. "fetal_movement"
sns.lmplot(data = df_fetal_health, x="percentage_of_time_with_abnormal_long_term_variability", y="fetal_movement", hue="class",legend_out=False)
plt.title('"Percentage of time with abnormal long term variability" vs. "fetal movement"')
plt.show()

# 4. "accelerations" vs. "fetal_movement"
sns.lmplot(data = df_fetal_health, x="accelerations", y="fetal_movement", hue="class",legend_out=False)
plt.title('"accelerations" vs. "fetal movement"')
plt.show()


# We confirm the correlation  behavior from these plots. Also, we can see there are some outliers like we found in our boxplots.



# ====================================================
#          Splitting dataset 50/50 train/test
# ====================================================

# We will be splitting the dataset 50/50 using train_test_split. This way we will train 50% of the data and test the remaining 50% on it.
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size = 0.5, random_state = 21)



# ==========================================
#          Common Tasks / Functions
# ==========================================

# I am assuming Positive event as Fetal Health Class = “Normal” or "1" and Negative Event as Fetal Health Class = “Abnormal” or "0".
# The "calculate_statistics" function is used to calculate the confusion matrix statistics.
# It takes the True values / array and predicted value/ array as input parameters to calculate the confusion metrics statstics.
# It returns a dictionary with the respective statistics names as key and their values.
def calculate_statistics(true_value, predicted_value):
    stats_dict_1 = {"TP":0, "FN":0, "FP":0, "TN":0, "TNR":0, "TPR":0}
    conf_mat = confusion_matrix(true_value, predicted_value, labels=[1, 0])
    tpr = recall_score(true_value, predicted_value, pos_label= 1)
    accuracy = accuracy_score(true_value, predicted_value)
    
    stats_arr = conf_mat.ravel()
    stats_dict_1["TP"] = stats_arr[0]   
    stats_dict_1["FN"] = stats_arr[1]
    stats_dict_1["FP"] = stats_arr[2]
    stats_dict_1["TN"] = stats_arr[3]
    stats_dict_1["accuracy"] = accuracy * 100
    stats_dict_1["TNR"] = (stats_arr[3] / (stats_arr[3] + stats_arr[2])) * 100
    stats_dict_1["TPR"] = tpr * 100
    
    return stats_dict_1



# The "print_stats_table" function is used to print the confusion matrix statistics.
# As input parameter, it takes the dictionary with the respective statistics names as key and their values from "calculate_statistics" function and prints it in a tabular format.
def print_stats_table(stats_dict_print, model_nm):
    # Print the values in the Table
    print("|" + model_nm.center(21) +
          "|" + str(round(stats_dict_print["TP"], 2)).center(6) +
          "|" + str(round(stats_dict_print["FP"], 2)).center(6) +
          "|" + str(round(stats_dict_print["TN"], 2)).center(6) +
          "|" + str(round(stats_dict_print["FN"], 2)).center(6) +
          "|" + str(round(stats_dict_print["accuracy"], 2)).center(13) +
          "|" + str(round(stats_dict_print["TPR"], 2)).center(8) +
          "|" + str(round(stats_dict_print["TNR"], 2)).center(8) + "|")
    print("===================================================================================")



# The "calculate_classifier_accuracy" function is used to calculate and return the accuracy of a classifier model.
# It takes in the training and testing arrays and classifier name as input parameters and returns % accuracy of the model.
# It will work for "Naive Bayesian", "Decision Tree", and "Random Forest" with combination of N (number of estimators) and d (maximum depth).
# The "summarize_results" parameter will decide whether we are running this for individual classifiers or creating a summary table. 
def calculate_classifier_accuracy(X_train, Y_train, X_test, Y_test, classifier_nm, summarize_results = "No"):    
    
    # "KNN (k=3)"
    if classifier_nm == "KNN (k = 3)":
        classifier = KNeighborsClassifier(n_neighbors = 3)
        classifier = classifier.fit(X_train, Y_train)
    
    # "Logistic Regression"
    elif classifier_nm == "Logistic Regression":
        classifier = LogisticRegression(solver='lbfgs', max_iter=3000)
        classifier = classifier.fit(X_train, Y_train)
        
    # "Naive Bayesian"
    elif classifier_nm == "Naive Bayesian":
        classifier = GaussianNB().fit(X_train, Y_train)
    
    # "Decision Tree"
    elif classifier_nm == "Decision Tree":
        classifier = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 21)
        classifier = classifier.fit(X_train, Y_train)
        
    # "Random Forest" with combination of N (number of estimators) and d (maximum depth).
    elif classifier_nm == "Random Forest":
        classifier = RandomForestClassifier(n_estimators = 8, max_depth = 5, criterion ='entropy', random_state = 21)
        classifier = classifier.fit(X_train, Y_train)
        
    
    prediction = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, prediction)
    accuracy = round(accuracy * 100, 2)
    
    # We also claculate the confusion matrix statistics using "calculate_statistics" function
    stats_dict_2 = calculate_statistics(Y_test, prediction)
    
    if summarize_results == "No":
        # Returns the accuracy of the model depending on which classifier name is requested for in the input parameters.
        print("\n")
        print('The Accuracy using {clf} is = {acc}%'.format(clf = classifier_nm if classifier_nm != "Random Forest" else "Random Forest with best N = 8 and d = 5", acc = accuracy))
            
        # We will print the confusion matrix statistics table using "print_stats_table" function
        # Printing the header of the Confusion Matrix table.
        print("\n")
        print("===================================================================================")
        print("|        Model        |  TP  |  FP  |  TN  |  FN  | accuracy(%) | TPR(%) | TNR(%) |")
        print("===================================================================================")
    
        print_stats_table(stats_dict_2, classifier_nm)
    
    else:
              
        print_stats_table(stats_dict_2, classifier_nm)
   


# =======================================================
# We will now run some classifier models listed below on the 50/50 train and test data to classify the labels and predict their accuracy.
# Models that we will run are as follows:
# 1. KNN Classifier (find optimal k and re-run to get the final accuracy)
# 2. Logistic Regression classifier
# 3. Naive Bayesian
# 4. Decision Tree
# 5. Random Forest
# =======================================================



# =======================================================
#          k-NN classifier using sklearn library
# =======================================================
"""
We will take k = 3, 5, 7, 9, 11. Use the same Xtrain and Xtest as before. 
For each k, we will train the k-NN classifier on Xtrain and compute its accuracy for Xtest
"""

# We will scale the training and testing data and stored scaled data in X_train_scaled and X_test_scaled respectively
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)

error_rate = []
knn_accuracy = []
print("\n\n")

# We will now run the KNN classifier for k = 3, 5, 7, 9, 11 and print out the Accuracies for each k to find the optimal k*.
cntr = 0
for k in range (3, 13, 2):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train_scaled, Y_train)
    pred_k = knn_classifier.predict(X_test_scaled)
    error_rate.append(np.mean(pred_k != Y_test))
    knn_accuracy.append(round(knn_classifier.score(X_test_scaled, Y_test) * 100, 2))
    print('The Accuracy for KNN classifier with k = {k} = {accuracy}%'.format(k = k, accuracy = knn_accuracy[cntr]))
    cntr += 1



"""
Plot a graph showing the accuracy. 
On x axis we will plot k and on y-axis we will plot accuracy. 
We will then find the optimal value k* of k.
"""

# Accuracy graph
figure, ax = plt.subplots()
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer = True))
plt.plot(range(3, 13, 2), knn_accuracy, color ='blue', linestyle ='dashed', marker ='o', markerfacecolor ='black', markersize = 10)
plt.title('Accuracy vs. k for Fetal Health Data')
plt.xlabel('number of neighbors : k')
plt.ylabel('Accuracy')
plt.show()

print("\n")
print("The optimal value k* is k = 3")



"""
We will use the optimal value k* to compute performance measures and summarize them in the table.
"""

# Calulate the confusion matrix statistics for optimal k = 3
calculate_classifier_accuracy(X_train_scaled, Y_train, X_test_scaled, Y_test, "KNN (k = 3)")



# ======================================================================
#          Logistic Regression classifier using sklearn library
# ======================================================================
"""
We will use the same Xtrain and Xtest as before and train our logistic regression classifier on Xtrain and compute its accuracy for Xtest
"""

calculate_classifier_accuracy(X_train, Y_train, X_test, Y_test, "Logistic Regression")



# ================================
#          Naive Bayesian
# ================================
"""
We will run the classifier on our 50/50 dataset, train NB on Xtrain and predict class labels in Xtest.
We will find out the accuracy and compute the confusion matrix
"""
    
# Accuracy and Confusion Matrix using Naive Bayes
calculate_classifier_accuracy(X_train, Y_train, X_test, Y_test, "Naive Bayesian")



# ================================
#          Decision Tree
# ================================
"""
We will run the classifier on our 50/50 dataset, train Decision tree on Xtrain and predict class labels in Xtest.
We will find out the accuracy and compute the confusion matrix
"""

# Accuracy and Confusion Matrix using Decision tree
calculate_classifier_accuracy(X_train, Y_train, X_test, Y_test, "Decision Tree")



# ================================
#          Random Forest
# ================================
"""
We will take N = 1,...,10 and d = 1, 2,...,5. 
For each value of N and d, we will split our data into Xtrain and Xtest, 
We will construct a random tree classifier (use "entropy" as splitting criteria - this is the default) 
Finally, we will train our classifier on Xtrain and compute the error rate for Xtest.
"""

# We will create a dictionary "error_rates_dict_RF", that will store the respective 10 error rates for each estimator. In turn these 10 error rates will be stored with max-depth value key name.
# This dictionary setup will help us to plot the data accordingly.
error_rates_dict_RF = {"max-depth = 1":0, "max-depth = 2":0, "max-depth = 3":0, "max-depth = 4":0, "max-depth = 5":0}

# Loop through the 5 max-depth d values
for d in range(1, 6):
    error_rates_RF = []
    
    # Loop through the 10 estimators N values
    for n in range(1, 11):
        model = RandomForestClassifier(n_estimators = n , max_depth = d, criterion ='entropy', random_state = 21)
        model = model.fit(X_train, Y_train)
        prediction_RF = model.predict(X_test)
        error_rate = np.mean(prediction_RF != Y_test)
        error_rates_RF.append(error_rate)
    
    error_rates_dict_RF["max-depth = "+str(d)] = error_rates_RF 



"""
We will now plot the error rates and find the best combination of N and d.
We will then calculate the accuracy for the best combination of N and k.
We will then compute the confusion matrix using the best combination of N and d.
"""

# Plot of Error Rates vs. Number of Estimators (Random Forest for Fetal Health Data)
for d in range(1, 6):
    plt.plot(range(1, 11), error_rates_dict_RF["max-depth = "+str(d)], marker='o', label = "max-depth = "+str(d))

plt.xlabel("Number of estimators")
plt.ylabel("Error Rate")
plt.title("Random Forest for Fetal Health Data")
plt.legend(bbox_to_anchor=(1.2, 0.5), loc = "center")
plt.show()


# From the plot we can see that our best number of estimators N = 8 and max-depth d = 5
# Accuracy and Confusion Matrix using Random Forest with best N = 8 and d = 5
calculate_classifier_accuracy(X_train, Y_train, X_test, Y_test,"Random Forest")



# ==========================================
#          Confusion Matrix Summary
# ==========================================
"""
We will now summarize our results for all the classifiers in a table and discuss our findings.
"""

print("\n")
print("============================= Summary of Results ==================================")
print("|        Model        |  TP  |  FP  |  TN  |  FN  | accuracy(%) | TPR(%) | TNR(%) |")
print("===================================================================================")

for model_nm in ["KNN (k = 3)", "Logistic Regression", "Naive Bayesian", "Decision Tree", "Random Forest"]:
    if model_nm == "KNN (k = 3)":
        calculate_classifier_accuracy(X_train_scaled, Y_train, X_test_scaled, Y_test, model_nm, "Yes")
    else:
        calculate_classifier_accuracy(X_train, Y_train, X_test, Y_test, model_nm, "Yes")


# Summary of Confusion Matrix results:
"""
Here, Positive event is "Normal"" / class 1 of Fetal health. Negative event is "Abnormal" or class 0 of Fetal Health.

KNN (k = 3) classifier gave us the best overall accuracy of 92% which is close to Random Forest and Decision tree. 
It also predicted the most True Negatives among other classifiers closely followed by Decision Tree classifier.
It's True Negative Rate TNR = 78.63% which is close to TNR of Decision Tree = 77.78%

In Terms of correctly predicting the positive event, we see that Random Forest has predicted the most true Positives with TPR = 97.59 %, followed by KNN (k = 3). 
Decision Tree and Logistic regression predicted True Positives with same accuracy.  

Logistic Regression classifier has predicted the most False Positives which means that it has classified many "Abnormal" class fetal health as "Normal". However Logistic regression did a good job at predicting the "Normal" class Fetal Health.

Naive Bayesian classifier on the other hand has predicted the most False negatives which means it predicted many "Normal" class Fetal health as "Abnormal". The overall accuracy of Naive Bayesian classifier is the lowest with 83.25%.
Logistic regression classifier also has a low overall accuracy of 83.54%. 

Overall based on Accuracy, KNN (k = 3), Random Forest and Decision tree seemed to have performed well.
Random Forest is able to classify more "Normal" class fetal health correctly.
KNN (k = 3) and Decision Tree have predicted more "Abnormal" class Fetal health correctly.
"""



# ============================
#          Conclusion
# ============================
"""
We started with analyzing our Fetal health dataset and it was pretty clean dataset to begin with and we did not have to do any pre-processing activities.
Some features did have outliers, but we decided not to remove them as the medical data entry errors would be minimum and we went ahead without removing any outliers.

We created various visualizations to help support our analysis and gather more insights on the features present in the dataset.
For e.g. Count plot gave us an idea on distribution of our dataset based on class label.
Correlation Matrix provided us the top most correlated and non-correlated features with Fetal Health
Box plots provided us with insights on data distribution within each feature.
Regression plots provided us with correlation and distribution of features around the 2 class labels "Normal" and "Abnormal" fetal health and fetal movement data.

We split our dataset 50/50 into training and testing and ran through multiple classifiers on the data.
We documented the statistics in terms of classifier Accuracy to correctly classify the Fetal health and discussed the confusion matrix.
Finally based on the confusion matrix we concluded that based on Overall Accuracy, KNN (k = 3), Random Forest and Decision tree seemed to have performed well.
Random Forest is able to classify more "Normal" class fetal health correctly.
KNN (k = 3) and Decision Tree have predicted more "Abnormal" class Fetal health correctly.
"""
