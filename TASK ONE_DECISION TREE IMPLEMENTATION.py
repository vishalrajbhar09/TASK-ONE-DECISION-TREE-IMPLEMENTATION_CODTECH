#!/usr/bin/env python
# coding: utf-8

# # INTERN:-VISHAL RAMKUMAR RAJBHAR
# 
# Intern ID:- CT4MESG
# 
# Domain:- Machine Learning
# 
# Duration:-December 17, 2024, to April 17, 2025
# 
# Company:- CODETECH IT SOLUTIONS
# 
# Mentor:- Neela Santhosh Kumar

# # TASK ONE:DECISION TREE IMPLEMENTATION
# 
# BUILD AND VISUALIZE A DECISION TREE MODEL USING SCIKIT-LEARN TO CLASSIFY OR PREDICT OUTCOMES ON A CHOSEN DATASET.
# DELIVERABLE: A NOTEBOOK WITH MODEL VISUALIZATION AND ANALYSIS.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report

# Loading dataset (You can also replace this with your own dataset from(kaggle, weka etc))
data = load_iris()
X, y = data.data, data.target

# Converting the  dataset into a DataFrame for preview
iris_df = pd.DataFrame(X, columns=data.feature_names)
iris_df['target'] = y

# Dataset in preview also with the dataset along
print("Dataset preview:")
print(iris_df.head())

# Splited the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Predicted on the test data
y_pred = clf.predict(X_test)

# Evaluated the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names.tolist(), filled=True)
plt.show()

# Print Decision Tree Rules
print(export_text(clf, feature_names=data.feature_names))
