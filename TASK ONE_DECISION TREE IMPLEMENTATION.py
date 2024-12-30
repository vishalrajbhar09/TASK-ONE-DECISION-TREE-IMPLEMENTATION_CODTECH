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

# In[14]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns



# In[15]:


# Loading the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Displaying  the first few rows of the dataset
print("Dataset preview:")
print(data.head())


# In[17]:


#getting the top 10 values
data.head(10)


# In[18]:


# geeting the 10 bottom values
data.tail(10)


# In[19]:


# Spliting  the dataset into training and testing sets
X = data[iris.feature_names]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)



# Train the model on the training data
dt_model.fit(X_train, y_train)



# Evaluate the model on the test data
accuracy = dt_model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")



# Visualize feature importance
feature_importances = dt_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=iris.feature_names, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# In[21]:


# Visualize the Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names.tolist(), filled=True)
plt.title("Decision Tree Visualization ")
plt.show()



# In[22]:


# Export the tree in textual format
tree_rules = export_text(dt_model, feature_names=iris.feature_names)
print("Decision Tree Rules:")
print(tree_rules)


# Pairplot of the dataset to show relationships between features
sns.pairplot(data, hue="target", palette="deep", diag_kind="kde", markers=["o", "s", "D"],
             plot_kws={'alpha': 0.7})
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()


# In[ ]:




