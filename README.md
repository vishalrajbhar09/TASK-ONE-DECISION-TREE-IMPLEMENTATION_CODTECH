INTERN VISHAL RAMKUMAR RAJBHAR

Intern ID:- CT4MESG

Domain:- Machine Learning

Duration:-December 17, 2024, to April 17, 2025

Company:- CODETECH IT SOLUTIONS

Mentor:- Neela Santhosh Kumar




Task Title: Decision Tree Implementation - Build and Visualize a Model for Classification

This repository contains an end-to-end implementation of a Decision Tree classification model using Scikit-learn, applied to the widely-known Iris dataset. The project was undertaken as part of a machine learning task to demonstrate the power of Decision Trees in solving classification problems. The primary goal was to build a model that predicts the species of Iris flowers and present meaningful insights through a variety of visualizations.


The Iris dataset used in this project consists of 150 records, each representing an observation of a flower. The dataset includes four numerical features: sepal length, sepal width, petal length, and petal width, which help classify the flowers into three distinct species: Setosa, Versicolor, and Virginica. This small yet well-balanced dataset is a common starting point for exploring classification algorithms.


Project Workflow:


1.	Data Exploration and Preprocessing:

o	The dataset was loaded using Scikit-learn's built-in load_iris() function and converted to a Pandas DataFrame for easier handling.

o	Key summaries, such as the first few rows, top 10 and bottom 10 records, were displayed to understand the dataset's structure and distribution.

o	No significant preprocessing was required as the dataset was clean and well-structured.

OUTPUT:-01
![image](https://github.com/user-attachments/assets/b7be360e-9801-4753-9d58-60b2b3fad6ba)


2.	Splitting the Dataset:

o	The dataset was split into training (80%) and testing (20%) subsets using Scikit-learn's train_test_split method. This ensured that the model was evaluated on unseen data to avoid overfitting.

OUTPUT:-02
![image](https://github.com/user-attachments/assets/ff7d2add-7314-437e-818b-5a66ac69c9a3)


3.	Model Training and Evaluation:

o	A Decision Tree classifier (DecisionTreeClassifier) was instantiated with a fixed random seed for reproducibility.

o	The model was trained on the training data and evaluated on the test data, achieving a high accuracy score. The exact accuracy was printed as part of the analysis, showcasing the effectiveness of Decision Trees in handling small datasets.

4.	Feature Importance Analysis:

o	The relative importance of each feature in making predictions was calculated and visualized using a bar chart. This revealed that petal length and petal width were the most influential features in determining the species of Iris flowers.

OUTPUT:-03
![image](https://github.com/user-attachments/assets/5c2ab4c1-af98-439c-9cc0-38c742f31bbc)


5.	Decision Tree Visualization:

o	The trained Decision Tree was visualized in a comprehensive diagram using Scikit-learn's plot_tree function. The visualization included feature names, target classes, and color-coded nodes for easy interpretability.

o	This allowed a deeper understanding of how the model makes decisions at each step based on feature thresholds.

OUTPUT :-04
![image](https://github.com/user-attachments/assets/57bc9e54-c7c5-4363-ac63-879f58d4540b)


6.	Exploratory Data Analysis (EDA):

o	A Seaborn pairplot was created to analyze the relationships between features, color-coded by species. This visualization made it easier to observe patterns, such as the distinct separability of the Setosa species and some overlap between Versicolor and Virginica.

7.	Model Rules Export:

o	The decision-making process of the tree was extracted as human-readable rules using Scikit-learn's export_text function. This textual representation provided a concise summary of the tree structure and the conditions used for classification.

OUTPUT:-05
![image](https://github.com/user-attachments/assets/03db6796-8f86-406a-b6b0-19deba151391)


Key Deliverables:


•	High-Accuracy Model: The Decision Tree achieved outstanding accuracy on the test set, highlighting its suitability for this classification task.

•	Insightful Visualizations:

o	A feature importance chart showing the contribution of each feature to the classification task.


o	A Decision Tree diagram to provide an intuitive understanding of the decision-making process.

o	A pairplot for visual exploration of feature relationships and separability between target classes.

•	Human-Readable Rules: A plain-text representation of the tree's logic to ensure transparency and explainability.

FINAL OUTPUT :-
![image](https://github.com/user-attachments/assets/bd0c5e3c-2e16-4631-91cc-6f6e87ec359b)




