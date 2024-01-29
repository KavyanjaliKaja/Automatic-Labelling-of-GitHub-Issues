# Databricks notebook source
# MAGIC %md
# MAGIC <h3>Loading the json file of final dataset from DBFS.

# COMMAND ----------

import pandas as pd

# Define the DBFS path
dbfs_path = "/FileStore/tables/FinalData/Final.json"

# Read the JSON file into a Spark DataFrame
df_spark = spark.read.json(dbfs_path)

# Convert the Spark DataFrame to a Pandas DataFrame
final_df = df_spark.toPandas()

# Display the Pandas DataFrame
final_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h1>Implementing Machine Learning Models

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>1. Multinomial Naive Bayes Classifier

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Handle NaN values in the data set
final_df.fillna('', inplace=True)

# Split the data into train and test sets
X = final_df['issues'] + ' ' + final_df['title'] + ' ' + final_df['body']
Y = final_df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create and train the Multinomial Naive Bayes model
mnb_model = MultinomialNB()
mnb_model.fit(X_train, Y_train)

# Predict on the test data
Y_pred = mnb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print("MNB Model Accuracy:", accuracy)

print(confusion_matrix(Y_test, Y_pred))

cm = confusion_matrix(Y_test, Y_pred)
cm_df = pd.DataFrame(cm,
                     index=['BUG', 'FEATURE', 'QUESTION'],
                     columns=['BUG', 'FEATURE', 'QUESTION'])

# Plotting the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# Allow the user to input title and body for predidction for 3 times
for i in range(3):
    # Ask for user input for title and body
    input_title = input("Enter the title: ")
    input_body = input("Enter the body: ")
    input_text = input_title + ' ' + input_body

    # Convert user input to numerical features using the same vectorizer
    input_text = vectorizer.transform([input_text])

    # Predict on the user input using MNB model
    mnb_predicted_label = mnb_model.predict(input_text)[0]

    print("MNB Predicted Label:", mnb_predicted_label)
    print("="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>2. Support Vector machine Classifier

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Handle NaN values in the 'issues', 'title' and 'body' columns
final_df.fillna('', inplace=True)

# Split the data into train and test sets
X = final_df['issues'] + ' ' +final_df['title'] + ' ' + final_df['body']
Y = final_df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create and train the Support Vector Machine model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, Y_train)

# Predict on the test data
Y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print("SVM Model Accuracy:", accuracy)

print(confusion_matrix(Y_test, Y_pred))
# print((Y_test=="bug").sum())

cm=confusion_matrix(Y_test, Y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['BUG','FEATURE','QUESTION'], 
                     columns = ['BUG','FEATURE','QUESTION'])

# Plotting the confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

# Allow the user to input title and body for 3 times
for i in range(3):
    # Ask for user input for title and body
    input_title = input("Enter the title: ")
    input_body = input("Enter the body: ")
    input_text = input_title + ' ' + input_body

    # Convert user input to numerical features using the same vectorizer
    input_text = vectorizer.transform([input_text])

    # Predict on the user input using SVM model
    svm_predicted_label = svm_model.predict(input_text)[0]

    print("SVM Predicted Label:", svm_predicted_label)
    print("="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>3. Random Forest Classifier

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Handle NaN values in the 'issues', 'title' and 'body' columns
final_df.fillna('', inplace=True)

# Split the data into train and test sets
X = final_df['issues'] + ' ' +final_df['title'] + ' ' + final_df['body']
Y = final_df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Predict on the test data
Y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print("RF Model Accuracy:", accuracy)

print(confusion_matrix(Y_test, Y_pred))
# print((Y_test=="bug").sum())

cm=confusion_matrix(Y_test, Y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['BUG','FEATURE','QUESTION'], 
                     columns = ['BUG','FEATURE','QUESTION'])

# Plotting the confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
Fix existing custom operators' templated fields that include logic checksFix existing custom operators' templated fields that include logic checksplt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

# Allow the user to input title and body for prediction for 3 times
for i in range(3):
    # Ask for user input for title and body
    input_title = input("Enter the title: ")
    input_body = input("Enter the body: ")
    input_text = input_title + ' ' + input_body

    # Convert user input to numerical features using the same vectorizer
    input_text = vectorizer.transform([input_text])

    # Predict on the user input using RF model
    rf_predicted_label = rf_model.predict(input_text)[0]

    print("RF Predicted Label:", rf_predicted_label)
    print("="*50)
