# Databricks notebook source
# MAGIC %md
# MAGIC <h3>Imported required functions from pyspark.sql library

# COMMAND ----------

import pyspark
import json
from pyspark.sql.functions import col, count
from pyspark.sql.types import StructType
from pyspark.sql.functions import explode

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Reading combined JSON file of GHArchive datasets(2015-01-01-15 & 2015-01-01-16) into a dataframe

# COMMAND ----------

# THIS PARTICULAR Cmd IS USED TO IMPORT THE DIFFERENT JSON FILES AVAILABLE IN THE SPECIFIED DIRECTORY, COMBINE THEM AND LOAD THEM AS A SINGLE SPARK DATAFRAME IN DATABRICKS.
 
# Path
container_path = "/FileStore/JSON/" 
 
# List all files available in the path
files = dbutils.fs.ls(container_path)
 
# Extracting the JSON file names
json_files = [file.name for file in files if file.name.endswith(".json")]
 
# Loading the JSON files into a list of DataFrames
dfs = []
for json_file in json_files:
    file_path = container_path + json_file
    df = spark.read.json(file_path)
    dfs.append(df)
 
# Combining the DataFrames
combined_df = dfs[0]
for df in dfs[1:]:
    df1 = combined_df.union(df)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Displaying the combined dataset and its schema

# COMMAND ----------

df1.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Finding no. of rows and no. of columns in the combined dataset

# COMMAND ----------

num_rows = df1.count()
num_columns = len(df1.columns)
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>STEPS</h3><br>
# MAGIC <b><i>* TRANSFORMATIONS<br>
# MAGIC * CLEANSED DATASET<br>
# MAGIC * CATEGORIZATION<br>
# MAGIC * ANALYSIS<br>
# MAGIC * VISUALIZATIONS<br>

# COMMAND ----------

# MAGIC %md
# MAGIC <h1>* TRANSFORMATIONS</h1><br>
# MAGIC <b><i>Performed some pre-processing or cleansing steps in order to clean the extracted dataset

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>1. Calculating null count of every column
# MAGIC

# COMMAND ----------

null_count=[(c,df1.filter(col(c).isNull()).count())for c in df1.columns]
null_count

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC Obtained null count of every column.

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>2. Dropping the columns having high null count

# COMMAND ----------

df2=df1.drop('org')
df2.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC Obtained a dataframe by dropping "org" column as it has highest null count i.e., 16667.

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>3. Flattening of data</h3><br>
# MAGIC As the extracted dataset consists of data in the form of nested datastructure, so inorder to make that as a flat and clear table here i performed flattening and selected only few columns from the extracted dataset.

# COMMAND ----------

df3=df2.select(
    col("actor.login").alias("Actor_Login"),
    col("actor.id").alias("Actor_ID"), 
    col("created_at").alias("Event_CreatedAt"),
    col("id").alias("Event_ID"),
    col("payload.issue.body").alias("body"),
    col("payload.issue.created_at").alias("Issue_CreatedAt"),
    col("payload.issue.id").alias("Issue_ID"),
    col("payload.issue.labels.color").alias("Label_Color"),
    col("payload.issue.labels.name").alias("Label_Name"),
    col("payload.issue.labels.url").alias("Label_URL"),
    col("payload.issue.state").alias("Issue_State"),
    col("payload.issue.title").alias("title"),
    col("payload.issue.updated_at").alias("Issue_UpdatedAt"),
    col("repo.name").alias("Repo_Name"),
    col("type").alias("Event_Type"),
    col("payload.release.author.login").alias("Release_Author_Login"),
    col("payload.release.tag_name").alias("Release_Tag_Version"),
    col("payload.forkee.owner.login").alias("Forkee_Owner_Login"),
    col("payload.forkee.owner.repos_url").alias("Forkee_Owner_repo_URL"),
    col("payload.commits.author.name").alias("Commits_Author_Name"))
df3.display()
df3.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC Obtained flattened dataset and its schema which consists of only relevant and focused columns of the extracted dataset which are useful for getting insights of the data and which are useful for implementing the usecase i.e., "Automatic Labelling of Issues".

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>4. Casting the datatypes</h3><br>
# MAGIC Here data types of some columns are mismatched, so i casted them to their correct data types.

# COMMAND ----------

df4=(df3.withColumn("Event_ID", df3.Event_ID.cast("long"))
        .withColumn("Event_CreatedAt", df3.Event_CreatedAt.cast("timestamp"))
        .withColumn("Issue_CreatedAt", df3.Issue_CreatedAt.cast("timestamp"))
        .withColumn("Issue_UpdatedAt", df3.Issue_UpdatedAt.cast("timestamp")))
df4.display()
df4.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC Obtained dataset and schema with the updated data types of "Event_ID" from string to "long" and "Event_CreatedAt", "Issue_CreatedAt", and "Issue_UpdatedAt" from string to "timestamp".

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>5. Calculating null percentage of every column in the flattened dataset

# COMMAND ----------

total_rows = df4.count()
null_percentages=[(c,df4.filter(col(c).isNull()).count()/total_rows)for c in df4.columns]
null_percentages

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC Obtained null percentage of every column, but here i doesn't dropped any columns as i performed flattening and selected the columns which are usefull.

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>6. Checking for row level duplicates</h3><br>
# MAGIC Checking whether there are any duplicate rows in the dataset inorder to drop if there are any duplicate rows.

# COMMAND ----------

duplicates_df4 = df4.groupBy(df4.columns).count().filter(col("count") > 1)
num_rows = duplicates_df4.count()
if num_rows==0:
    print("There are no Duplicates")
else:
    print(f"Number of rows: {num_rows}")

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC Here it was displayed like "There are no Duplicates", as the dataset doesn't have any row level duplicates. So, here there is no need to drop any rows. 

# COMMAND ----------

# MAGIC %md
# MAGIC <h1>* CLEANSED DATASET</h1><br>
# MAGIC <b><i>This is the cleansed dataset obtained after performing all required and possible transformations on the extracted dataset

# COMMAND ----------

df4.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h1>* CATEGORIZATION</h1><br>
# MAGIC Categorizing issues as:<br>
# MAGIC <b><i>0 = Bug<br>
# MAGIC 1 = Feature<br>
# MAGIC 2 = Question

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>1. Exploding Label_Name column</h3><br>
# MAGIC Exploding "Label_Name" column and giving that data to newly added "issues" column, because the "label_Name" column itself consists of different types of issues.

# COMMAND ----------

df5=df4.withColumn("issues",explode(df4.Label_Name))
df5.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC A new column, i.e., "issues" was added to the dataframe which consists of different types of issues.

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>2. Classification</h3><br>
# MAGIC Classification of issues present in the issues column as Bugs, Features, and Questions.

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import when

def categorize_issue(issue):
    bug_keywords = ["bug", "problem", "error", "invalid", "exception", "reproducible", "confirmed", "in progress", "fix", "issue", "glitch", "crash", "malfunction", "defect", "inconsistency", "unexpected behaviour"]
    feature_keywords = ["feature", "enhance", "implement", "suggest", "improve", "proposal", "edit", "addition", "functionality", "upgrade", "capability"]
    question_keywords = ["question", "request", "need", "help", "discuss", "support", "inquiry", "clarification", "information", "query", "assistance"]

    issue_lower = issue.lower()

    if any(keyword in issue_lower for keyword in bug_keywords):
        return "bug"
    elif any(keyword in issue_lower for keyword in feature_keywords):
        return "feature"
    elif any(keyword in issue_lower for keyword in question_keywords):
        return "question"
    else:
        return None

categorize_issue_udf = udf(categorize_issue, StringType())
df6 = df5.withColumn("label", categorize_issue_udf(df5.issues))
df6.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC Every issue in the issues column was classified as Bugs, Features, and Questions and also we got some Null values for issues other than Bugs, Features, and Questions and finally this categorized data was stored in the newly added "label" column.

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>3. Filtering</h3><br>
# MAGIC Filtering only the rows having Bugs, Features, and Questions from the label Column.

# COMMAND ----------

df7=df6.filter(col("label").isNotNull())
df7.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC All Null rows were removed from the label column and obtained a dataframe which consists of different categories of issues such as Bugs, Features, and Questions.<br>
# MAGIC Hence, Categorization of issues such as Bugs, Features, and Questions was done.

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>4. Categorization</h3><br>
# MAGIC Now categorizing issues as, 0=Bug, 1=Feature, and 2=Question.

# COMMAND ----------

df8 = df7.withColumn("label", when(df7.label == "bug", 0)
                                  .when(df7.label == "feature", 1)
                                  .when(df7.label == "question", 2))
df8.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Result</h4><br>
# MAGIC Finally obtained categorized dataframe which consists of numerical labels, in which "0" indicates "Bug", "1" indicates "Feature" and "2" indicates "Question".

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>--> Selecting only focussed columns from categorized dataset which are required for building and training a machine learning model.

# COMMAND ----------

df7=df7.select(
    col("Actor_Login"),
    col("body"),
    col("Issue_CreatedAt"),
    col("Issue_ID"),
    col("Issue_State"),
    col("title"),
    col("Issue_UpdatedAt"),
    col("Event_Type"),
    col("issues"),
    col("label"))
df7.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>--> Saving the final dataset as a json file in DBFS

# COMMAND ----------

# Specify the directory path
directory_path = '/FileStore/tables/FinalData/'
 
# Create the directory if it doesn't exist
dbutils.fs.mkdirs(directory_path)

df = df7.toPandas()
modified_json = df.to_json(orient='records')

# Write the JSON file directly to the desired directory in DBFS
dbutils.fs.put(directory_path + "Final.json", modified_json, overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC <h1>* ANALYSIS</h1><br>
# MAGIC <b><i>Doing some Analysis on Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>1. Counting no. of bugs, features, and questions triggered by the users

# COMMAND ----------

df9=df7.select(col("label"))
df9=df9.groupBy("label").count()
df9.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>2. Checking how may issues were in closed state and how many issues were in open state

# COMMAND ----------

df10=df4.select(col("Issue_State"))
df10=df10.filter(col("Issue_State").isNotNull())
df10=df10.groupBy("Issue_State").count()
df10.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>3. Checking how many issues were created at a particular timestamp

# COMMAND ----------

from pyspark.sql.functions import desc
df11=df4.select(col("Issue_CreatedAt"))
df11=df11.filter(col("Issue_CreatedAt").isNotNull())
df11=df11.groupBy("Issue_CreatedAt").count()
df11=df11.orderBy(desc("count"))
df11=df11.limit(10)
df11.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>4. Checking how many issues were updated at a particular minute of a timestamp

# COMMAND ----------

from pyspark.sql.functions import minute
df12=df4.select(col("Issue_UpdatedAt"))
df12=df12.filter(col("Issue_UpdatedAt").isNotNull())
df12 = df12.select(col("Issue_UpdatedAt"), minute(df12["Issue_UpdatedAt"]).alias("Minute"))
df12.display()
df13=df12.select(col("Minute"))
df13 = df13.groupBy("Minute").count()
df13 = df13.orderBy(desc("count"))
df13=df13.limit(10)
df13.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>5. Identifying the most frequent Issue URL explored by user

# COMMAND ----------

df14= df4.select(col("Label_URL"))
df14=df14.selectExpr("explode(Label_URL) as Label_URL")
df14=df14.groupBy("Label_URL").count()
df14=df14.orderBy(desc("count"))
df14=df14.limit(10)
df14.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>6. Counting different types of events in order to know which events were frequently taken place by the users

# COMMAND ----------

df15=df4.select(col("Event_Type"))
df15=df15.groupBy("Event_Type").count()
df15=df15.orderBy(desc("count"))
df15.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>7. Counting the most events occured per repository from the data

# COMMAND ----------

df16 = df4.select(col("Repo_Name"), 
                  col("Event_Type"))
df16=df16.groupBy('Repo_Name', 'Event_Type').count()
df16 = df16.orderBy(desc("count"))
df16=df16.limit(10)
df16.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>8. Checking for the most frequent users who triggered the events

# COMMAND ----------

df17=df4.select(col("Actor_Login"))
df17= df17.groupBy("Actor_Login").count()
df17 = df17.orderBy(desc("count"))
df17=df17.limit(10)
df17.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>9. Identifying and counting the events occured per actor from the data

# COMMAND ----------

df18=df4.select(
    col("Actor_Login"),
    col("Event_Type"))
df18=df18.groupBy('Actor_Login', 'Event_Type').count()
df18 = df18.orderBy(desc("count"))
df18=df18.limit(10)
df18.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>10. Identifying and counting the type of events raised by the particular organisation

# COMMAND ----------

df19=df1.select(col("actor.login").alias("Actor_Login"),
                col("org.login").alias("Organisation_Login"),
                col("type").alias("Event_Type"))
df19 = df19.filter(col("Organisation_Login").isNotNull())
df19= df19.groupBy("Actor_Login","Organisation_Login","Event_Type").count()
df19 = df19.orderBy(desc("count"))
df19=df19.limit(10)
df19.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>11. Count of no. of releases done by the authors

# COMMAND ----------

df20=df4.select(col("Release_Author_Login"),
                col("Release_Tag_Version"))
df20 = df20.filter(col("Release_Author_Login").isNotNull())
df20 = df20.groupBy("Release_Author_Login","Release_Tag_Version").count()
df20 = df20.orderBy(desc("count"))
df20 = df20.limit(10)
df20.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>12. Identifying the users who forked others repositories

# COMMAND ----------

df21=df4.select(col("Forkee_Owner_Login"),
                col("Forkee_Owner_Repo_URL"))
df21=df21.filter(col("Forkee_Owner_Login").isNotNull())
df21 = df21.groupBy("Forkee_Owner_Login","Forkee_Owner_Repo_URL").count()
df21 = df21.orderBy(desc("count"))
df21 = df21.limit(10)
df21.display()

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>13. Identifying on which repository more number of commits were done by the user

# COMMAND ----------

df22=df4.select(col("Commits_Author_Name"),
                col("Repo_Name"))
df22=df22.filter(col("Commits_Author_Name").isNotNull()) 
df22=df22.withColumn("Commits_Author_Name", explode("Commits_Author_Name"))
df22 = df22.groupBy("Commits_Author_Name","Repo_Name").count()
df22 = df22.orderBy(desc("count"))
df22 = df22.limit(10)             
df22.display()

# COMMAND ----------

# MAGIC %md <h1>* VISUALIZATIONS</h1><br>
# MAGIC <b><i>Reports based on the Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>1. Pie Chart on the Issue Count which includes no. of Features, Bugs, and Questions

# COMMAND ----------

df9.display()

# COMMAND ----------

# MAGIC %md <h4> Insight </h4>
# MAGIC The dataset consists of:
# MAGIC <b> <i> 47.7% Features, 39.1% Bugs, 13.2% Questions

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> 2. Pie Chart on the State Count which includes whether the issue is open or closed

# COMMAND ----------

df10.display()

# COMMAND ----------

# MAGIC %md <h4> Insights </h4>
# MAGIC The dataset consists of:
# MAGIC <b> <i> 68.2% Open States, 31.8% Closed States 

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> 3. Line Graph which despicts the no. of issues raised at a particular timestamp 

# COMMAND ----------

df11.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC Maximum issues are raised at these 3 timestamps <b><i>2015-01-01T01:02:34.000+0000, 2014-12-29T00:50:38.000+0000, 2015-01-01T16:05:57.000+0000</i></b> i.e., <i><b>12

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>4.  Line Graph which despicts the no. of issues updated per minute

# COMMAND ----------

df13.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC Highest no. of issues are updated i.e., <b><i>68</i> </b> at <b><i>15th</i> </b> minute

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>5.  Bar Graph on most frequently explored Issue URL by actor

# COMMAND ----------

df14.display()

# COMMAND ----------

# MAGIC %md <h3> Insight </h3>
# MAGIC Mostly Explored URL: <i> <b>https://api.github.com/repos/samplesizeofone/objective/labels/todo </b></i> for <i><b> 16 </b></i> times
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> 6. Area Graph which shows the frequency of events occurred 

# COMMAND ----------

df15.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC  Most Occuring Event is <b><i>Push Event</i></b> for <i><b>12,145</b></i> times

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> 7. Bar graph on particular type of event count in a repository 

# COMMAND ----------

df16.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC Repository with highest no.of events: <b><i>KenanSulayman/heartbeat</i> </b> i.e., <b><i>159 Push Events</i> </b><br>
# MAGIC Type of event which occured mostly: <b><i>Push Event 

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>8.  Area Graph on no. of events done by a particular actor

# COMMAND ----------

df17.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC <b><i> KenanSulayman</i></b> did more no. of events i.e., <b><i>159

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> 9.  Bar graph on particular type of event count done by actor

# COMMAND ----------

df18.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC <b><i> KenanSulayman </i></b> did more no. of events i.e., <i><b> 159 </i></b><br>
# MAGIC Next <b><i>opencm</i></b>  did more no. of events : <i><b>131 </i></b> (66 Create Event, 65 Delete Event)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>10. Bar graph on particular type of event count done by organisation

# COMMAND ----------

df19.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC <b><i> cloudify-cosmo  </i></b> organisation did more no. of events<br>
# MAGIC <h6>Total: 127 (64 Create Event, 63 Delete Event)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>11.  Bar Graph on no. of releases done by an author

# COMMAND ----------

df20.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC <b><i> stevewest </i></b>'s version(2.0.0) count is high i.e., <b><i>30

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>12. Area Graph on no. of forks done by a particular user

# COMMAND ----------

df21.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC <b><i>Tripurstar</i></b> forked highest no. of repositories i.e., <b><i>16

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>13. Bar Graph on no. of commits done by a particular user on a particular repo

# COMMAND ----------

df22.display()

# COMMAND ----------

# MAGIC %md <h4>Insight</h4>
# MAGIC Highest no. of commits were done on <b><i>sakai-mirror/melete </i></b>  <br>
# MAGIC <h6>Total: 352 (308 by thoppaymallika@fhda.edu, 44 by rashmi@etudes.org)
