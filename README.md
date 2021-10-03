
# Thyroid Detection Using Machine Learning

A classification methodology to predict the type of Thyroid based on the given training data.


## Architecture

https://ibb.co/HGccrzb


## Data Description
The client will send data in multiple sets of files in batches at a given location. Data will contain different classes of thyroid and 30 columns of different values.
"Class" column will have four unique values “negative,compensated_hypothyroid,
primary_hypothyroid, secondary_hypothyroid”.

we also require a "schema" file from the client, which contains all the relevant information about the training files such as:
Name of the files, Length of Date stamp in FileName, Length of Time stamp in FileName, Number of Columns, Name of the Columns, and their datatypes.


  
## Data Validation 

In this step, we perform different sets of validation on the given set of training raw files.  

1.	 ## Name Validation- 
We validate the name of the files based on the given name in the schema file. We have created a regex expression as per the name given in the schema file to use for our validation. After validating the pattern in the name, we check the length of date in the file name & length of time in the file name. If all the values are as per the requirement, we move such files to "Good_Data Folder" else we move such files to "Bad_Data Folder."

2.	 ## Number of Columns - 
We validate the number of columns present in the files, and if it doesn't match with the value given in the schema file, then the file is moved to "Bad_Data Folder."


3.	## The data type of columns - 
The datatype of columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Bad_Data Folder".


4.	## Null values in columns - 
If any of the columns in a file have all the values as NULL or missing, we discard such a file and move it to "Bad_Data Folder".


## Data Transformation
converts all the columns with string data type such that each value for that column is enclosed in quotes.

## Data Insertion in Database

1) ## Database Creation and connection - 
Create a database with the given name passed. If the database is already created, open the connection to the database. 

2) ## Table creation in the database - 
Table with name - "Good_Raw_Data", is created in the database for inserting the files in the "Good_Data  Folder" based on given column names and data type in the schema file. 

3) ## Insertion of files in the table - 
All the files in the "Good_Data Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Bad_Data Folder".

## Model Training 
1) ## Data Export from Db - 
The data in a stored database is exported as a CSV file for model training.

2) ## Data Preprocessing   
   a) Drop columns not useful for training the model. Such columns were selected while doing the EDA.
   b) Replace the invalid values with numpy “np.nan” so we can use imputer on such values.
   c) Encode the categorical values
   d) Check for null values in the columns. If present, impute the null values using the KNN imputer.
   e)  Handle the imbalanced dataset by using RandomOverSampler.

3) ## Clustering - 
KMeans algorithm is used to create clusters in the preprocessed data. Select the number of clusters by using "KneeLocator" function.
The Kmeans model is trained and the model is saved for further use in prediction.

4) ## Model Selection - 
After clusters are created, we find the best model for each cluster. We are using two algorithms, "Random Forest" and "KNN". For each cluster, both the algorithms are passed with the best parameters derived from GridSearch. We calculate the AUC scores for both models and select the model with the best score. All the models for every cluster are saved for use in prediction.

## Prediction Data Description

Client will send the data in multiple set of files in batches at a given location. Data will contain 29 columns of different values.

Apart from prediction files, we also require a "schema" file from client which contains all the relevant information about the training files such as:
Name of the files, Length of Date Stamp in FileName, Length of Time Stamp in FileName, Number of Columns, Name of the Columns and their data types.

## Data Validation
In this step, we perform different sets of validation on the given set of training files.  

1) ## Name Validation- 
We validate the name of the files based on the given name in the schema file. We have created a regex expression as per the name given in the schema file to use for our validation. After validating the pattern in the name, we check the length of date in the file name & length of time in the file name. If all the values are as per the requirement, we move such files to "Good_Data Folder" else we move such files to "Bad_Data Folder."

2) ## Number of Columns - 
We validate the number of columns present in the files, if it doesn't match with the value given in the schema file then the file is moved to "Bad_Data Folder". 

3) ## Data type of columns - 
The data type of columns is given in the schema file. This is validated when we insert the files into Database. If data type is wrong then the file is moved to "Bad_Data Folder". 

4) ## Null values in columns - 
If any of the columns in a file have all the values as NULL or missing, we discard such file and move it to "Bad_Data Folder".  

## Data Transformation
converts all the columns with string data type such that each value for that column is enclosed in quotes.
Data Insertion in Database 

1) ## Database Creation and connection - 
Create database with the given name passed. If the database is already created, open the connection to the database. 

2) ## Table creation in the database - 
Table with name - "Good_Raw_Data", is created in the database for inserting the files in the "Good_Data Folder" based on the column names and datatype in the schema file. 

3) ## Insertion of files in the table - 
All the files in the "Good_Data  Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Bad_Data Folder".

## Prediction 

1) ## Data Export from Db - 
The data in the stored database is exported as a CSV file for prediction.

2) ## Data Preprocessing   
   a) Drop columns not useful for training the model. Such columns were selected while doing the EDA.
   b) Replace the invalid values with numpy “np.nan” so we can use imputer on such values.
   c) Encode the categorical values
   d) Check for null values in the columns. If present, impute the null values using the KNN imputer.
3) ## Clustering - 

KMeans model created during training is loaded  and clusters for the preprocessed prediction data is predicted.

4) ## Prediction - 
Based on the cluster number, the respective model is loaded and is used to predict the data for that cluster.

5)  Once the prediction is made for all the clusters, the predictions along with the original names before label encoder are saved in a CSV file at a given location and the location is returned to the client.

