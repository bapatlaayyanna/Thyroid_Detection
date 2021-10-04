
# Thyroid Detection Using Machine Learning

A classification methodology to predict the type of Thyroid based on the given training data.

## Ex: 
1.negative,
2.compensated_hypothyroid, 
3.primary_hypothyroid, 
4.secondary_hypothyroid

## Architecture

![image](https://user-images.githubusercontent.com/47603420/135798450-56c8f1af-7a0a-4747-ad12-0a0e394cb1a4.png)



## Data Description
The client will send data in multiple files in batches to a given location. Data will contain different classes of thyroid and 30 columns of different values.
"Class" column will have four unique values “negative,compensated_hypothyroid,
primary_hypothyroid, secondary_hypothyroid”.

we also require a "schema" file from the client, which contains all the relevant information about the training files such as:
Name of the files, Length of Date stamp in FileName, Length of Time stamp in FileName, Number of Columns, Name of the Columns, and their datatypes.


  
## Data Validation 

Here, we perform different sets of validations on the given set of training raw files.  

1.	 ## Name Validation- 
We validate the name of the files based on the given name in the schema file. We have created a regex expression as per the name given in the schema file to use it for our validation. After validating the pattern in the name, we check the length of date in the file name & length of time in the file name. If all the values are as per the schema file, we move these files to "Good_Data Folder" else we move such files to "Bad_Data Folder."

2.	 ## Number of Columns - 
We validate the number of columns present in the files as per the value given in the schema file. If it doesn't match with the value given in the schema file, then the file is moved to "Bad_Data Folder."


3.	## Data type of the columns - 
Data type of the columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Bad_Data Folder".


4.	## Null values in columns - 
If any of the columns in a file have all the values as NULL or missing, we discard such file and move it to "Bad_Data Folder".


## Data Transformation
converts all the columns with string data type so that each value for that column is enclosed in quotes.

## Data Insertion in Database

1) ## Database Creation and Connection - 
Create a database with the given name. If the database is already created, open the connection to the database. 

2) ## Table creation in the database - 
Create a Table with name - "Good_Raw_Data" in database for inserting the files from the "Good_Data  Folder" based on given column names and data types in the schema file. 

3) ## Insertion of files into the table - 
All the files in the "Good_Data Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Bad_Data Folder".

## Model Training 
1) ## Data Export from DB  
The data in a stored database is exported as a CSV file for model training.

2) ## Data Pre-Processing   
   a) Drop the columns which not useful for training the model. These columns were selected while doing the EDA.
   
   b) Replace the invalid values with numpy “np.nan” so that we can use imputer on such values.
   
   c) Encode the categorical values
   
   d) Check for null values in the columns and impute these null values using the KNN imputer.
   
   e)  Handle the imbalanced dataset by using RandomOverSampler.

3) ## Clustering 
KMeans algorithm is used to create clusters in the pre-processed data. Select the number of clusters by using "KneeLocator" function.
Now, the Kmeans model is trained and the model is saved for the further use during prediction.

4) ## Model Selection 
After clusters are created, we find the best model for each cluster. Here, we are using two algorithms, "Random Forest" and "KNN". For each cluster, both the algorithms are passed with the best parameters derived from GridSearch. We calculate the AUC scores for both models and select the model with the best score. All the models for each cluster are saved for using them during prediction.

## Description of Prediction Data 

Client will send the data in multiple files in batches to a given location. Data will contain 29 columns of different values.

Apart from prediction files, we also require a "schema" file from client which contains all the relevant information about the training files such as:
Name of the files, Length of Date Stamp in FileName, Length of Time Stamp in FileName, Number of Columns, Name of the Columns and their data types.

## Data Validation
We validate the name of the files based on the given name in the schema file. We have created a regex expression as per the name given in the schema file to use it for our validation. After validating the pattern in the name, we check the length of date in the file name & length of time in the file name. If all the values are as per the schema file, we move these files to "Good_Data Folder" else we move such files to "Bad_Data Folder." 

1) ## Name Validation- 
We validate the name of the files based on the given name in the schema file. We have created a regex expression as per the name given in the schema file to use for our validation. After validating the pattern in the name, we check the length of date in the file name & length of time in the file name. If all the values are as per the requirement, we move such files to "Good_Data Folder" else we move such files to "Bad_Data Folder."

2) ## Number of Columns - 
We validate the number of columns present in the files as per the value given in the schema file. If it doesn't match with the value given in the schema file, then the file is moved to "Bad_Data Folder."

3) ## Data type of columns - 
Data type of the columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Bad_Data Folder".

4) ## Null values in columns - 
If any of the columns in a file have all the values as NULL or missing, we discard such file and move it to "Bad_Data Folder".

## Data Transformation
converts all the columns with string data type so that each value for that column is enclosed in quotes.
Data Insertion in Database 

1) ## Database Creation and connection - 
Create a database with the given name. If the database is already created, open the connection to the database. 

2) ## Table creation in the database - 
Create a Table with name - "Good_Raw_Data" in database for inserting the files from the "Good_Data  Folder" based on given column names and data types in the schema file. 

3) ## Insertion of files in the table - 
All the files in the "Good_Data Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Bad_Data Folder".

## Prediction 

1) ## Data Export from Db - 
The data in a stored database is exported as a CSV file for model training.

2) ## Data Pre-Processing   
   a) Drop the columns which not useful for training the model. These columns were selected while doing the EDA.
   
   b) Replace the invalid values with numpy “np.nan” so that we can use imputer on such values.
   
   c) Encode the categorical values
   
   d) Check for null values in the columns and impute these null values using the KNN imputer model which was saved during training.
3) ## Clustering - 

KMeans model created during training is loaded  and the clusters for the pre-processed prediction data is predicted.

4) ## Prediction - 
Based on the cluster number, the respective model is loaded and is used to predict the data for that cluster.

5)  Once the predictions are made for all the clusters, the predictions along with the original names before label encoder are saved in a CSV file to a given location and the location is returned to the client.


## Deployment (Heroku Platform)

1. Create an account on Heroku and github

2. Download and install git and Heroku CLI in your PC
https://cli-assets.heroku.com/heroku-x64.exe

https://git-scm.com/download/win

3. Run  "pip install gunicorn" in anaconda command prompt

4. Run "pip freeze>requirements.txt"
Requirements.txt file consists of all the packages that you need to deploy the app in the cloud.

![image](https://user-images.githubusercontent.com/47603420/135798516-8e9ba5cc-eb06-4dec-a908-f3857f88dea3.png)

main.py is the entry point of our application, where the flask server starts. 

![image](https://user-images.githubusercontent.com/47603420/135798565-4124da43-3a7f-4688-a6b4-eef376f415ac.png)


5. Create an empty "Procfile" and write content “web: gunicorn main:app” inside and save it.

Procfile :- It contains the entry point to the app.

![image](https://user-images.githubusercontent.com/47603420/135798571-4a8f5624-914b-4856-9087-e2e7a3e803b5.png)


6. Login to Github & create a Repository in Github and 
run below commands to load the project into Git.

git config --global user.email "<git registered mail id>"

git config --global user.name "<git user name>"

git init

git add .

git commit -m "first commit"

git branch -M main

git remote add origin <git hub url for your repo>

git push -u origin main

7.  Login to Heroku and create new App & 

run below commands to deploy our Flask application into Heroku platform.

heroku login

heroku git:remote -a thyroiddetection1

git push heroku main


## Training - Sending training request from Postman
  
  ![image](https://user-images.githubusercontent.com/47603420/135798640-4ae471a6-3628-4738-9c93-a05f70be795e.png)


## Prediction - Sending prediction request from Postman
  
  ![image](https://user-images.githubusercontent.com/47603420/135798691-bcd824bb-4e30-4e28-9b29-9d917b947b39.png)


  
