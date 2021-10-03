from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger

class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
    def trainingModel(self):

        self.log_writer.log(self.file_object, 'Start of Training')
        try:

            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()
            self.log_writer.log(self.file_object,"Got the data and entered into preprocessing")



            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            self.log_writer.log(self.file_object,'created preprocessing obj')

            #removing unwanted columns
            data = preprocessor.dropUnnecessaryColumns(data,['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured','TBG','TSH','referral_source'])
            self.log_writer.log(self.file_object,'Removed unnecessary columns')

            #repalcing '?' values with np.nan

            data = preprocessor.replaceInvalidValuesWithNull(data)

            # get encoded values for categorical data

            data = preprocessor.encodeCategoricalValues(data)

            #  separate features and labels
            X,Y=preprocessor.separate_label_feature(data,label_column_name='Class')

            # check if missing values are present in the dataset
            is_null_present=preprocessor.is_null_present(X)


            # if missing values are there, replace them
            if(is_null_present):
                X=preprocessor.impute_missing_values(X)

            X,Y = preprocessor.handleImbalanceDataset(X,Y)


            kmeans=clustering.KMeansClustering(self.file_object,self.log_writer)
            number_of_clusters=kmeans.elbow_plot(X)

            # Divide the data into clusters
            X=kmeans.create_clusters(X,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels']=Y

            # getting the unique clusters from our dataset
            list_of_clusters=X['Cluster'].unique()

            for i in list_of_clusters:
                cluster_data=X[X['Cluster']==i]

                # Prepare the feature and Label columns
                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']

                # splitting the data into training and test set for each cluster
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)

                model_finder=tuner.Model_Finder(self.file_object,self.log_writer)

                #getting the best model for each of the clusters
                best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

                #saving the best model to the directory.
                file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                save_model=file_op.save_model(best_model,best_model_name+str(i))

            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training::%s'%str(e))
            self.file_object.close()
            raise e