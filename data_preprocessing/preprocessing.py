import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pickle
from imblearn.over_sampling import RandomOverSampler


class Preprocessor:
    """
        This class is used to clean and transform the data before training.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self,data,columns):
        """
        Description: This method removes the given columns from dataframe.
        On Failure: Raise Exception
        :param data:
        :param columns:
        :return: DataFrame after removing the specified columns
        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data=data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1)
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
        Description: This method separates the features and Label Columns
        On Failure: Raise Exception
        :param data:
        :param label_column_name:
        :return: two separate Dataframes, one containing features and the other containing Labels
        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1)
            self.Y=data[label_column_name]
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def dropUnnecessaryColumns(self,data,columnNameList):
        """
        Description: This method drops the unwanted columns
        :param data:
        :param columnNameList:
        :return: data after dropping unwanted columns
        """
        data = data.drop(columnNameList,axis=1)
        return data


    def replaceInvalidValuesWithNull(self,data):
        """
        Description: This method replaces'?' values with null
        :param data:
        :return: data after replacing '?' values with null
        """

        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data

    def is_null_present(self,data):
        """
        Description: This method checks whether there are null values present in the pandas Dataframe or not
        On Failure: Raise Exception
        :param data:
        :return: Returns a Boolean Value. True if null values are present in the DataFrame,False if they are not present
        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        try:
            self.null_counts=data.isna().sum()
            for i in self.null_counts:
                if i>0:
                    self.null_present=True
                    break
            if(self.null_present):
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv('preprocessing_data/null_values.csv')
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def encodeCategoricalValues(self,data):
        """
        Description: This method encodes all the categorical values in training set
        On Failure: Raise Exception
        :param data:
        :return: A Dataframe which has all the categorical values encoded
        """
        data['sex'] = data['sex'].map({'F': 0, 'M': 1})

        for column in data.columns:
            if len(data[column].unique()) == 2:
                data[column] = data[column].map({'f': 0, 't': 1})

        #data = pd.get_dummies(data,columns=['referral_source'])

        encode = LabelEncoder().fit(data['Class'])

        data['Class'] = encode.transform(data['Class'])

        with open('EncoderPickle/enc.pickle', 'wb') as file:
            pickle.dump(encode, file)

        return data


    def encodeCategoricalValuesPrediction(self,data):
        """
        Description: This method encodes all the categorical values in the prediction set
        On Failure: Raise Exception
        :param data:
        :return: A Dataframe which has all the categorical values encoded
        """

        data['sex'] = data['sex'].map({'F': 0, 'M': 1})
        cat_data = data.drop(['age','T3','TT4','T4U','FTI','sex'],axis=1)

        for column in cat_data.columns:
            if (data[column].nunique()) == 1:
             if data[column].unique()[0]=='f' or data[column].unique()[0]=='F':
                data[column] = data[column].map({data[column].unique()[0] : 0})
             else:
                 data[column] = data[column].map({data[column].unique()[0]: 1})
            elif (data[column].nunique()) == 2:
                data[column] = data[column].map({'f': 0, 't': 1})

        #data = pd.get_dummies(data, columns=['referral_source'])

        return data

    def handleImbalanceDataset(self,X,Y):
        """
        Description: This method handles the imbalance in the dataset by oversampler
        On Failure: Raise Exception
        :param X:
        :param Y:
        :return: A Dataframe after balancing
        """
        rdsmple = RandomOverSampler()
        x_sampled, y_sampled = rdsmple.fit_resample(X, Y)

        return x_sampled,y_sampled
    def impute_missing_values(self, data):
        """
        Description: This method replaces all the missing values in the Dataframe using KNN Imputer
        On Failure: Raise Exception
        :param data:
        :return: A Dataframe which has all the missing values imputed using KNN Imputer
        """

        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data)
            self.new_data=pd.DataFrame(data=np.round(self.new_array), columns=self.data.columns)
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

