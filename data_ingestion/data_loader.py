import pandas as pd

class Data_Getter:
    """
    Description : This class is to get data from the source for training
    """
    def __init__(self,file_object,logger_object):
        self.training_file_path='Training_FileFromDB/InputFile.csv'
        self.logger=logger_object
        self.file_object=file_object

    def get_data(self):
        self.logger.log(self.file_object,'Entered the get_data method of the Data_Getter class')
        try:
            self.data=pd.read_csv(self.training_file_path)
            self.logger.log(self.file_object,"Data Load Successful.Exited the get_data method of the Data_Getter class")
            return self.data
        except Exception as e:
            self.logger.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: %s'%str(e))
            self.logger.log(self.file_object,'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise e

