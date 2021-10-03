from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTransform_Training.DataTransformation import dataTransform
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from application_logging.logger import App_Logger

class train_validation:

    def __init__(self,path):
        self.raw_data=Raw_Data_validation(path)
        self.dataTransform=dataTransform()
        self.dBOperation=dBOperation()
        self.logger=App_Logger()
        self.file_object=open("Training_Logs/Training_Main_Log.txt", 'a+')

    def train_validation(self):
        try:
            self.logger.log(self.file_object,'Start of Validation on files for prediction!!')

            #extracting values from schema training json file
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns=self.raw_data.valuesFromSchema()

            #getting regex expression
            regex=self.raw_data.manualRegexCreation()

            #valdating training file names
            self.raw_data.validationFileNameRaw(regex,LengthOfDateStampInFile,LengthOfTimeStampInFile)

            #validating column length
            self.raw_data.validateColumnLength(noofcolumns)

            #validating any column has all missing values
            self.raw_data.validateMissingValuesInWholeColumn()
            self.logger.log(self.file_object,"Raw Data Validation Complete!!")

            self.logger.log(self.file_object,"Starting Data Transforamtion!!")
            #adding single quotes to string type values
            self.dataTransform.addQuotesToStringValuesInColumn()

            self.logger.log(self.file_object,"DataTransformation Completed!!!")

            self.logger.log(self.file_object,"Creating Training_Database and tables on the basis of given schema!!!")

            #create data table with the given name
            self.dBOperation.createTableDb("Training",column_names)
            self.logger.log(self.file_object,"Table creation Completed!!")

            self.logger.log(self.file_object,"Insertion of Data into Table started!!!!")
            #inserting Data into the data table
            self.dBOperation.insertIntoTableGoodData("Training")
            self.logger.log(self.file_object,"Insertion in Table completed!!!")

            self.logger.log(self.file_object,"Deleting Good Data Folder!!!")
            #deleting existing Good raw data folder
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.logger.log(self.file_object,"Good_Data folder deleted!!!")

            self.logger.log(self.file_object,"Moving bad files to Archive and deleting Bad_Data folder!!!")
            #Moving bad file into Archived folder
            self.raw_data.moveBadFilesToArchiveBad()
            self.logger.log(self.file_object,"Bad files moved to archive!! Bad folder Deleted!!")

            self.logger.log(self.file_object,"Validation Operation completed!!")

            self.logger.log(self.file_object,"Extracting csv file from table")
            #Exporting data into csv
            self.dBOperation.selectingDatafromtableintocsv("Training")
            self.file_object.close()

        except Exception as e:
            raise e











