# Full path
"""
Created on Tue Oct 12 2021 by Hajarah Nantege
A class to loop through a respository or database with multiple datasets and perform EDA and preprocessing on each dataset.
Inside the class, there is:
1. A Method to read each dataset with a try function to try reading multiple formats of the dataset, that is, csv, excel, txt, etc and another try function to capture errors and print out the error message
2. A methods to use AutoClean to preprocess the dataset with an exception of 'False' on missing data
3. A method to perform EDA on the dataset using dataprep
4. A method to save the EDA report in html format within folder named './EDA_reports' and subfolder named './EDA_reports/html_reports' and actual report with the name that matches the dataset name or title of the dataset. If folder doesn't exist, create it and if subfolder doesn't exist, create it
5. A method to convert the html report to pdf format and save it in the './EDA_reports/pdf_reports' folder with the name that matches the dataset name or title of the dataset. If folder doesn't exist, create it and if subfolder doesn't exist, create it
6. A method to save the preprocessed dataset in csv format with the name that matches the dataset name or title of the dataset in the './preprocessed_data' folder. If folder doesn't exist, create it.
The class is then closed and called in main function
"""

# Import libraries
import os
import pandas as pd
import AutoClean
import pdfkit
import dataprep.eda as eda
from dataprep.eda import plot, plot_correlation, plot_missing
from dataprep.eda import create_report

# Create a class to loop through a respository or database with multiple datasets
class PreProcess:
    def __init__(self):
        """Initialize the class"""
        self.raw_data_path = './raw_data'

    def read_data(self):
        """Read each dataset with a try function to try reading multiple formats of the dataset, that is, csv, excel, txt, etc 
        and anothother try function to capture errors and print out the error message"""
        try:
            for file in os.listdir(self.raw_data_path):
                if file.endswith('.csv'):
                    raw_data = pd.read_csv(os.path.join(self.path, file))
                elif file.endswith('.xlsx'):
                    raw_data = pd.read_excel(os.path.join(self.path, file))
                elif file.endswith('.txt'):
                    raw_data = pd.read_csv(os.path.join(self.path, file))
                else:
                    print('Invalid file format')
        except Exception as e:
            print(e)
        return raw_data

    def preprocess_data(self, raw_data):
        """Use AutoClean to preprocess the dataset with an exception of 'False' on missing data and save the cleaned dataset in csv format with the name that matches the dataset name or title of the dataset in the './preprocessed_data' folder."""
        clean_dataset = AutoClean(df, mode='auto', duplicates='False', missing_num='False', outliers='False')
        #save clean dataset in csv format
        clean_dataset.to_csv('./preprocessed_data/{}.csv'.format(file))
        return clean_dataset

    def eda(self, clean_dataset):
        """Perform EDA on the cleaned dataset using DataPrep function 'create_report'
        Save the EDA report in html format within folder named './EDA_reports' and subfolder named './EDA_reports/html_reports' and actual report with the name that matches the dataset name or title of the dataset. 
        If folder doesn't exist, create it and if subfolder doesn't exist, create it"""
        report = create_report(clean_dataset.output)
        report.show_browser()
        report.save('./EDA_reports/html_reports/{}.html')
        return report

    def save_pdf(self):
        """Convert the html report saved in '.EDA_reports/html_reports/' to pdf format and save it in the './EDA_reports/pdf_reports' folder with the name that matches the corresponding html report name.
        If folder doesn't exist, create it and if subfolder doesn't exist, create it"""
        pdf_report = pdfkit.from_file('./EDA_reports/html_reports/{}.html', './EDA_reports/pdf_reports/{}.pdf')
        pdf_report.save('./EDA_reports/pdf_reports/{}.pdf')   


# Close the class and call it in main function
if __name__ == '__main__':
    preprocess = PreProcess()
    preprocess.read_data()
    preprocess.preprocess_data()
    preprocess.eda()
    preprocess.save_pdf()

# Update all imports in this python file and dependencies in the requirements.txt file
# Push all changes to your forked repository
