import os
import re
import pandas as pd
import string
from AutoClean import AutoClean
from ydata_profiling import ProfileReport
import pdfkit  # Import pdfkit for HTML to PDF conversion
import matplotlib
import warnings

class DataProcessor:
    """
    A class for processing datasets, performing EDA, and saving reports.
    """

    def __init__(self):
        """
        Initialize the DataProcessor with folder paths.
        """
        self.raw_data_folder = './raw_data'
        self.preprocessed_data_folder = './preprocessed_data'
        
    def create_folders(self):
        """
        Create necessary folders if they don't exist.
        """
        for folder in [self.preprocessed_data_folder]:
            os.makedirs(folder, exist_ok=True)
            for category in os.listdir(self.raw_data_folder):
                category_path = os.path.join(folder, category)
                os.makedirs(category_path, exist_ok=True)
                os.makedirs(os.path.join(category_path, 'html_reports'), exist_ok=True)
                os.makedirs(os.path.join(category_path, 'pdf_reports'), exist_ok=True)

    def read_dataset(self, dataset_path):
        """
        Read a dataset from various formats (e.g., CSV, Excel, text).

        Args:
            dataset_path (str): The path to the dataset file.

        Returns:
            pd.DataFrame or None: The dataset as a DataFrame or None if an error occurs.
        """
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(dataset_path)
            elif dataset_path.endswith('.txt'):
                # Add code to read txt files if needed
                pass
            else:
                print(f"Unsupported file format for {dataset_path}")
                df = None
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        except Exception as e:
            print(f"Error reading {dataset_path}: {str(e)}")
            df = None
        return df

    def preprocess_dataset(self, dataset_path):
        """
        Preprocess a dataset by converting Int64 columns to float, dropping specific columns,
        removing columns whose rows contain Arabic alphabet characters,
        and dropping rows with incompatible data types or values.

        Args:
            dataset_path (str): The path to the dataset file.

        Returns:
            pd.DataFrame or None: The preprocessed dataset as a DataFrame or None if an error occurs.
        """
        df = self.read_dataset(dataset_path)
        if df is not None and not df.empty:
            # Check if columns exist before attempting to drop them
            columns_to_drop = ['Unnamed: 0']
            existing_columns = [col for col in columns_to_drop if col in df.columns]

            # Drop specified columns if they exist
            if existing_columns:
                df = df.drop(columns=existing_columns)

            # Convert Int64 columns to float, handling non-integer values gracefully
            int64_columns = df.select_dtypes(include=['Int64']).columns
            for col in int64_columns:
                # Use pd.to_numeric with errors='coerce' to convert invalid values to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove columns whose rows contain Arabic alphabet characters
            for col in df.columns:
                if df[col].apply(lambda x: bool(re.search(r'[\u0600-\u06FF]', str(x)))).any():
                    df = df.drop(columns=[col])

            # Drop rows with NaN values
            if not df.empty:
                df = df.dropna()

            # Perform AutoClean
            if not df.empty:
                cleaned = AutoClean(df, mode='auto')
                df = cleaned.output

            return df
        return None


    def perform_eda(self, category, dataset_name, df):
        """
        Perform exploratory data analysis (EDA) and save reports.

        Args:
            category (str): The category of the dataset.
            dataset_name (str): The name of the dataset.
            df (pd.DataFrame): The preprocessed dataset.

        Returns:
            None
        """
        if df is not None and not df.empty:
            sanitized_dataset_name = self.sanitize_filename(dataset_name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
                profile = ProfileReport(df, title=f"{sanitized_dataset_name} Profiling Report")

            # Construct the file paths for saving HTML and PDF reports
            html_report_path = os.path.join(self.preprocessed_data_folder, category, 'html_reports', f"{sanitized_dataset_name}.html")
            pdf_report_path = os.path.join(self.preprocessed_data_folder, category, 'pdf_reports', f"{sanitized_dataset_name}.pdf")

            # Check if the HTML report was created successfully
            if profile.to_file(html_report_path) is not None:
                # Convert HTML to PDF using pdfkit
                pdfkit.from_file(html_report_path, pdf_report_path)

    def process_datasets(self):
        """
        Process datasets, perform EDA, and save reports.
        """
        self.create_folders()

        for category in os.listdir(self.raw_data_folder):
            category_path = os.path.join(self.raw_data_folder, category)
            if os.path.isdir(category_path):
                for dataset_file in os.listdir(category_path):
                    dataset_name, ext = os.path.splitext(dataset_file)
                    dataset_path = os.path.join(category_path, dataset_file)

                    preprocessed_df = self.preprocess_dataset(dataset_path)
                    if preprocessed_df is not None:
                        preprocessed_folder = os.path.join(self.preprocessed_data_folder, category)
                        os.makedirs(preprocessed_folder, exist_ok=True)
                        preprocessed_df.to_csv(os.path.join(preprocessed_folder, f"{dataset_name}.csv"), index=False)

                    self.perform_eda(category, dataset_name, preprocessed_df)

    @staticmethod
    def sanitize_filename(filename):
        """
        Sanitize a filename to remove special characters.

        Args:
            filename (str): The filename to sanitize.

        Returns:
            str: The sanitized filename.
        """
        valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
        return ''.join(c for c in filename if c in valid_chars)
    
if __name__ == "__main__":
    data_processor = DataProcessor()
    data_processor.process_datasets()