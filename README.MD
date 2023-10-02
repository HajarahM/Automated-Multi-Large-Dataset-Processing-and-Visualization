# Data Processor

The Data Processor is a Python class designed to simplify the process of performing Exploratory Data Analysis (EDA) and data preprocessing on multiple datasets stored in a specified folder structure. It supports reading datasets in various formats (e.g., CSV, Excel), preprocessing using the AutoClean library, generating EDA reports, and saving cleaned datasets and reports in organized folders.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [License](#license)

## Features

- Read datasets from different file formats (CSV, Excel, etc.).
- Preprocess datasets using the AutoClean library.
- Generate EDA reports using pandas profiling.
- Save cleaned datasets and reports in organized folders.

## Requirements

- Python 3.6 or higher.
- AutoClean library (Install using `pip install AutoClean`).
- pandas library (Included in the standard requirements).

## Usage

1. Clone or download this repository.

2. Place your datasets in the `./raw_data` folder organized into subfolders by category.

3. Update the `./raw_data` folder structure according to your dataset categories.

4. Run the `data_processor.py` script to process the datasets:

   ```bash
   python data_processor.py

##Folder Structure

The recommended folder structure for your datasets and reports is as follows:

project-root/
│
├── data_processor.py
│
├── raw_data/
│   ├── category1/
│   │   ├── dataset1.csv
│   │   ├── dataset2.xlsx
│   │   └── ...
│   ├── category2/
│   │   ├── dataset3.csv
│   │   ├── dataset4.xlsx
│   │   └── ...
│   └── ...
│
├── preprocessed_data/
│   ├── category1/
│   │   ├── dataset1.csv
│   │   ├── dataset2.csv
│   │   └── ...
│   ├── category2/
│   │   ├── dataset3.csv
│   │   ├── dataset4.csv
│   │   └── ...
│   └── ...
│
└── EDA_reports/
    ├── html_reports/
    │   ├── category1/
    │   │   ├── dataset1.html
    │   │   ├── dataset2.html
    │   │   └── ...
    │   ├── category2/
    │   │   ├── dataset3.html
    │   │   ├── dataset4.html
    │   │   └── ...
    │   └── ...
    ├── pdf_reports/
    │   ├── category1/
    │   │   ├── dataset1.pdf
    │   │   ├── dataset2.pdf
    │   │   └── ...
    │   ├── category2/
    │   │   ├── dataset3.pdf
    │   │   ├── dataset4.pdf
    │   │   └── ...
    │   └── ...
    └── ...

Ensure your dataset files are placed in the ./raw_data folder and organized into subfolders by category. 
The Data Processor will create the necessary subfolders in ./preprocessed_data and ./EDA_reports as needed.

## License
This Data Processor is provided under the MIT License.

Author: Hajarah Nantege Muwonge