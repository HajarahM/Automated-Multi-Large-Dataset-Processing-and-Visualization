import os
import pandas as pd
import pygwalker as pyg
import streamlit as st
import streamlit.components.v1 as components
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html

class InteractiveVisualization:
    """
    A class for interactive visualization of datasets using PygWalker and Streamlit.
    """

    def __init__(self):
        """
        Initialize the InteractiveVisualization with folder paths.
        """
        self.preprocessed_data_folder = './preprocessed_data'

    def list_datasets(self):
        """
        List available preprocessed datasets in the 'preprocessed_data' folder.

        Returns:
            list: A list of available dataset names.
        """
        dataset_names = []
        for category in os.listdir(self.preprocessed_data_folder):
            category_path = os.path.join(self.preprocessed_data_folder, category)
            if os.path.isdir(category_path):
                for dataset_file in os.listdir(category_path):
                    dataset_name, ext = os.path.splitext(dataset_file)
                    if ext == '.csv':
                        dataset_names.append(f"{category}/{dataset_name}")
                    
        return dataset_names

    def visualize_dataset(self):
        """
        Allow the user to select and visualize a dataset using PygWalker and Streamlit.
        """
        st.title("Interactive Dataset Visualization")
        dataset_names = self.list_datasets()

        if not dataset_names:
            st.warning("No preprocessed datasets found in the 'preprocessed_data' folder.")
            return

        selected_dataset = st.selectbox("Select a dataset to visualize", dataset_names)
        selected_dataset_path = os.path.join(self.preprocessed_data_folder, selected_dataset.replace('/', os.sep) + '.csv')

        # Create two columns for buttons
        col1, col2 = st.columns(2)

        if col1.button("Visualize"):
            st.write(f"Visualizing dataset: {selected_dataset}")
            
            # Load the selected dataset
            df = pd.read_csv(selected_dataset_path)

            # Initialize pygwalker communication
            init_streamlit_comm()

            # When using `use_kernel_calc=True`, you should cache your pygwalker html, if you don't want your memory to explode
            @st.cache_resource
            def get_pyg_html(df: pd.DataFrame) -> str:
                # When you need to publish your application, you need set `debug=False`,prevent other users to write your config file.
                # If you want to use feature of saving chart config, set `debug=True`
                html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
                return html

            @st.cache_data
            def get_df() -> pd.DataFrame:
                return df

            df = get_df()

            components.html(get_pyg_html(df), width=1300, height=1000, scrolling=True)

        if col2.button("View Report"):
            st.write(f"Viewing report for dataset: {selected_dataset}")
            
            # Extract category and dataset_name
            category, dataset_name = selected_dataset.split("/")

            # Call the view_html_report method
            self.view_html_report(category, dataset_name)

    def view_html_report(self, category, dataset_name):
        """
        View the HTML report for the selected dataset.

        Args:
            category (str): The category of the dataset.
            dataset_name (str): The name of the dataset.
        """
        html_report_path = os.path.join(
            self.preprocessed_data_folder, category, 'html_reports', f"{dataset_name}.html"
        )
        if os.path.exists(html_report_path):
            st.subheader("HTML Report")
            st.write(f"Viewing report for dataset: {dataset_name}")

            # Display the selected HTML report
            with open(html_report_path, "r", encoding="utf-8") as report_file:
                report_html = report_file.read()
                components.html(report_html, width=1300, height=1000, scrolling=True)
        else:
            st.warning(f"No HTML report found for dataset: {dataset_name}")

    def run_app(self):
        """
        Run the interactive visualization app.
        """
        self.visualize_dataset()

if __name__ == "__main__":
    visualization = InteractiveVisualization()
    visualization.run_app()