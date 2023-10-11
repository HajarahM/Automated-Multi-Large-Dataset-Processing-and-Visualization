import os
import openai
import pandas as pd
import pygwalker as pyg
import streamlit as st
import streamlit.components.v1 as components
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
from transformers import AutoModelForCausalLM
from decouple import config

# Get the directory where your application script is located
# script_directory = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to your llm model
# relative_model_path = "~/Applications/gpt4all/gpt4all-training/chat"

# Load the OpenAI API key from the .env file
openai_api_key = config('OPENAI_API_KEY', default='')

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

# Set the OpenAI API key
openai.api_key = openai_api_key

DB_FAISS_PATH = 'vectorstore/db_faiss'


class InteractiveVisualization:
    def __init__(self):
        self.preprocessed_data_folder = './preprocessed_data'

    def list_datasets(self):
        dataset_names = []
        for category in os.listdir(self.preprocessed_data_folder):
            category_path = os.path.join(self.preprocessed_data_folder, category)
            if os.path.isdir(category_path):
                for dataset_file in os.listdir(category_path):
                    dataset_name, ext = os.path.splitext(dataset_file)
                    if ext == '.csv':
                        dataset_names.append(f"{category}/{dataset_name}")
        return dataset_names

    def visualize_dataset(self, selected_dataset, selected_dataset_path, col1, col2):

        if col1.button("Visualize"):
            st.write(f"Visualizing dataset: {selected_dataset}")
            
            df = pd.read_csv(selected_dataset_path)
            init_streamlit_comm()

            @st.cache_resource
            def get_pyg_html(df: pd.DataFrame) -> str:
                html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
                return html

            @st.cache_data
            def get_df() -> pd.DataFrame:
                return df

            df = get_df()
            components.html(get_pyg_html(df), width=1300, height=1000, scrolling=True)

        if col2.button("View Report"):
            st.write(f"Viewing report for dataset: {selected_dataset}")
            
            category, dataset_name = selected_dataset.split("/")
            self.view_html_report(category, dataset_name)

    def chat_with_dataset(self, selected_dataset):

        loader = CSVLoader(file_path=selected_dataset, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)

        llm = self.load_llm()

        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me about " + selected_dataset.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

    def view_html_report(self, category, dataset_name):
        html_report_path = os.path.join(
            self.preprocessed_data_folder, category, 'html_reports', f"{dataset_name}.html"
        )
        if os.path.exists(html_report_path):
            st.subheader("HTML Report")
            st.write(f"Viewing report for dataset: {dataset_name}")

            with open(html_report_path, "r", encoding="utf-8") as report_file:
                report_html = report_file.read()
                components.html(report_html, width=1300, height=1000, scrolling=True)
        else:
            st.warning(f"No HTML report found for dataset: {dataset_name}")

    def load_llm(self):
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
        )

    def run_app(self):
        st.title("Interactive Dataset Visualization")
        dataset_names = self.list_datasets()
        col1, col2, col3 = st.columns(3)

        if not dataset_names:
            st.warning("No preprocessed datasets found in the 'preprocessed_data' folder.")
            return

        selected_dataset = st.selectbox("Select a dataset to visualize", dataset_names)
        selected_dataset_path = os.path.join(self.preprocessed_data_folder, selected_dataset.replace('/', os.sep) + '.csv')

        self.visualize_dataset(selected_dataset, selected_dataset_path, col1, col2)

        if col3.button("Chat with Dataset"):
            st.write(f"Chatting with dataset: {selected_dataset}")
            self.chat_with_dataset(selected_dataset_path)

if __name__ == "__main__":
    visualization = InteractiveVisualization()
    visualization.run_app()