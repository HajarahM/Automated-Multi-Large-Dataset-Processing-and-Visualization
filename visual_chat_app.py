import os
import pandas as pd
import pygwalker as pyg
import streamlit as st
import streamlit.components.v1 as components
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html

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

    def visualize_dataset(self):
        st.title("Interactive Dataset Visualization")
        dataset_names = self.list_datasets()

        if not dataset_names:
            st.warning("No preprocessed datasets found in the 'preprocessed_data' folder.")
            return

        selected_dataset = st.selectbox("Select a dataset to visualize", dataset_names)
        selected_dataset_path = os.path.join(self.preprocessed_data_folder, selected_dataset.replace('/', os.sep) + '.csv')

        col1, col2 = st.columns(2)

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

    def chat_with_dataset(self, uploaded_file):
        st.title("Interactive Dataset Visualization")
        dataset_names = self.list_datasets()

        if not dataset_names:
            st.warning("No preprocessed datasets found in the 'preprocessed_data' folder.")
            return

        selected_dataset = st.selectbox("Select a dataset to visualize", dataset_names)
        selected_dataset_path = os.path.join(self.preprocessed_data_folder, selected_dataset.replace('/', os.sep) + '.csv')

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
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
                st.session_state['generated'] = ["Hello ! Ask me(LLAMA2) about " + uploaded_file.name + " 🤗"]

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
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5
        )
        return llm

    def run_app(self):
        st.title("Interactive Dataset Visualization")
        dataset_names = self.list_datasets()

        if not dataset_names:
            st.warning("No preprocessed datasets found in the 'preprocessed_data' folder.")
            return

        selected_dataset = st.selectbox("Select a dataset to visualize", dataset_names)
        selected_dataset_path = os.path.join(self.preprocessed_data_folder, selected_dataset.replace('/', os.sep) + '.csv')

        self.visualize_dataset()

        if st.button("Chat with Dataset"):
            st.write(f"Chatting with dataset: {selected_dataset}")
            self.chat_with_dataset(selected_dataset)

if __name__ == "__main__":
    visualization = InteractiveVisualization()
    visualization.run_app()