import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import pandas as pd
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
import json

load_dotenv()

# Accessing secrets from st.secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Parsing the JSON strings stored in the TOML file
token_info = json.loads(st.secrets["token"]["token"])
client_secret_info = json.loads(st.secrets["client_secret"]["client_secret"])

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def create_google_drive_service(credentials):
    return build('drive', 'v3', credentials=credentials)

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # To allow OAuth on http for localhost

def authenticate_google_drive():
    # Use the parsed JSON objects
    flow = Flow.from_client_config(
        client_secret_info,
        scopes=SCOPES
    )
    flow.run_local_server(port=0)
    credentials = flow.credentials
    return credentials

def fetch_google_drive_files(service):
    results = service.files().list(
        pageSize=50,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    files = results.get('files', [])
    return files

def download_file(service, file_id, mime_type):
    if mime_type == 'application/vnd.google-apps.document':
        # For Google Docs
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
    elif mime_type == 'application/pdf':
        # For PDF files
        request = service.files().get_media(fileId=file_id)
    elif mime_type == 'text/plain':
        # For plain text files
        request = service.files().get_media(fileId=file_id)
    else:
        st.error(f"Unsupported file type: {mime_type}")
        return None

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read().decode('utf-8')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf.name.lower().endswith('.pdf'):
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            st.error(f"File {pdf.name} is not a PDF. Please upload only PDF files.")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=750,
        chunk_overlap=250,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.write("Please upload your customer interview information or paste text to begin.")
        return

    # Update chat history with user input
    st.session_state.chat_history.append({"type": "user", "content": user_question})

    # Display updated chat history
    for message in st.session_state.chat_history:
        if message["type"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    
    with st.spinner("Thinking..."):
        # Get response from conversation chain
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history.append({"type": "bot", "content": response['answer']})

    # Display bot response
    st.write(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)

def display_suggestions():
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = [
            "Summarize my interviews",
            "What are the key insights?",
            "What are the patterns?",
            "What are the problems?",
        ]
    
    cols = st.columns(len(st.session_state.suggestions))
    for i, suggestion in enumerate(st.session_state.suggestions):
        if cols[i].button(suggestion):
            return suggestion
    return None

def main():
    st.set_page_config(page_title="thnkrAI", page_icon="favicon-transparent-256x256.png", layout="centered") 
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Understand Customer Interviews Better :rocket:")
    selected_suggestion = display_suggestions()
    user_question = st.chat_input("Ask a question about your user interviews")

    if user_question:
        handle_userinput(user_question)
    elif selected_suggestion:
        handle_userinput(selected_suggestion)

    with st.sidebar:
        st.image("logo-transparent-png (1).png", use_column_width=True)
        st.subheader("Your Interview Docs")
        if 'google_credentials' not in st.session_state:
            if st.button("Connect to Google Drive"):
                st.session_state.google_credentials = authenticate_google_drive()
                if st.session_state.google_credentials:
                    st.success("Successfully connected to Google Drive!")
                    st.rerun()  # Rerun the app to update the sidebar

        st.title("Google Drive Authentication")
        if "google_credentials" not in st.session_state:
            st.session_state.google_credentials = authenticate_google_drive()
            
            relevant_files = [file for file in files if file['mimeType'] in 
                ['text/plain', 'application/vnd.google-apps.document', 'application/pdf']]

            selected_files = st.multiselect(
                "Select files from Google Drive",
                options=[f['name'] for f in relevant_files]
            )

            if selected_files and st.button("Process Selected Files"):
                with st.spinner("Processing files..."):
                    combined_text = ""
                    for file_name in selected_files:
                        file = next(f for f in relevant_files if f['name'] == file_name)
                        file_content = download_file(service, file['id'], file['mimeType'])
                        if file_content:
                            combined_text += f"File: {file_name}\n\n{file_content}\n\n"

                    if combined_text:
                        text_chunks = get_text_chunks(combined_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Files processed successfully!")
                    else:
                        st.error("Failed to retrieve file content")
        
        pdf_docs = st.file_uploader(
            "Upload your interview notes or transcripts", accept_multiple_files=True)
            
        if pdf_docs:
            if st.button("Upload"):
                with st.spinner("Uploading"):
                    # get pdf contents
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.raw_text = raw_text
                    st.write(raw_text)
                   
                    # break pdf contents into text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.text_chunks = text_chunks

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # conversation chain created
                    st.session_state.conversation = get_conversation_chain(vectorstore)

        st.subheader("Or Paste Text")
        user_text = st.text_area("Paste text here")
       
        if user_text:
            if st.button("Process Text"):
                with st.spinner("Processing"):
                    text_chunks = get_text_chunks(user_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Text processed successfully!")

if __name__ == '__main__':
    main()
