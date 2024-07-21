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
#from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain.prompts import PromptTemplate
#import webbrowser

#load_dotenv()

openai_api_key = st.secrets["OPENAI_API_KEY"]
huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
slack_bot_token = st.secrets["SLACK_BOT_TOKEN"]


client_secret_info = {
    "installed": {
        "client_id": st.secrets["google_oauth"]["client_id"],
        "project_id": st.secrets["google_oauth"]["project_id"],
        "auth_uri": st.secrets["google_oauth"]["auth_uri"],
        "token_uri": st.secrets["google_oauth"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["google_oauth"]["auth_provider_x509_cert_url"],
        "client_secret": st.secrets["google_oauth"]["client_secret"],
        "redirect_uris": st.secrets["google_oauth"]["redirect_uris"]
    }
}


SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def create_google_drive_service(credentials):
    return build('drive', 'v3', credentials=credentials)

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # To allow OAuth on http for localhost

def authenticate_google_drive():
    #st.write("Starting Google Drive authentication...")
    
    # Set up the OAuth flow
    redirect_uri = st.secrets["PRODUCTION_URL"] if "PRODUCTION_URL" in st.secrets else st.write('error')
    flow = Flow.from_client_config(
        client_secret_info,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    
    # Initialize session state
    if 'google_auth_state' not in st.session_state:
        st.session_state.google_auth_state = {}

    # Debug information
    #st.write(f"Session state: {st.session_state.google_auth_state}")
    #st.write(f"Query params: {st.query_params}")

    # Handle the redirect from Google
    if "code" in st.query_params:
        st.write("Received code from Google, attempting to fetch token...")
        try:
            flow.fetch_token(code=st.query_params["code"])
            st.session_state.google_auth_state['credentials'] = flow.credentials
            st.success("Successfully authenticated with Google Drive!")
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during authentication: {str(e)}")
            st.write(f"Error details: {e}")
    
    # If not authenticated, provide the authorization URL
    if 'credentials' not in st.session_state.google_auth_state:
        if 'auth_url' not in st.session_state.google_auth_state:
            authorization_url, _ = flow.authorization_url(prompt='consent')
            st.session_state.google_auth_state['auth_url'] = authorization_url

        #st.write("Click the button below to authorize the application:")
        auth_url = st.session_state.google_auth_state['auth_url']
        
        # Create a button that opens the auth URL in a new tab
        if st.button("Connect to Google Drive"):
            js = f"window.open('{auth_url}', '_blank');"
            html = f'<script>{js}</script>'
            st.components.v1.html(html, height=0)
        
        #st.info("After authorizing, you will be redirected back to this app.")

    return st.session_state.google_auth_state.get('credentials')

def fetch_google_drive_files(service):
    # Define the MIME types we're interested in
    mime_types = [
        'application/vnd.google-apps.document',  # Google Docs
        'text/plain',  # Text files
        'application/pdf'  # PDF files
    ]
    
    # Construct the query string
    query = " or ".join([f"mimeType='{mime_type}'" for mime_type in mime_types])
    
    results = service.files().list(
        q=query,
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

def authenticate_slack():
    if 'slack_credentials' not in st.session_state:
        slack_auth_url = "https://slack.com/oauth/v2/authorize?client_id=6004153764880.7387601103296&scope=incoming-webhook,chat:write&user_scope="
        
        if st.button("Connect to Slack"):
            js = f"window.open('{slack_auth_url}', '_blank');"
            html = f'<script>{js}</script>'
            st.components.v1.html(html, height=0)
        
        #st.info("After authorizing with Slack, you will be redirected back to this app.")
    else:
        st.success("Connected to Slack")

    return st.session_state.get('slack_credentials')
    
def handle_slack_oauth():
    if "code" in st.query_params:
        client_id = st.secrets["SLACK_CLIENT_ID"]
        client_secret = st.secrets["SLACK_CLIENT_SECRET"]
        redirect_uri = st.secrets["PRODUCTION_URL"]  # Make sure this matches your Slack app settings

        oauth_client = WebClient()
        try:
            oauth_response = oauth_client.oauth_v2_access(
                client_id=client_id,
                client_secret=client_secret,
                code=st.query_params["code"],
                redirect_uri=redirect_uri
            )
            
            # Store the bot token
            st.session_state.slack_credentials = oauth_response['access_token']
            st.success("Successfully connected to Slack!")
            st.rerun()
        except SlackApiError as e:
            st.error(f"Error during Slack authentication: {e}")


def generate_summary():
    if st.session_state.conversation is None:
        return "No conversation data available."
    
    summary_prompt = "Summarize the key insights from the customer interviews."
    response = st.session_state.conversation({'question': summary_prompt})
    return response['answer']

def send_to_slack(message, channel):
    client = WebClient(token=st.session_state.slack_credentials)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        return True
    except SlackApiError as e:
        error = e.response['error']
        if error == 'not_in_channel':
            st.error(f"Bot is not in the channel {channel}. Please invite the bot using /invite @YourBotName")
        elif error == 'channel_not_found':
            st.error(f"Channel {channel} not found. Please check the channel name and try again.")
        else:
            st.error(f"Error sending message to Slack: {error}")
        return False


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
    
    # Create a prompt template for the product manager role
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Provide detailed insights related to the information provided and question as a smart product manager with expertise in analyzing customer interviews would. "
            "Context: {context} "
            "Question: {question} "
            "Response:"
        )
    )

    # Create the ConversationalRetrievalChain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
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

def main():
    st.set_page_config(page_title="thnkrAI", page_icon="favicon-transparent-256x256.png", layout="centered") 
    st.write(css, unsafe_allow_html=True)

    #handle_slack_oauth()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    st.header("Understand Customer Interviews Better :rocket:")

    # Move Google Drive connection to main screen
    if 'google_credentials' not in st.session_state:
        credentials = authenticate_google_drive()
        if credentials:
            st.session_state.google_credentials = credentials
            st.rerun()

    # Google Drive file selection
    if "google_credentials" in st.session_state:
        service = create_google_drive_service(st.session_state.google_credentials)
        relevant_files = fetch_google_drive_files(service)

        selected_files = st.multiselect(
            "Select files from Google Drive",
            options=[f['name'] for f in relevant_files]
        )

        new_files = [f for f in selected_files if f not in st.session_state.uploaded_files]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} new file(s)..."):
                combined_text = ""
                for file_name in new_files:
                    file = next(f for f in relevant_files if f['name'] == file_name)
                    file_content = download_file(service, file['id'], file['mimeType'])
                    if file_content:
                        combined_text += f"File: {file_name}\n\n{file_content}\n\n"

                if combined_text:
                    text_chunks = get_text_chunks(combined_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.uploaded_files.extend(new_files)
                    st.success(f"Processed {len(new_files)} new file(s) successfully!")
                else:
                    st.error("Failed to retrieve file content")

    # File upload section
    #st.subheader("Or Upload PDF Files")
    pdf_docs = st.file_uploader(
        "", accept_multiple_files=True, label_visibility="hidden")
        
    if pdf_docs:
        new_files = [doc for doc in pdf_docs if doc.name not in st.session_state.uploaded_files]
        if new_files:
            with st.spinner("Processing new files..."):
                raw_text = get_pdf_text(new_files)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.uploaded_files.extend([doc.name for doc in new_files])
                st.success(f"Processed {len(new_files)} new file(s) successfully!")

    # Chat input
    user_question = st.chat_input("Ask a question about your user interviews")

    if user_question:
        handle_userinput(user_question)


    # Slack summary section
    #if st.session_state.conversation is not None and 'slack_credentials' in st.session_state:
    #    st.subheader("Send Summary to Slack")
    #    channel = st.text_input("Enter Slack channel name (e.g., #general)")
    #    if st.button("Send Summary"):
    #        summary = generate_summary()
    #        if channel and send_to_slack(summary, channel):
    #            st.success(f"Summary sent to Slack channel: {channel}")
    #        else:
    #            st.error("Failed to send summary to Slack")

                
    # Sidebar
    with st.sidebar:
        st.image("logo-transparent-png (1).png", use_column_width=True)
        st.subheader("Your Interview Docs")
        
        # Display uploaded file names
        if st.session_state.uploaded_files:
            st.write("Uploaded files:")
            for file in st.session_state.uploaded_files:
                st.write(f"- {file}")
        else:
            st.write("No files uploaded yet.")

        #slack_credentials = authenticate_slack()

if __name__ == '__main__':
    main()
