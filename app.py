import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import pandas as pd

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
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.write("Please upload your customer interview information to begin.")
        return

    # Update chat history with user input
    st.session_state.chat_history.append({"type": "user", "content": user_question})

    # Display updated chat history
    for message in st.session_state.chat_history:
        if message["type"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

    # Get response from conversation chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history.append({"type": "bot", "content": response['answer']})

    # Display bot response
    st.write(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)

def extract_insights():
    if "interview_data" not in st.session_state:
        st.session_state.interview_data = []

    # Extract unique insights from text chunks
    for chunk in st.session_state.text_chunks:
        question = f"What is the unique insight for this section:\n{chunk}"
        response = st.session_state.conversation({'question': question})
        insight = response['answer']
        st.session_state.interview_data.append({"Unique Insight": insight})

def display_interview_table():
    if "interview_data" in st.session_state and st.session_state.interview_data:
        df = pd.DataFrame(st.session_state.interview_data)
        st.table(df)

def main():
    load_dotenv()
    st.set_page_config(page_title="thnkrAI", page_icon="favicon-transparent-256x256.png", layout="centered") 
    st.write(css, unsafe_allow_html=True)

    

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Understand Customer Interviews Better :rocket:")
    user_question = st.chat_input("Ask a question about your user interviews")

    if user_question:
        handle_userinput(user_question)

    if "raw_text" in st.session_state:
        extract_insights()
        display_interview_table()

    with st.sidebar:
        st.logo("logo-transparent-png (1).png")
        st.subheader("Your Interview Docs")
        pdf_docs = st.file_uploader(
            "Upload your interview notes or transcripts", accept_multiple_files=True)

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

if __name__ == '__main__':
    main()
