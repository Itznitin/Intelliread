from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import time

def main():
    load_dotenv()
    st.set_page_config(page_title="Intelliread")
    
    # Display the heading and subheading with default styling
    st.header("INTELLIREAD")
    st.subheader("Illuminating PDFs with Intelligent Answers")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        progress_bar = st.progress(0)
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            progress_bar.progress(33)

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=250,
            chunk_overlap=75,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        progress_bar.progress(66)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        progress_bar.progress(100)
        
        time.sleep(0.5)
        progress_bar.empty()

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            with st.spinner('Processing your query...'):
                docs = knowledge_base.similarity_search(user_question)
                
                
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question, prompt=docs[0])
                    print(cb)
                
                st.write(response)
    

if __name__ == '__main__':
    main()
