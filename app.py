import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import os
from botocore.config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain.prompts import PromptTemplate
import boto3
from langchain_aws import ChatBedrock

# Configuring Boto3
retry_config = Config(
    region_name=os.environ.get("region_name"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id=os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key=os.environ.get("aws_secret_access_key"),
                                       config=retry_config)

# Define the Streamlit app
def main():
    st.title("Document QA with RAG System")

    # Input for S3 path
    s3_path = st.text_input("Enter the S3 path of the PDF document:")
    
    # Process document button
    if st.button("Process Document"):
        try:
            # Load the document from S3
            loader = AmazonTextractPDFLoader(s3_path)
            document = loader.load()
            
            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=0
            )
            texts = text_splitter.split_documents(document)
            
            # Create embeddings and vector store
            embeddings = BedrockEmbeddings(client=bedrock, model_id=os.environ.get("EMBEDDING_MODEL"))
            db = FAISS.from_documents(documents=texts, embedding=embeddings)
            
            retriever = db.as_retriever(search_type='mmr', search_kwargs={"k": 3})
            
            template = """
            Answer the question as truthfully as possible strictly using only the provided text, and if the answer is not contained within the text, say "I don't know". Skip any preamble text and reasoning and give just the answer.
            <text>{context}</text>
            <question>{question}</question>
            <answer>"""
            
            qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            
            chain_type_kwargs = {"prompt": qa_prompt, "verbose": False}
            
            bedrock_llm = ChatBedrock(client=bedrock, model_id=os.environ.get("BEDROCK_MODEL"))
            qa = RetrievalQA.from_chain_type(
                llm=bedrock_llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs,
                verbose=False
            )
            
            st.session_state.qa = qa
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {e}")

    # Input for question
    question = st.text_input("Enter your question:")
    
    # Process question button
    if st.button("Get Answer"):
        if 'qa' in st.session_state and question:
            try:
                # Get the answer to the question
                result = st.session_state.qa.invoke(question)
                st.write(result["result"])
            except Exception as e:
                st.error(f"Error getting answer: {e}")
        else:
            st.error("Please process the document first and enter a question.")

if __name__ == "__main__":
    main()
