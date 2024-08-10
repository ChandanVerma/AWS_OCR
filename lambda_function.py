import boto3
from langchain import hub
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock, BedrockEmbeddings
from botocore.config import Config
import os

retry_config = Config(
    region_name=os.environ.get("region_name"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id = os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key = os.environ.get("aws_secret_access_key"),
                                       config=retry_config)

embeddings = BedrockEmbeddings(client=bedrock, model_id=os.environ.get("EMBEDDING_MODEL"))

def load_data(s3_path):
    # loader = S3FileLoader(
    # "testing-hwc", "fake.docx", aws_access_key_id="xxxx", aws_secret_access_key="yyyy")
    loader = AmazonTextractPDFLoader(s3_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def get_response(s3_path, query):
    prompt = hub.pull("rlm/rag-prompt")
    bedrock_llm = ChatBedrock(client=bedrock, model_id=os.environ.get("BEDROCK_MODEL"))
    retriever = load_data(s3_path)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | bedrock_llm
        | StrOutputParser()
    )
    return rag_chain.invoke(query)

def lambda_handler(event, context):
    query = event.get("question")
    s3_path = event.get("s3_path")
    response = get_response(s3_path, query)
    print("response:", response)
    return {"body": response, "statusCode": 200}