# AWS_OCR

### CREATE .env in the same folder as app.py
aws_access_key_id = 'you access key'
aws_secret_access_key = 'your secret'
region_name='your region'
BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
EMBEDDING_MODEL = "amazon.titan-embed-image-v1"

pip install -r requirements.txt

streamlit run app.py
