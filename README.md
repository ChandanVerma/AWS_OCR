# Setup Instructions

## Step 1: Create a `.env` file
Create a `.env` file in the same folder as `app.py` and add the following content:

```
aws_access_key_id = 'your access key'
aws_secret_access_key = 'your secret'
region_name='your region'
BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
EMBEDDING_MODEL = "amazon.titan-embed-image-v1" 
```

## Step 2: Install Dependencies
Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Step 3: Run the Streamlit App
Start the Streamlit app with the following command:

```
streamlit run app.py
```
