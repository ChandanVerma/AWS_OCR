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
## Step 2: Create Conda environment
Run the following command to create conda environment:

```bash
conda create -n aws_ocr python==3.10 -y
```

## Step 3: Activate Conda environment
Run the following command to activate conda environment:

```bash
conda activate aws_ocr
```

## Step 4: Install Dependencies
Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Step 5: Run the Streamlit App
Start the Streamlit app with the following command:

```
streamlit run app.py
```

## FOR DOCKER LAMBDA
```
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 519183793155.dkr.ecr.ap-south-1.amazonaws.com

docker build --platform linux/amd64 -t aws-ocr-lambda-image:latest .

docker tag aws-ocr-lambda-image:latest 519183793155.dkr.ecr.ap-south-1.amazonaws.com/aws-ocr-lambda-image:latest

docker push 519183793155.dkr.ecr.ap-south-1.amazonaws.com/aws-ocr-lambda-image:
```

