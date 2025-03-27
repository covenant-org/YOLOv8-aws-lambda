import boto3
from botocore.exceptions import NoCredentialsError
import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

def upload_to_s3(local_file, bucket, s3_file):
    # Get the AWS credentials from the credentials.txt file
    with open(os.path.join(script_dir, 'credentials.txt'), 'r') as file:
        access_key = file.readline().strip()
        secret_key = file.readline().strip()
        region_name = file.readline().strip()
    # Access the S3 client
    s3 = boto3.client(
    's3',     
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region_name
    )
    try:
        # Upload the file
        s3.upload_file(local_file, bucket, s3_file)
        print(f"Upload Successful: {s3_file}")
        return True
    except FileNotFoundError:
        print("The file was not found.")
    except NoCredentialsError as e:
        print("Credentials not available.")
        print(e)
    return False
