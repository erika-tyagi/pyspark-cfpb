import boto3
import pandas as pd


def load_and_store(): 
    # Create client 
    s3 = boto3.client('s3', 
                      region_name='us-east-1')

    # Create bucket 
    bucket = s3.create_bucket(Bucket='macs-30123-final-proj-tyagie')

    # Read complaints data as CSV 
    complaints = pd.read_csv('http://files.consumerfinance.gov/ccdb/complaints.csv.zip')

    # Clean up column names 
    complaints.columns = [col.lower().replace(' ', '_') for col in 
                          complaints.columns]

    # Save complaints data as a parquet file 
    complaints.to_parquet('complaints.parquet')

    # Upload to S3 bucket 
    s3.upload_file(Filename='complaints.parquet', 
                   Bucket='macs-30123-final-proj-tyagie', 
                   Key='complaints.parquet', 
                   ExtraArgs={'ACL':'public-read'})

    url = "https://{}.s3.amazonaws.com/{}".format(
        'macs-30123-final-proj-tyagie', 'complaints.parquet') 

    print(url)


if __name__ == "__main__":
    
    load_and_store() 





