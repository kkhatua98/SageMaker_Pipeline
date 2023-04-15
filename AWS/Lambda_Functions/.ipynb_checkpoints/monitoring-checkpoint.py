import json
import boto3


def lambda_handler(event, context):
    """ """
    sm_client = boto3.client("sagemaker")

    # The name of the model created in the Pipeline CreateModelStep
    performance_file_location = event["performance_file_location"]
    sns_topic_arn = event["sns_topic_name"]
    
    
    s3_client = boto3.client('s3')
    
    # Download the file from S3
    bucket_name = performance_file_location.split('/')[2]
    file_name = performance_file_location[len(bucket_name) + 6: ]
    s3_client.download_file(bucket_name, file_name, "Monitor.csv")
    
    headers = subprocess.check_output(['head', '-1', "Monitor.csv"]).decode('utf-8').split(',')
    values = list(map(float, subprocess.check_output(['tail', '-1', "Monitor.csv"]).decode('utf-8').split(',')))
    
    mail_content = {header:value for header, value in zip(headers, values)}
    
    
    import datetime
    snsClient = boto3.client("sns")
    response = snsClient.publish(
        TopicArn = sns_topic_arn, 
        # Message = json.dumps(message_sns),
        Message = mail_content,
        Subject = f"Model Performance Metrics for {str(datetime.datetime.today()).split(' ')[0]}"
    )
    print(response)

#     return {
#         "statusCode": 200,
#         "body": json.dumps("Created Endpoint!"),
#         "other_key": "example_value",
#     }