import subprocess
import json

## Loading the configurations from config.json file.
import json
with open("config.json") as file:
    build_parameters = json.load(file)

# subprocess.run(["mkdir", "tmp_targz"])

n_models = build_parameters["number_of_models"]
for i in range(n_models):
    model_specification = build_parameters['model_specifications'][f"model{i}"]
    subprocess.run(["cp", f"SageMaker_Pipeline_Component_Codes/Training/{model_specification['entry_point']}", "tmp_targz"])
    subprocess.run(["cp", f"Requirements/{model_specification['dependencies']}", "tmp_targz"])
    subprocess.run(["tar", "-czvf", f"{model_specification['model_name']}.tar.gz", "-C", "tmp_targz", f"{model_specification['entry_point']}", f"{model_specification['dependencies']}"])
    subprocess.run(["aws", "s3", "cp", f"{model_specification['model_name']}.tar.gz", f"s3://{build_parameters['input_bucket']}/codes/"])

subprocess.run(["cp", f"SageMaker_Pipeline_Component_Codes/Training/{build_parameters['single_model_evluation_entry_point']}", "tmp_targz"])
subprocess.run(["tar", "-czvf", "evaluation.tar.gz", "-C", "tmp_targz", f"{build_parameters['single_model_evluation_entry_point']}"])
subprocess.run(["aws", "s3", "cp", f"evaluation.tar.gz", f"s3://{build_parameters['input_bucket']}/codes/"])
subprocess.run(["aws", "s3", "cp", f"SageMaker_Pipeline_Component_Codes/Training/{build_parameters['processing_code_file_name']}", f"s3://{build_parameters['input_bucket']}/codes/"])
subprocess.run(["aws", "s3", "cp", f"SageMaker_Pipeline_Component_Codes/Training/{build_parameters['get_best_model_code_file_name']}", f"s3://{build_parameters['input_bucket']}/codes/"])
subprocess.run(["aws", "s3", "cp", f"{build_parameters['scoring_preprocessing_code_location']}", f"s3://{build_parameters['input_bucket']}/codes/"])

subprocess.run(["cp", f"{build_parameters['scoring_code_location']}", "tmp_targz"])
subprocess.run(["tar", "-czvf", f"scoring.tar.gz", "-C", "tmp_targz", f"{build_parameters['scoring_code_location'].split('/')[-1]}"])
subprocess.run(["aws", "s3", "cp", "scoring.tar.gz", f"s3://{build_parameters['input_bucket']}/codes/"])

subprocess.run(["cp", f"{build_parameters['endpoint_scoring_code_location']}", "tmp_targz"])
subprocess.run(["tar", "-czvf", f"endpoint_scoring.tar.gz", "-C", "tmp_targz", f"{build_parameters['endpoint_scoring_code_location'].split('/')[-1]}"])
subprocess.run(["aws", "s3", "cp", "endpoint_scoring.tar.gz", f"s3://{build_parameters['input_bucket']}/codes/"])

subprocess.run(["cp", f"{build_parameters['lambda_code_location']}", "."])
subprocess.run(["zip", '-r', "lambda_codes.zip", f"{build_parameters['lambda_code_location'].split('/')[-1]}", "config.json"])
subprocess.run(["aws", "s3", "cp", "lambda_codes.zip", f"s3://{build_parameters['input_bucket']}/codes/"])

from sagemaker import get_execution_role

role = get_execution_role()
print(role)

lambda_function_name = build_parameters["lambda_function_name"]
import boto3
client = boto3.client('lambda')
response = client.list_functions()
functions = [func["FunctionName"] for func in response["Functions"]]

if lambda_function_name not in functions:
    response = client.create_function(
        Code={
            'S3Bucket': build_parameters["input_bucket"],
            'S3Key': '/codes/lambda_codes.zip',
        },
        Description='Update churn scoring endpoint',
        FunctionName=build_parameters["lambda_function_name"],
        Handler='update_endpoint.handler_name',
        Publish=True,
        Role='arn:aws:iam::123456789012:role/lambda-role',
        Runtime='python3.7'
    )