## Setting the session
import boto3
import sagemaker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pipeline_output_bucket', type=str)
parser.add_argument('--role', type=str)
parser.add_argument('--pipeline_input_bucket', type=str)
parser.add_argument('--package_group', type=str)
parser.add_argument('--target_column', type=str)
parser.add_argument('--endpoint_name', type=str)
parser.add_argument('--inference_code_name', type=str)



args, _ = parser.parse_known_args()
pipeline_output_bucket = args.pipeline_output_bucket
role = args.role
pipeline_input_bucket = args.pipeline_input_bucket
package_group = args.package_group
target_column = args.target_column
endpoint_name = args.endpoint_name
inference_code_name = args.inference_code_name

region = boto3.Session().region_name
# pipeline_output_bucket = build_parameters["output_bucket"]   ##############################
sagemaker_session = sagemaker.session.Session(default_bucket = pipeline_output_bucket)
# sagemaker_session = sagemaker.session.Session()
# role = sagemaker.get_execution_role()
# role = "arn:aws:iam::852619674999:role/service-role/AmazonSageMaker-ExecutionRole-20220427T124311"  ######################




## Handling the input
# pipeline_input_bucket = build_parameters["input_bucket"]      #####################
input_feature_selection_file_uri = f"s3://{pipeline_input_bucket}/Feature_Selection.csv"


#### Obtaining the model from Sagemaker model registry.
# package_group = build_parameters["model_package_group_name"]      ###################

import boto3
client = boto3.client('sagemaker')
model_packages = client.list_model_packages(ModelPackageGroupName = package_group)


latest_package = model_packages["ModelPackageSummaryList"][0]
latest_package_arn = latest_package["ModelPackageArn"]

print(latest_package)
print(latest_package_arn)


latest_package_details = client.describe_model_package(ModelPackageName=latest_package_arn)




from sagemaker.model import Model
inference_model = Model(image_uri = latest_package_details['InferenceSpecification']['Containers'][0]['Image'], 
                        source_dir = f"s3://{pipeline_input_bucket}/codes/inference.tar.gz",
                        # source_dir = build_parameters["single_model_evluation_source_dir"],
                        entry_point = inference_code_name,
                        # entry_point="inference.py", 
                        model_data = latest_package_details['InferenceSpecification']['Containers'][0]["ModelDataUrl"], 
                        role = role,
                        sagemaker_session = sagemaker_session,
                        env = {"target_column":target_column,    #############
                               "feature_selection_file_location":input_feature_selection_file_uri,
                               "log_location":"/opt/ml/processing/logss"
                              }
                       )



try:
    predictor = inference_model.deploy(instance_type="ml.c4.xlarge", initial_instance_count=1,
                                   endpoint_name = endpoint_name,
                                   update_endpoint = True
                                  )
except:
    inference_model.create(instance_type = "ml.c4.xlarge")
    model_name = inference_model.name
    import random
    config_name = str(random.random())[2:]
    sagemaker_session.create_endpoint_config(
                                name=config_name,
                                model_name=model_name,
                                initial_instance_count=1,
                                instance_type='ml.m4.xlarge'
                                )
    client.update_endpoint(EndpointName=endpoint_name,
                           EndpointConfigName=config_name
                          )