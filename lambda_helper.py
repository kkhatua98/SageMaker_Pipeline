##
import json
import boto3


def lambda_handler(event, context):
    """ """
    sm_client = boto3.client("sagemaker")

    # The name of the model created in the Pipeline CreateModelStep
#     package_group = event["package_group"]
#     endpoint_name = event["endpoint_name"]
#     pipeline_input_bucket = event["pipeline_input_bucket"]
#     inference_code_name = event["inference_code_name"]
    
    pipeline_output_bucket = event["pipeline_output_bucket"]
    role = event["role"]
    pipeline_input_bucket = event["pipeline_input_bucket"]
    package_group = event[package_group]
    target_column = event[target_column]
    endpoint_name = event[endpoint_name]
    inference_code_name = event[inference_code_name]
    
    model_packages = client.list_model_packages(ModelPackageGroupName = package_group)
    latest_package = model_packages["ModelPackageSummaryList"][0]
    latest_package_arn = latest_package["ModelPackageArn"]
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
                            env = {"target_column":target_column, 
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