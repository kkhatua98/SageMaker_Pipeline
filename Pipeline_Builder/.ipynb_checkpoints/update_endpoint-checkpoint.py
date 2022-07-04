## Setting the session
import boto3
import sagemaker

region = boto3.Session().region_name
pipeline_output_bucket = build_parameters["output_bucket"]   ##############################
sagemaker_session = sagemaker.session.Session(default_bucket = pipeline_output_bucket)
# sagemaker_session = sagemaker.session.Session()
# role = sagemaker.get_execution_role()
role = "arn:aws:iam::852619674999:role/service-role/AmazonSageMaker-ExecutionRole-20220427T124311"  ######################




## Handling the input
pipeline_input_bucket = build_parameters["input_bucket"]      #####################
input_feature_selection_file_uri = f"s3://{pipeline_input_bucket}/Feature_Selection.csv"


#### Obtaining the model from Sagemaker model registry.
package_group = build_parameters["model_package_group_name"]      ###################

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
                        entry_point="inference.py", 
                        model_data = latest_package_details['InferenceSpecification']['Containers'][0]["ModelDataUrl"], 
                        role = role,
                        sagemaker_session = sagemaker_session,
                        env = {"target_column":"Churn",    #############
                               "feature_selection_file_location":input_feature_selection_file_uri,
                               "log_location":"/opt/ml/processing/logss"
                              }
                       )



try:
    predictor = inference_model.deploy(instance_type="ml.c4.xlarge", initial_instance_count=1,
                                   endpoint_name = "sagemaker-scikit-learn-2022-07-04-06-26-01-239",
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
    client.update_endpoint(EndpointName='sagemaker-scikit-learn-2022-07-04-06-26-01-239',
                           EndpointConfigName=config_name
                          )