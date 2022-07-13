import boto3
region = boto3.Session().region_name

s3 = boto3.resource('s3')
input_bucket = s3.Bucket("mlinput-churn-telecomchurn-852619674999-ap-south-1")

import sagemaker
sagemaker_session = sagemaker.session.Session()

for bucket_object in input_bucket.objects.filter(Prefix='models/'):
    models = bucket_object.key.split('/')[-1]
# mybucket.objects.filter(Prefix='foo/bar')


## Loading the configurations from config.json file.
import json
with open("../config.json") as file:
    build_parameters = json.load(file)
role = "arn:aws:iam::852619674999:role/service-role/AmazonSageMaker-ExecutionRole-20220427T124311"

from sagemaker import image_uris
sklearn_image_uri = image_uris.retrieve(framework='sklearn', region=region, version='0.23-1')

from sagemaker.model import Model
if build_parameters["given_model_type"] == "sklearn":
    given_model = Model(image_uri = sklearn_image_uri, 
                        # source_dir = f"s3://{pipeline_input_bucket}/codes/evaluation.tar.gz",
                        # source_dir = build_parameters["single_model_evluation_source_dir"],
                        # entry_point = build_parameters["single_model_evluation_entry_point"],
                        model_data = "s3://churn-input-bucket-us-east-1/model.tar.gz", 
                        role = role,
                        sagemaker_session = sagemaker_session
                        )
    
    given_model.register(content_types=["text/csv"],
                         response_types=["text/csv"],
                         inference_instances=[build_parameters["scoring_instance_type"]],
                         transform_instances=[build_parameters["scoring_instance_type"]],
                         model_package_group_name = build_parameters["model_package_group_name"],
                         image_uri = sklearn_image_uri,
                         # approval_status="Approved",
                         # role=role
                        )