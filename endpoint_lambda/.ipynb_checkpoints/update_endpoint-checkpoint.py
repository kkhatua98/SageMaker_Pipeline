##
def handler_name(event, context):
    import subprocess
    subprocess.run(["pip", "install", "boto3", "sagemaker"])
    ## Setting the session
    import boto3
    import sagemaker
    import json
    
    ## Loading the configurations from config.json file.
    with open("config.json") as file:
        build_parameters = json.load(file)
    
    
    ## Initiating session
    pipeline_output_bucket = build_parameters["output_bucket"] 
    sagemaker_session = sagemaker.session.Session(default_bucket = pipeline_output_bucket)
    
    
    ## Extracting message from event
    message = json.loads(event['Records'][0]['Sns']['Message'])
    message_type_subject_mapper = {"CodeCommit Pull Request State Change":"New pull request on repository ",
                                   "CodeCommit Repository State Change":"Codes updated in repository "}
    message_type = message["detail-type"]
    message_time = message["time"]
    
    if message_type == "SageMaker Model Package State Change":
        model_package_group = message["detail"]["ModelPackageGroupName"]
        model_package_version = message["detail"]["ModelPackageVersion"]
        model_approval_status = message["detail"]["ModelApprovalStatus"]
        
        
        if model_approval_status == "Approved":
            client = boto3.client('sagemaker')
            
            model_packages = client.list_model_packages(ModelPackageGroupName = model_package_group)
            latest_package = model_packages["ModelPackageSummaryList"][0]
            latest_package_arn = latest_package["ModelPackageArn"]
            latest_package_details = client.describe_model_package(ModelPackageName=latest_package_arn)
            
            from sagemaker.model import Model
            inference_model = Model(image_uri = latest_package_details['InferenceSpecification']['Containers'][0]['Image'], 
                                    
                                    ## -------- ##
                                    source_dir = f"s3://{build_parameters['input_bucket']}/codes/endpoint_scoring.tar.gz",
                                    entry_point = f"{build_parameters['endpoint_scoring_code_location'].split('/')[-1]}",
                                    # entry_point="../" + build_parameters["scoring_code_loaction"], 
                                    ## -------- ##
                                    
                                    model_data = latest_package_details['InferenceSpecification']['Containers'][0]["ModelDataUrl"], 
                                    role = role,
                                    sagemaker_session = sagemaker_session,
                                    
                                    env = {"target_column":build_parameters["target_column"],
                                           "feature_selection_file_location":f"s3://{build_parameters['input_bucket']}/Feature_Selection.csv",
                                           "log_location":"/opt/ml/processing/logss"
                                          }
                                   )
            
            try:
                predictor = inference_model.deploy(instance_type="ml.c4.xlarge", 
                                                   initial_instance_count=1,
                                                   endpoint_name = build_parameters["endpoint_name"],
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
                client.update_endpoint(EndpointName = build_parameters["endpoint_name"],
                                       EndpointConfigName=config_name
                                      )
            