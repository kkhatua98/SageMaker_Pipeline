
def main(event, context):
    import boto3
    import sagemaker
    import os
    region = boto3.Session().region_name
    sage_client = boto3.Session().client("sagemaker")

    ## You must have already run a hyperparameter tuning job to analyze it here.
    ## The Hyperparameter tuning jobs you have run are listed in the Training section on your SageMaker dashboard.
    ## Copy the name of a completed job you want to analyze from that list.
    ## For example: tuning_job_name = 'mxnet-training-201007-0054'.
    tuning_job_name = event["tuning_job_name"]


    # Get the results from hyper parameter tuning job
    tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
        )

    # Get the best training job's metric value
    best_metric_value = tuning_job_result["BestTrainingJob"]["FinalHyperParameterTuningJobObjectiveMetric"]["Value"]


    # Getting best training job's model location
    best_training_job_name = tuning_job_result["BestTrainingJob"]["TrainingJobName"]
    best_training_job_result = sage_client.describe_training_job(TrainingJobName=best_training_job_name)
    best_model_location = best_training_job_result['ModelArtifacts']["S3ModelArtifacts"]



    # The following few lines will be necessary if we are trying to keep show all the results of the hyper parameter tuning job.
    objective = tuning_job_result["HyperParameterTuningJobConfig"]["HyperParameterTuningJobObjective"]
    is_minimize = objective["Type"] != "Maximize"

    tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
    full_df = tuner.dataframe()

    df = full_df[full_df["FinalObjectiveValue"] > -float("inf")]
    if len(df) > 0:
        df = df.sort_values("FinalObjectiveValue", ascending=is_minimize)
    df.to_csv()
    
    
    return {"best_model_location":best_model_location, "best_metric_value":best_metric_value}