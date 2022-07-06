

def preprocessing_function():
    import os
    import logging
    import traceback
    import argparse
    import numpy as np
    import pandas as pd
    import json
    import shutil
    import boto3
    from datetime import date
    import subprocess
    import sys
    
    subprocess.run(["pip", "install", "-r", "/opt/ml/processing/input/code/preprocessing_requirements.txt"])
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "fsspec"])


    
    print(os.environ.get('SM_CHANNEL_TEST'))


    ## Creating a logger.
    # logger = logging.getLogger()
    # logging.captureWarnings(True)
    # logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler())
    # logger.info("Preprocessing started.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--final_model_location', type=str, default="/opt/ml/processing/final_model")
    parser.add_argument('--logs_location', type=str, default="/opt/ml/processing/logs")
    parser.add_argument('--model_metric_input_location', type=str, default="/opt/ml/processing/metrics")
    parser.add_argument('--model_metric_output_location', type=str, default="/opt/ml/processing/metrics_folder")
    parser.add_argument('--input_folder', type=str, default="/opt/ml/processing/input")
    parser.add_argument('--objective_metric', type=str, default="accuracy")
    parser.add_argument('--property_file_location', type = str, default = "/opt/ml/processing/evaluation")
    parser.add_argument("--feature_importance_output_file_location", type = str, default = "/opt/ml/processing/feature_importance")
    
    
        
    args, _ = parser.parse_known_args()
    final_model_location = args.final_model_location
    logs_location = args.logs_location
    model_metric_input_location = args.model_metric_input_location
    model_metric_output_location = args.model_metric_output_location
    input_folder = args.input_folder
    objective_metric = args.objective_metric
    property_file_location = args.property_file_location
    feature_importance_output_file_location = args.feature_importance_output_file_location
    feature_importance_input_file_location = args.feature_importance_input_file_location
    
    
    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f'{logs_location}/logfile.log')
    logger.addHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    
    


    try:
        folders = os.listdir(input_folder)
        metrics = []
        for folder in folders:
            if "model" in folder:
                folder_contents = os.listdir(f"{input_folder}/{folder}")
                for folder_content in folder_contents:
                    if "csv.out" in folder_content:
                        evaluation_file_name = folder_content
                with open(f"{input_folder}/{folder}/{evaluation_file_name}") as file:
                    file_content = json.load(file)
                    metric_value = file_content["metrics"][f"{objective_metric}"]["value"]
                    metrics.append(metric_value)
        max_metric = max(metrics)
        max_metric_index = metrics.index(max_metric)
        # original = f"/opt/ml/processing/input/model{max_metric_index}/model.tar.gz"
        # target = "/opt/ml/processing/train/model.tar.gz"
        
        folder_contents = os.listdir(f"{input_folder}/model{max_metric_index}")
        for folder_content in folder_contents:
            if "csv.out" in folder_content:
                evaluation_file_name = folder_content
        with open(f"{input_folder}/model{max_metric_index}/{evaluation_file_name}") as file:
            file_content = json.load(file)
            model_data_location = file_content["model_data"]
            best_model_name = file_content["model_name"]
        
        model_data_bucket = model_data_location[5:].split('/')[0]

        # shutil.copyfile(original, target)
        s3 = boto3.client ('s3')
        s3.download_file(model_data_bucket, model_data_location[len(model_data_bucket)+6:], f"{final_model_location}/{model_data_location.split('/')[-1]}")
        s3.download_file(model_data_bucket, model_data_location[len(model_data_bucket)+6:-len(model_data_location.split('/')[-1])]+"output.tar.gz", f"output.tar.gz")
        subprocess.run(["tar", "-xvf", "output.tar.gz"])
        feature_importance = pd.read_csv("Feature_Importance.csv")
        feature_importance_values = feature_importance.iloc[0].tolist()
        feature_importance_column_names = feature_importance.columns.tolist()
        
        
        try:
            # metric_folder_contents = os.listdir(model_metric_input_location)
            # model_performance_metrics = pd.read_csv(f"{model_metric_input_location}/{metric_folder_contents[0]}")
            model_performance_metrics = pd.read_csv(model_metric_input_location)
        except:
            model_performance_metrics = pd.DataFrame([], columns = ["Date", "Metric", "Metric Value"])
        
        len_metrics = len(model_performance_metrics)
        today = date.today()
        # model_performance_metrics.iloc[len_metrics, :] = [today, "accuracy", max_metric]
        model_performance_metrics = model_performance_metrics.append({"Date":today, "Metric":objective_metric, "Metric Value":max_metric}, ignore_index = True)
        
        # model_performance_metrics.to_csv("s3://churn-output-bucket/Training_Pipeline_Output/Model_Performance_Metrics.csv")
        model_performance_metrics.to_csv(f"{model_metric_output_location}/Model_Performance_Metrics.csv", index = False)
        
        
        try:
            # metric_folder_contents = os.listdir(model_metric_input_location)
            # model_performance_metrics = pd.read_csv(f"{model_metric_input_location}/{metric_folder_contents[0]}")
            feature_importance_records = pd.read_csv(feature_importance_input_file_location)
        except:
            feature_importance_records = pd.DataFrame([], columns = feature_importance_column_names)
        
        len_records = len(feature_importance_records)
        today = date.today()
        # model_performance_metrics.iloc[len_metrics, :] = [today, "accuracy", max_metric]
        feature_importance_records = feature_importance_records.append({column:value for column, value in zip(feature_importance_column_names, feature_importance_values)}, ignore_index = True)
        
        feature_importance_records.to_csv(f"{feature_importance_output_file_location}/Feature_Importance.csv", index = False)
        
        # model_performance_metrics.to_csv("s3://churn-output-bucket/Training_Pipeline_Output/Model_Performance_Metrics.csv")
        model_performance_metrics.to_csv(f"{model_metric_output_location}/Model_Performance_Metrics.csv", index = False)
        
        
        report_dict = {"best_model_name":best_model_name}
        
        #output_dir = "/opt/ml/processing/evaluation"
        #pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
        evaluation_path = f"{property_file_location}/property_file.json"
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))
        
        
        logger.removeHandler(handler)
        handler.close()
    
    except:
        var = traceback.format_exc()
        logger.error(var)
        
        logger.removeHandler(handler)
        handler.close()



if __name__ == "__main__":
    preprocessing_function()
