

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
    
    subprocess.run(["pip", "install", "-r", "/opt/ml/processing/input/requirements/preprocessing_requirements.txt"])
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
    parser.add_argument("--feature_importance_input_file_location", type = str)
    
    
        
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
        print(model_data_location)
        print(model_data_location[len(model_data_bucket)+6:-len(model_data_location.split('/')[-1])]+"output.tar.gz")
        subprocess.run(["tar", "-xvf", "output.tar.gz"])
        subprocess.run(["ls"])
        feature_importance = pd.read_csv("Feature_Importance.csv")
        feature_importance_values = feature_importance.iloc[0].tolist()
        feature_importance_column_names = feature_importance.columns.tolist()
        
        pmp_best = pd.read_csv("Metrics.csv")
        dates = pmp_best["Training_Date"].tolist()
        data_sets = pmp_best["Dataset"].tolist()
        metrics = pmp_best["Metric"].tolist()
        values = pmp_best["Value"].tolist()
        
        
        try:
            # metric_folder_contents = os.listdir(model_metric_input_location)
            # model_performance_metrics = pd.read_csv(f"{model_metric_input_location}/{metric_folder_contents[0]}")
            metric_bucket = model_metric_input_location[5:].split('/')[0]
            print(f"Model metric location: {model_metric_input_location}")
            s3.download_file(metric_bucket, model_metric_input_location[len(metric_bucket)+6:], model_metric_input_location.split('/')[-1])
            model_performance_metrics = pd.read_csv(model_metric_input_location.split('/')[-1])
            print(f"Model performance metric is {model_performance_metrics}")
        except:
            model_performance_metrics = pd.DataFrame([], columns = ["Performance_Training_Date","Dataset","Metric", "Value"])
        
        len_metrics = len(model_performance_metrics)
        increase_decrease = [0] * len(values)
        if len_metrics > 0:
            last_training_date = model_performance_metrics.iloc[-1,0]
            subset = model_performance_metrics.loc[model_performance_metrics["Performance_Training_Date"] == last_training_date, :]
            old_values = subset["Value"].tolist()
            increase_decrease = [(values[i] - old_values[i])/old_values[i] for i in range(len(values))]
        today = date.today()
        # model_performance_metrics.iloc[len_metrics, :] = [today, "accuracy", max_metric]
#         model_performance_metrics = model_performance_metrics.append({"Date":today, "Metric":objective_metric, "Metric Value":max_metric}, ignore_index = True)
        for i in range(len(dates)):
            model_performance_metrics = model_performance_metrics.append({"Performance_Training_Date":dates[i], "Dataset":data_sets[i], "Metric":metrics[i], "Value":values[i], "Increase_from_Previous":increase_decrease[i]}, ignore_index = True)
        
        # model_performance_metrics.to_csv("s3://churn-output-bucket/Training_Pipeline_Output/Model_Performance_Metrics.csv")
        model_performance_metrics.to_csv(f"{model_metric_output_location}/Model_Performance_Metrics.csv", index = False)
        
        
        try:
            # metric_folder_contents = os.listdir(model_metric_input_location)
            # model_performance_metrics = pd.read_csv(f"{model_metric_input_location}/{metric_folder_contents[0]}")
            feature_bucket = feature_importance_input_file_location[5:].split('/')[0]
            s3.download_file(feature_bucket, feature_importance_input_file_location[len(feature_bucket)+6:], feature_importance_input_file_location.split('/')[-1])
            feature_importance_records = pd.read_csv(feature_importance_input_file_location.split('/')[-1])
            print(f"Contents of feature importance file are {feature_importance_records}")
        except:
            feature_importance_records = pd.DataFrame([], columns = ["Feature_Training_Date", "Dataset", "Variable_Name", "Importance_Value"])
        
        len_records = len(feature_importance_records)
        today = date.today()
        # model_performance_metrics.iloc[len_metrics, :] = [today, "accuracy", max_metric]
        # feature_importance_records = feature_importance_records.append({column:value for column, value in zip(feature_importance_column_names, feature_importance_values)}, ignore_index = True)
        for i in range(len(feature_importance_column_names)):
            feature_importance_records=feature_importance_records.append({"Feature_Training_Date":today, "Dataset":"Training", "Variable_Name":feature_importance_column_names[i], "Importance_Value":feature_importance_values[i]}, ignore_index = True)
        
        feature_importance_records.to_csv(f"{feature_importance_output_file_location}/Feature_Importance.csv", index = False)
        
        # model_performance_metrics.to_csv("s3://churn-output-bucket/Training_Pipeline_Output/Model_Performance_Metrics.csv")
        model_performance_metrics.to_csv(f"{model_metric_output_location}/Model_Performance_Metrics.csv", index = False)
        
        
        
        #### Writing confusion matrix
        ## Reading new data
        matrix = pd.read_csv("Confusion_Matrix.csv")
        dates = matrix["Confusion_Date"].tolist()
        data_sets = matrix["Data"].tolist()
        tn = matrix["TN"].tolist()
        fp = matrix["FP"].tolist()
        fn = matrix["FN"].tolist()
        tp = matrix["TP"].tolist()
        
        ## Downloading old data
        try:
            feature_bucket = feature_importance_input_file_location[5:].split('/')[0]
            print(f"Confusion Matrid location: {feature_importance_input_file_location}")
            s3.download_file(feature_bucket, "Training_Pipeline_Output/Confusion_Matrix.csv", "Confusion_Matrix.csv")
            old_matrix = pd.read_csv("Confusion_Matrix.csv")
        except:
            old_matrix = pd.DataFrame([], columns = ["Confusion_Date", "Data", "TN", "FP", "FN", "TP"])
        
        ## Appending new data to old data
        for i in range(len(dates)):
            old_matrix = old_matrix.append({"Confusion_Date":dates[i], "Data":data_sets[i], "TN":tn[i], "FP":fp[i], "FN":fn[i], "TP":tp[i]}, ignore_index = True)
            
        ## Writing appended data
        old_matrix.to_csv(f"/opt/ml/processing/confusion_matrix/Confusion_Matrix.csv", index = False)
        
        
        
        
        #### Combining all dashboard data in one file
        df_concat = pd.concat([feature_importance_records, model_performance_metrics, old_matrix], axis=1)
        ## Writing appended data
#         df_concat.to_csv(f"/opt/ml/processing/Combined/Combined_Data.csv", index = False)
        
        
        
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
