
def monitoring_function():
    import pandas as pd 
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score,roc_auc_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_curve
    import glob
    import subprocess
    import sys
    
    subprocess.run(["pip", "install", "fsspec==2022.5.0", "s3fs==2022.5.0"])
    import s3fs
    s3 = s3fs.S3FileSystem(anon=False)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_actual_location', type=str)
    parser.add_argument('--y_predicted_location', type=str)
    parser.add_argument("--metrics_input_location", type = str)
    parser.add_argument("--metrics_output_location", type = str)
    args, _ = parser.parse_known_args()

    y_actual = pd.read_csv(glob.glob(f"{args.y_actual_location}/*.csv")[0]).loc[:, "Churn"]
    y_predicted_s3 = f"s3://{s3.glob(args.y_predicted_location)[0]}"
    # y_predicted = pd.read_csv(glob.glob(f"{args.y_predicted_location}/*.csv")).loc[:, "Prediction"]
    y_predicted = pd.read_csv(y_predicted_s3)

    tn, fp, fn, tp = confusion_matrix(y_actual,y_predicted).ravel()
    accuracy = round(100*accuracy_score(y_actual,y_predicted),2)
    precision =  round(100*precision_score(y_actual,y_predicted,zero_division=0),2)
    recall =  round(100*recall_score(y_actual,y_predicted),2)
    specificity = tn / (tn + fp)
    specificity =  round(100*specificity,2)
    f1 = round(100*f1_score(y_actual, y_predicted),2)
    # roc_auc = roc_auc_score(y_actual, y_pred_proba)

    try:
        metrics_df = pd.read_csv(f"{args.metrics_input_location}")
    except:
        metrics_df = pd.DataFrame([], columns = ["tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "specificity", "f1"])

    metrics_df = metrics_df.append({"tn":tn, "fp":fp, "fn":fn, "tp":tp, "accuracy":accuracy, "precision":precision, "recall":recall, "specificity":specificity, "f1":f1}, ignore_index = True)
        
    metrics_df.to_csv(f"{args.metrics_output_location}/Monitor.csv", index = False)
    
    import json
    mail_content = json.dumps({"tn":str(tn), "fp":str(fp), "fn":str(fn), "tp":str(tp), "accuracy":str(accuracy), "precision":str(precision), "recall":str(recall), "specificity":str(specificity), "f1":str(f1)})
    
    
    import datetime
    import boto3
    snsClient = boto3.client("sns", region_name = "ap-south-1")
    response = snsClient.publish(
        TopicArn = "arn:aws:sns:ap-south-1:852619674999:Approvals", 
        # Message = json.dumps(message_sns),
        Message = mail_content,
        Subject = f"Model Performance Metrics for {str(datetime.datetime.today()).split(' ')[0]}"
    )
    
    
    if accuracy < 100:
        sagemakerClient = boto3.client('sagemaker', region_name = "ap-south-1")
        now = datetime.datetime.now()
        response = sagemakerClient.start_pipeline_execution(
            PipelineName='churn-training',
            PipelineExecutionDisplayName=f'Demo{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}'
        )
    
    return tn, fp, fn, tp,accuracy,precision,recall,specificity,f1
    # , roc_auc


if __name__ =='__main__':
    monitoring_function()