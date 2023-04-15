import pandas as pd
import os
import subprocess
import joblib

data_folder = "/opt/ml/processing/input/data"
model_folder = "/opt/ml/processing/input/model_folder"

data_folder_contents = os.listdir(data_folder)
for content in data_folder_contents:
    if ".csv" in content:
        data_file_name = content
scoring_data = pd.read_csv(f"/opt/ml/processing/input/data/{data_file_name}")

dir_list = os.listdir(model_folder)
for content in dir_list:
    if "tar.gz" in content:
        zip_file_name = content
subprocess.run(["cp", f"{model_folder}/{zip_file_name}", '.'])
subprocess.run(["tar", "-xf", f"{zip_file_name}"])
print(dir_list)

dir_list = os.listdir('.')
for content in dir_list:
    if "joblib" in content:
        file_name = content
preprocessor = joblib.load(f"{file_name}")

predictions = preprocessor.predict(scoring_data)
pd.DataFrame(predictions, columns=["Prediction"]).to_csv("/opt/ml/processing/train/Predictions.csv", index = False)