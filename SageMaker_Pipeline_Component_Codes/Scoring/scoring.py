import pandas as pd
import os
data_folder = "/opt/ml/processing/input/data"
model_folder = "/opt/ml/processing/input/model_folder"

data_folder_contents = os.listdir(data_folder)
for content in data_folder_contents:
    if ".csv" in content:
        data_file_name = content
scoring_data = pd.read_csv(f"/opt/ml/processing/input/data/{data_file_name}")

dir_list = os.listdir("/opt/ml/processing/input/model_folder")
    for content in dir_list:
        if "joblib" in content:
            file_name = content
print(dir_list)
preprocessor = joblib.load(os.path.join("/opt/ml/processing/input/model_folder", file_name))

predictions = preprocessor.predict(scoring_data)
predictions.to_csv("/opt/ml/processing/train/Predictions.csv")