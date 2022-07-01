

def model_fn(model_dir):
    import os
    import pandas
    import logging
    import argparse
    import joblib


    ## Creating a logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info("Inference started.")
    
    
    
    print("Files are")
    dir_list = os.listdir("/opt/ml/model")
    print(dir_list)
    
    preprocessor = joblib.load(os.path.join(model_dir, dir_list[0]))
    return preprocessor







def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    import pandas
    from io import StringIO
    df = pandas.read_csv(StringIO(request_body))
    return df
#     if request_content_type == "text/csv":
#         df = pandas.read_csv(StringIO(request_body))
#         return df.drop(columns = ["Churn"])
#     else:
#         # Handle other content-types here or raise an Exception
#         # if the content type is not supported.
#         raise ValueError("{} not supported by script!".format(request_content_type))
    
    

    

# def output_fn(prediction, accept):
#     """Format prediction output

#     The default accept/content-type between containers for serial inference is JSON.
#     We also want to set the ContentType or mimetype as the same value as accept so the next
#     container can read the response payload correctly.
#     """
#     if accept == "application/json":
#         instances = []
#         for row in prediction.tolist():
#             instances.append({"features": row})

#         json_output = {"instances": instances}

#         return worker.Response(json.dumps(json_output), accept, mimetype=accept)
#     elif accept == 'text/csv':
#         return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
#     else:
#         raise RuntimeException("{} accept type is not supported by this script.".format(accept))
        
        





def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    # import xgboost as xgb
    from sklearn.metrics import accuracy_score
    import pathlib
    import json
    import os
    print(f"Input data columns: {input_data.columns}")
    # print(f"Model columns: {model.feature_names}")
    # features = model.predict(xgb.DMatrix(input_data.values))
    features = model.predict(input_data.drop(columns = ["Churn"]))
    
    accuracy = accuracy_score(input_data.Churn, features)
    model_data_s3_location = os.environ["MODELS3LOCATION"]
    model_name = os.environ["MODELNAME"]
    
    report_dict = {
        "metrics": {
            "accuracy": {
                "value": accuracy,
                # "standard_deviation": std
            },
        },
        "model_data":model_data_s3_location,
        "model_name":model_name
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    return report_dict

#     if label_column in input_data:
#         # Return the label (as the first column) and the set of features.
#         return np.insert(features, 0, input_data[label_column], axis=1)
#     else:
#         # Return only the set of features
#         return features

    
    
    

# if __name__ =='__main__':
#     model_fn(model_dir = "")
