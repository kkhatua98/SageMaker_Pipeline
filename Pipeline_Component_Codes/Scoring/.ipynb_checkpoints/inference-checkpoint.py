

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
    for content in dir_list:
        if "joblib" in content:
            file_name = content
    print(dir_list)
    
    # preprocessor = joblib.load(os.path.join(model_dir, dir_list[0]))
    preprocessor = joblib.load(os.path.join(model_dir, file_name))
    return preprocessor







def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    import pandas
    from io import StringIO
    if request_content_type == "text/csv":
        df = pandas.read_csv(StringIO(request_body))
        print(df.head())
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
    print(f"Input data columns: {input_data.columns}")
    # print(f"Model columns: {model.feature_names}")
    # features = model.predict(xgb.DMatrix(input_data.values))
    features = model.predict(input_data)
    
    return features

#     if label_column in input_data:
#         # Return the label (as the first column) and the set of features.
#         return np.insert(features, 0, input_data[label_column], axis=1)
#     else:
#         # Return only the set of features
#         return features

    
    
    

# if __name__ =='__main__':
#     model_fn(model_dir = "")
