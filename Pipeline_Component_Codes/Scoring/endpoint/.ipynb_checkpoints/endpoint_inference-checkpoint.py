

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



def preprocessing_function(whole_data):
    import os
    import traceback
    import sys
    import logging
    import argparse
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import boto3
    
    print(os.environ.get('SM_CHANNEL_TEST'))
    
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--target_column', type=str, default="Churn")
    # parser.add_argument('--batch_data_location', type=str, default="/opt/ml/processing/input")
    # parser.add_argument('--feature_selection_file_location', type=str, default="/opt/ml/processing/input/feature_selection")
    # parser.add_argument('--preprocessed_batch_data_location', type=str, default="/opt/ml/processing/train")
    # parser.add_argument('--log_location', type=str, default="/opt/ml/processing/logss")
    
        
    # args, _ = parser.parse_known_args()
    # target = args.target_column
    # batch_data_location = args.batch_data_location
    # feature_selection_file_location = args.feature_selection_file_location
    # preprocessed_batch_data_location = args.preprocessed_batch_data_location
    # log_location = args.log_location
    
    target = os.environ.get('target_column')
    feature_selection_file_location = os.environ.get('feature_selection_file_location')
    log_location = os.environ.get('log_location')


    ## Creating a logger.
    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
#     handler = logging.FileHandler(f'{log_location}/logfile.log')
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)


    try:
        ## Reading and cleaning data.
#         base_dir = "/opt/ml/processing"
#         base_dir_contents = os.listdir(f"{batch_data_location}")
#         for content in base_dir_contents:
#             if ".csv" in content:
#                 input_data_file = content
#         print(f"Files or folders present in the folder is {input_data_file}")
#         print(input_data_file)
#         whole_data = pd.read_csv(f"{batch_data_location}/{input_data_file}")
        feature_selection_file_location = os.environ.get('feature_selection_file_location')
        s3_client = boto3.client('s3')
        # Download the file from S3
    
        bucket = feature_selection_file_location.split('/')[2]
        feature_selection_file_name = feature_selection_file_location.split('/')[-1]
        s3_client.download_file(bucket, feature_selection_file_name, feature_selection_file_name)
    
        feature_selection = pd.read_csv(feature_selection_file_name)
        column_names = feature_selection['Column'].tolist()
        whole_data.columns = column_names
        
        
        whole_data = whole_data.dropna()



        #################
        ## Correct data types
        #################

        whole_data['Area code'] = whole_data['Area code'].astype('object')
        logger.info("Data types corrected.")

        ############## --  ################





        #################
        ## Create New Features
        #################        
        # Account length features
        whole_data['Account_length_bins'] = pd.qcut(whole_data['Account length'], 4, labels= ['q1', 'q2', 'q3', 'q4'])
        whole_data['Account_length_bins'] = whole_data['Account_length_bins'].astype('object')

        # Voice mail messages
        whole_data['zero_vmails'] = 'No'
        whole_data.loc[whole_data['Number vmail messages'] == 0 , 'zero_vmails'] = 'Yes'

        # Minutes and Number of calls related features
        whole_data['Total_minutes'] = (whole_data['Total day minutes'] + whole_data['Total eve minutes'] + 
                                       whole_data['Total night minutes'] + whole_data['Total intl minutes'])

        whole_data['Total_calls'] = (whole_data['Total day calls'] + whole_data['Total eve calls'] + 
                                     whole_data['Total night calls'] + whole_data['Total intl calls'])

        whole_data['Minutes_per_call_overall'] = whole_data['Total_minutes']/whole_data['Total_calls']
        whole_data['Minutes*call_overall'] = whole_data['Total_minutes']*whole_data['Total_calls']

        whole_data['Minutes_per_call_int'] = whole_data['Total intl minutes']/whole_data['Total intl calls']
        whole_data['Minutes_per_call_int'].fillna(0, inplace=True)
        whole_data['Minutes*call_int'] = whole_data['Total intl minutes']*whole_data['Total intl calls']

        whole_data['Minutes_per_call_day'] = whole_data['Total day minutes']/whole_data['Total day calls']
        whole_data['Minutes_per_call_day'].fillna(0, inplace=True)
        whole_data['Minutes*call_day'] = whole_data['Total day minutes']*whole_data['Total day calls']

        whole_data['Minutes_per_call_eve'] = whole_data['Total eve minutes']/whole_data['Total eve calls']
        whole_data['Minutes_per_call_eve'].fillna(0, inplace=True)
        whole_data['Minutes*call_eve'] = whole_data['Total eve minutes']*whole_data['Total eve calls']

        whole_data['Minutes_per_call_night'] = whole_data['Total night minutes']/whole_data['Total night calls']
        whole_data['Minutes_per_call_night'].fillna(0, inplace=True)
        whole_data['Minutes*call_night'] = whole_data['Total night minutes']*whole_data['Total night calls']

    
        # Total charge feature
        whole_data['Total_charge'] = (whole_data['Total day charge'] + whole_data['Total eve charge'] + 
                                      whole_data['Total night charge'] + whole_data['Total intl charge'])

        # Customer service calls related features
        whole_data['Day_minutes_per_customer_service_calls'] = whole_data['Total day minutes']/whole_data['Customer service calls']
        whole_data['Day_minutes_per_customer_service_calls'].replace(np.inf, 0, inplace=True)
        whole_data['Day_minutes*customer_service_calls'] = whole_data['Total day minutes']*whole_data['Customer service calls']

        whole_data['Customer_service_calls_bins'] = pd.cut(whole_data['Customer service calls'], 4, labels= ['q1', 'q2', 'q3', 'q4'])
        whole_data['Customer_service_calls_bins'] = whole_data['Customer_service_calls_bins'].astype('object')

        # Minutes features
        whole_data['Total_day_minutes_wholenum'] = whole_data['Total day minutes'].apply(lambda x:x//1)
        whole_data['Total_day_minutes_decimalnum'] = whole_data['Total day minutes'].apply(lambda x:x%1)

        whole_data['Total_minutes_wholenum'] = whole_data['Total_minutes'].apply(lambda x:x//1)
        whole_data['Total_minutes_decimalnum'] = whole_data['Total_minutes'].apply(lambda x:x%1)
    
    
    
        # Having both voice and international plan
        a = (whole_data['International plan']=='Yes')
        b = (whole_data['Voice mail plan']=='Yes')
        whole_data['Voice_and_Int_plan'] = a&b
        whole_data['Voice_and_Int_plan'].replace(True, 1, inplace=True)
        whole_data['Voice_and_Int_plan'].replace(False, 1, inplace=True)

        # Having both voice and international plan
        a = (whole_data['International plan']=='Yes')
        b = (whole_data['Voice mail plan']=='Yes')
        whole_data['Voice_and_Int_plan'] = a&b
        whole_data['Voice_and_Int_plan'].replace(True, 1, inplace=True)
        whole_data['Voice_and_Int_plan'].replace(False, 1, inplace=True)       

        # Having only international plan
        a = (whole_data['International plan']=='Yes')
        b = (whole_data['Voice mail plan']=='No')
        whole_data['Only_Int_plan'] = a&b
        whole_data['Only_Int_plan'].replace(True, 1, inplace=True)
        whole_data['Only_Int_plan'].replace(False, 1, inplace=True)


        # Having only voice mail plan
        a = (whole_data['International plan']=='No')
        b = (whole_data['Voice mail plan']=='Yes')
        whole_data['Only_vmail_plan'] = a&b
        whole_data['Only_vmail_plan'].replace(True, 1, inplace=True)
        whole_data['Only_vmail_plan'].replace(False, 1, inplace=True)

        # Having no plans
        a = (whole_data['International plan']=='No')
        b = (whole_data['Voice mail plan']=='No')
        whole_data['No_plans'] = a&b
        whole_data['No_plans'].replace(True, 1, inplace=True)
        whole_data['No_plans'].replace(False, 1, inplace=True)

        logger.info("New features created.")

        ############## --  ################





        #################
        ## Dropping some features.
        #################      
        redundant_cols = ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
        feats_eng = []
        feats_still_to_eng = []
        leakage_feats = []
        target = ['Churn']


#         feature_dir_contents = os.listdir(feature_selection_file_location)
#         for content in feature_dir_contents:
#             if ".csv" in content:
#                 input_feature_file = content
#         print(f"Files or folders present in the folder is {input_feature_file}")
#         print(input_feature_file)
#         feature_selection = pd.read_csv(f"{feature_selection_file_location}/{input_feature_file}")

        
        # feature_selection = pd.read_csv("/opt/ml/processing/input/feature_selection/Feature_Selection.csv")
        feature_selection = feature_selection.fillna('N')
        rejected_columns = feature_selection.loc[feature_selection["Selection"] == 'N', 'Column'].tolist()


        whole_data = whole_data.drop(columns= redundant_cols + feats_eng + feats_still_to_eng + leakage_feats + target + rejected_columns)

        logger.info("Dropped some features.")

        ############## --  ################




        #################
        ## Scaling and One-Hot Encoding.
        #################
        categorical_feats = list(whole_data.select_dtypes('object').columns)
        numeric_feats = list(whole_data.select_dtypes('number').columns)


        ## Scaling numeric variables using Min-Max Scaling.
        scaler = MinMaxScaler()
        model=scaler.fit(whole_data[numeric_feats])
        whole_data[numeric_feats]=model.transform(whole_data[numeric_feats])


        ## Creating dummy variables (One-Hot Encoding)
        whole_data = pd.get_dummies(whole_data, columns = categorical_feats, drop_first = True)

        logger.info("Scaling and One-Hot Encoding done.")

        ############## --  ################




        ## Writing data into specific location.
        # pd.DataFrame(whole_data).to_csv(f"{preprocessed_batch_data_location}/Processed.csv", index=False)
        logger.info("Data written to disk inside container.")


        logger.info("Preprocessing completed.")
        
        logger.removeHandler(handler)
        handler.close()
        
        return pd.DataFrame(whole_data)

    except:
        var = traceback.format_exc()
        logger.error(var)
        
        logger.removeHandler(handler)
        handler.close()






def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    import os
    import pandas
    from io import StringIO
    # import subprocess
    import boto3
    
    # subprocess.run(["pip", "install", "fsspec"])
    df = pandas.read_csv(StringIO(request_body))
    
    return preprocessing_function(df)
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
