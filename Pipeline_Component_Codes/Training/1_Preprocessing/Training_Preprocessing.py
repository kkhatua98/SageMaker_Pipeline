##

def preprocessing_function():
    import os
    import logging
    import traceback
    import argparse
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    
    print(os.environ.get('SM_CHANNEL_TEST'))

    


    ###########################     Extracting the command line arguments     ########################

    parser = argparse.ArgumentParser()
    ## Adding arguments
    # Inputs
    parser.add_argument('--full_data_location', type=str, default="/opt/ml/processing/input/data/churn-bigml-80.csv")
    parser.add_argument('--feature_selection_file_location', type=str, default="/opt/ml/processing/input/feature_selection/Feature_Selection.csv")

    # Outputs
    parser.add_argument('--output_location', type=str, default="/opt/ml/processing/logss")

    # Others
    parser.add_argument('--target_column', type=str, default="Churn")
    parser.add_argument('--stop_split', type=str, default="")
    

    ## Parsing    
    args, _ = parser.parse_known_args()
    # Inputs
    full_data_location = args.full_data_location
    feature_selection_file_location = args.feature_selection_file_location

    # Outputs
    output_location = args.output_location

    # Others
    target = args.target_column
    stop_split = args.stop_split

    ###########################     Extracting the command line arguments : End     ########################
    




    ###########################     Creating the log extractor     ########################

    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f'{output_location}/logfile.log')
    logger.addHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    ###########################     Creating the log extractor : End     ########################
    
    
        



    try:
        
        ## Reading datasets
        whole_data = pd.read_csv(full_data_location)
        feature_selection = pd.read_csv(feature_selection_file_location)



        whole_data = whole_data.dropna()


        #--------------------------------------------------------------------
        # Correct data types
        #--------------------------------------------------------------------
        whole_data['Area code'] = whole_data['Area code'].astype('object')

        #--------------------------------------------------------------------
        # Create New Features
        #--------------------------------------------------------------------        
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

        #--------------------------------------------------------------------
        # Features to drop
        #--------------------------------------------------------------------        
        redundant_cols = ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
        feats_eng = []
        feats_still_to_eng = []
        leakage_feats = []
        # target = ['Churn']
        
        
        
        feature_selection = feature_selection.fillna('N')
        rejected_columns = feature_selection.loc[feature_selection["Selection"] == 'N', 'Column'].tolist()

        # return whole_data.drop(columns= redundant_cols + feats_eng + feats_still_to_eng + leakage_feats + target)

        # whole_data = whole_data.drop(columns= redundant_cols + feats_eng + feats_still_to_eng + leakage_feats + target)
        whole_data = whole_data.drop(columns= redundant_cols + feats_eng + feats_still_to_eng + leakage_feats + rejected_columns)

        categorical_feats = list(whole_data.select_dtypes('object').columns)
        numeric_feats = list(whole_data.select_dtypes('number').columns)

        scaler = MinMaxScaler()
        model=scaler.fit(whole_data[numeric_feats])
        whole_data[numeric_feats]=model.transform(whole_data[numeric_feats])

        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc.fit(X)
        # X[categorical_feats] = enc.transform(X[categorical_feats])

        whole_data = pd.get_dummies(whole_data, columns = categorical_feats, drop_first = True)
        
        if stop_split:
            whole_data.to_csv(f"{output_location}/evaluation.csv", index=False)
            
            logger.info("Data written to disk inside container.")
            logger.info("Preprocessing completed.")
            
            return

        ## Splitting the whole data in train and test set.
        train, test = train_test_split(whole_data, test_size=0.33, random_state=42)
        logger.info("Train and test set created.")


        ## Writing the splitted data into specific location.
        pd.DataFrame(train).to_csv(f"{output_location}/train.csv", index=False)
        pd.DataFrame(test).to_csv(f"{output_location}/test.csv", index=False)
        # pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", index=False)
        logger.info("Data written to disk inside container.")


        logger.info("Preprocessing completed.")
        
        logger.removeHandler(handler)
        handler.close()
    
    except:
        var = traceback.format_exc()
        logger.error(var)
        
        logger.removeHandler(handler)
        handler.close()



if __name__ == "__main__":
    preprocessing_function()
