

def lr_training_function():
    import os
    import traceback
    import sys
    import pandas
    import logging
    import argparse
    import joblib
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier



    

    ###########################     Extracting the command line arguments     ########################

    ## Getting the Data, model, and output directories from the parent hyperparameter tuning job.
    ## If not given it takes a default value.

    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # Outputs
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))


    # Hyperparameters
    ## Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n_estimators', type=str, default=2)
    parser.add_argument('--max_depth', type=float, default=2)
    parser.add_argument('--learning_rate', type=str, default=1)
    parser.add_argument('--objective_metric', type=str, default="accuracy")

    args, _ = parser.parse_known_args()

    ###########################     Extracting the command line arguments : End     ########################
    


    ###########################     Creating the log extractor     ########################

    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f'{args.output_data_dir}/XGB_logfile.log')
    # handler = logging.FileHandler('logfile.log')
    logger.addHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)  

    ###########################     Creating the log extractor : End     ########################
    


    try:



        # Loading train and test data from args.train and args.test.
        train_data = pandas.read_csv(f"{args.train}/train.csv")
        test_data = pandas.read_csv(f"{args.test}/test.csv")
        # print(train_data.columns)



        ## Fitting model.
        X_train = train_data.drop(columns = ["Churn"])
        y_train = train_data.Churn
        X_test = test_data.drop(columns = ["Churn"])
        y_test = test_data.Churn


        bst = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate)
        bst.fit(X_train, y_train)
        logger.info("Model fitted.")



        ## Writing model to disk.
        # joblib.dump(bst, os.path.join(args.model_dir, "model.joblib"))
        bst.save_model(f'{args.model_dir}xgb_model.json')
        logger.info("Model written to disk.")



        ## Getting predictions and calculating accuracy.
        prediction=bst.predict(X_test)
        pandas.DataFrame(prediction, columns = ["Predictions"]).to_csv(f"{args.output_data_dir}/Prediction.csv")
        logger.info("Predictions written to disk.")
        
        
        ## Getting feature importance value
        # feat_importance = mod_lr.coef_.tolist()[0]
        feat_importance = list(bst.get_booster().get_score(importance_type='gain').values())
        # print(feat_importance)
        # print(X_train.columns.tolist())
        # print(len(feat_importance))
        # print(len(X_train.columns.tolist()))
        print(list(bst.feature_importances_)[0])
        # print(len(bst.feature_importances_))
        # https://datascience.stackexchange.com/questions/19882/xgboost-how-to-use-feature-importances-with-xgbregressor
        feat_importance_record = pandas.DataFrame({"Variables":X_train.columns.tolist(), "Importance_Values":list(bst.feature_importances_)})
        feat_importance_record.to_csv(f"{args.output_data_dir}/Feature_Importance.csv", index = False)
        
        
        objective_metric = args.objective_metric
        if objective_metric == "anything":
            objective_metric = "accuracy"
        
        if objective_metric == "accuracy":
            metric_value = accuracy_score(y_test, prediction)
        if objective_metric == "precision":
            metric_value = precision_score(y_test, prediction)
        if objective_metric == "recall":
            metric_value = recall_score(y_test, prediction)
        if objective_metric == "f1-score":
            metric_value = f1_score(y_test, prediction)
        
        with open(os.path.join(args.output_data_dir, 'metrics.txt'), 'w') as out:
            out.write(str(metric_value))
        print(f"{objective_metric}:{metric_value}")
        logger.info(f"{objective_metric} calculated.")
        
        
        ## Writing varius model performance metrics
        from datetime import date
        today = date.today()
        metrices = ["F1","Recall","Accuracy","Precision"]
        train_prediction = bst.predict(X_train)
        metrics = pandas.DataFrame([], columns = ["Training_Date","Dataset","Metric", "Value"])
        metrics["Training_Date"] = [today] * len(metrices) * 2
        metrics["Dataset"] = ["Train"] * len(metrices) + ["Test"] * len(metrices)
        metrics["Metric"] = metrices * 2
        metrics["Value"] = [f1_score(y_train, train_prediction), recall_score(y_train, train_prediction), accuracy_score(y_train, train_prediction), precision_score(y_train, train_prediction)] + [f1_score(y_test, prediction), recall_score(y_test, prediction), accuracy_score(y_test, prediction), precision_score(y_test, prediction)]
        metrics.to_csv(f"{args.output_data_dir}/Metrics.csv", index = False)
        
        
        
        ## Confusion matrix
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_train, train_prediction).ravel()
        train_row = [tn, fp, fn, tp]
        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        test_row = [tn, fp, fn, tp]
        matrix = pandas.DataFrame([[today, "Train"] + train_row, [today, "Test"] + test_row], columns = ["Confusion_Date", "Data", "TN", "FP", "FN", "TP"])
        matrix.to_csv(f"{args.output_data_dir}/Confusion_Matrix.csv", index = False)
        
        
        ## Closing the logger.
        logger.removeHandler(handler)
        handler.close()
        
    except:
        var = traceback.format_exc()
        logger.error(var)
        
        logger.removeHandler(handler)
        handler.close()
        





if __name__ =='__main__':
    lr_training_function()
