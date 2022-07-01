

def lr_training_function():
    import os
    import pandas
    import logging
    import argparse
    # from sklearn.externals import joblib
    import joblib
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression


    ## Creating a logger.
    # logger = logging.getLogger()
    # logging.captureWarnings(True)
    # logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler())
    # logger.info("Training started.")
    
    
    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler('/opt/ml/output/data/logfile.log')
    # handler = logging.FileHandler('logfile.log')
    logger.addHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)  
    


    try:
        # Hyper Parameter Tuning
        ## Hyperparameters sent by the client are passed as command-line arguments to the script.
        parser = argparse.ArgumentParser()
        parser.add_argument('--penalty', type=str, default="l2")
        parser.add_argument('--C', type=float, default=1.0)
        parser.add_argument('--solver', type=str, default="lbfgs")
        parser.add_argument('--objective_metric', type=str, default="accuracy")



        ## Getting the Data, model, and output directories from the parent hyperparameter tuning job.
        ## If not given it takes a default value.
        parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
        logger.info(os.environ.get('SM_CHANNEL_TEST'))

        args, _ = parser.parse_known_args()
        logger.info("Arguments Parsed")
        logger.info(args.test)


        solver_penalty_mapper = {"newton-cg":["l2"], "lbfgs":["l2"], "liblinear":["l1", "l2"], "sag":["l2"], "saga":["elasticnet", "l1", "l2"]}


        print(f"Solver type: {args.solver}")
        print(f"Value of C: {args.C}")
        print(f"Penalty type: {args.penalty}")

        print(f"Checking whether given solver is {args.solver} compatible with the given penalty type {args.penalty}.")
        if args.penalty in solver_penalty_mapper[args.solver]:
            penalty = args.penalty
            print("Check: OK")
        else:
            penalty = solver_penalty_mapper[args.solver][0]
            print(f"Check: Penalty type changed from {args.penalty} to {penalty}")



        # Loading train and test data from args.train and args.test.
        train_data = pandas.read_csv(f"{args.train}/train.csv")
        test_data = pandas.read_csv(f"{args.test}/test.csv")
        # print(train_data.columns)



        ## Fitting model.
        X_train = train_data.drop(columns = ["Churn"])
        y_train = train_data.Churn
        X_test = test_data.drop(columns = ["Churn"])
        y_test = test_data.Churn

        # mod_dt = DecisionTreeClassifier(max_depth = args.max_depth, random_state = args.random_state)
        # mod_dt.fit(X_train, y_train)
        # logger.info("Model fitted.")

        mod_lr = mod_lr = LogisticRegression(penalty = penalty, C = args.C, solver = args.solver)
        mod_lr.fit(X_train, y_train)
        logger.info("Model fitted.")



        ## Writing model to disk.
        joblib.dump(mod_lr, os.path.join(args.model_dir, "model.joblib"))
        logger.info("Model written to disk.")



        ## Getting predictions and calculating accuracy.
        prediction=mod_lr.predict(X_test)
        pandas.DataFrame(prediction, columns = ["Predictions"]).to_csv(f"{args.output_data_dir}/Prediction.csv")
        logger.info("Predictions written to disk.")
        
        
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
