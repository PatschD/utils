utils script:

mt = MainTransformer()
ct = CategoricalTransformer(drop_original=True)
n_fold = 3
folds = StratifiedKFold(n_splits=n_fold)

classifier_model = ClassifierModel(original_columns=X.columns, model_wrapper=LGBWrapper())
classifier_model.fit(X=X_train, y=y_train, X_holdout=X_holdout, y_holdout=y_holdout,
                    folds=folds, params=lgb_params, transformer=ct, preprocesser=mt)
