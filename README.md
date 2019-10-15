mt = MainTransformer()
ct = CategoricalTransformer(drop_original=True)
n_fold = 3
folds = StratifiedKFold(n_splits=n_fold)

classifier_model = ClassifierModel(original_columns=X.columns, model_wrapper=LGBWrapper())
classifier_model.fit(X=X_train, y=y_train, X_holdout=X_holdout, y_holdout=y_holdout,
                    folds=folds, params=lgb_params, transformer=ct, preprocesser=mt)
The code consists of several parts:

transformer with some general data processing: for example creating feature interactions;
transformer which will be trained and applied on folds: for example target encoding;
model wrapper. Different models require data in different formats and have different API. I plan to write several wrappers for most popular models: lgb, xgb. catboost and sklearn models. As wrappers have the same API, they can be easily used in the main training function;
training class itself. It trains the models on folds, shows feature importances and so on.
I plan to develop this utility script and use it in my kernels.
