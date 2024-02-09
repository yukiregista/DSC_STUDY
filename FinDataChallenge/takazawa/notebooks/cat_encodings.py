import pandas as pd
import category_encoders as ce
from sklearn.ensemble import HistGradientBoostingClassifier


def count_encoding(df, columns):
    count_encoder = ce.CountEncoder(cols=columns)
    df_encoded = count_encoder.fit_transform(df)
    for col in columns:
        df[col + "_count"] = df_encoded[col]
    return df


def count_rank_encoding(df, columns):
    for col in columns:
        count_rank = df.groupby(col)[col].count().rank(ascending=False)
        df[col + '_rank'] = df[col].map(count_rank)
    return df


def Holdout_target_encoding(X,y, column,folds):
  df = X; df['target']=y
  df[column + "_target"] = 0.9
  tmp = df[[column, column + "_target"]]
  for idx1, idx2 in folds:
    train = df.iloc[idx1]
    #val = df.iloc[idx2]
    mean = train.groupby(column)['target'].mean()
    for ind, v in tmp.iloc[idx2].iterrows():
      try:
        tmp.loc[ind,column+"_target"] = mean.loc[tmp.loc[ind, column]]
      except:
        continue
  df[column+ "_target"] = tmp[column + "_target"]
  return df

def Holdout_target_encoding_smoothed(X,y,column,folds):
  df = X; df['target']=y
  df[column + "_target"] = 0.9
  tmp = df[[column, column + "_target"]]
  for idx1, idx2 in folds:
    train = df.iloc[idx1]
    krr = HistGradientBoostingClassifier(l2_regularization=1)
    krr = krr.fit(train[[column]], train['target'])
    val = df.iloc[idx2]
    index = tmp.iloc[idx2].index
    tmp.loc[index, column + "_target"] = krr.predict_proba(val[[column]])[:,1]
  df[column+ "_target"] = tmp[column + "_target"]
  return df

def target_encode_test(train_X, train_y, test_X, column):
  df = train_X; df['target'] = train_y
  mean = train_X.groupby(column)['target'].mean()
  test_X.loc[:, column + "_target"] = 0.9
  for ind in mean.index:
      test_X.loc[test_X[column] == ind, column + "_target"] = mean[ind]
  return test_X
  
  
def target_encode_smooth_test(train_X, train_y, test_X, column):
  df = train_X; df['target'] = train_y
  krr = HistGradientBoostingClassifier()
  krr = krr.fit(df[[column]],df['target'])
  test_X.loc[:, column + "_target"] = 0.9
  test_X.loc[:, column + "_target"] = krr.predict_proba(test_X[[column]])[:,1]
  return test_X