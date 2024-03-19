# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import category_encoders as ce


# 訓練とテスト分けて前処理していただけると助かります．

# Read recipe inputs
train_prepared_stacked = dataiku.Dataset("（訓練データまたはテストデータ）")
train_prepared_stacked_df = train_prepared_stacked.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

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


train_prepared_stacked = count_encoding(train_prepared_stacked)

# columns used for training -> all_cols
num_cols = ['NoEmp', 'CreateJob', 'RetainedJob', 'ApprovalFY', 'DisbursementGross', 'GrAppv', 'SBA_Appv']
retained_cat_cols = ['NewExist', 'RevLineCr', 'LowDoc', 'UrbanRural']
timestamp_cols = ['DisbursementDate_Year','ApprovalDate_Year']
franchise_cols = ['FranchiseCode1', 'FranchiseCode0']
target_encode_cols = ['Sector', 'State', 'BankState']
count_encoded_cols = [item + "_count" for item in target_encode_cols]
target_encode_smooth_cols = ["longitude", "latitude"]
target_encoded_smooth_cols = [item + "_target" for item in target_encode_smooth_cols]
location_cols = ['latitude', 'longitude']
all_cols = num_cols + retained_cat_cols + timestamp_cols + franchise_cols + count_encoded_cols + location_cols + target_encoded_smooth_cols


train_prepared_stacked_df = train_prepared_stacked_df[all_cols]



done_df = train_prepared_stacked_df # For this sample code, simply copy input to output


# Write recipe outputs
done = dataiku.Dataset("done")
done.write_with_schema(done_df)