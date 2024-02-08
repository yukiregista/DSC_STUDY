import pandas as pd
def clean_data(data):
    # convert data as in eda.ipynb
    # import addfips
    # af = addfips.AddFIPS()

    categorical_cols = ['FranchiseCode','RevLineCr', 'LowDoc', 'Sector', 'UrbanRural', 'NewExist']
    date_cols = ["DisbursementDate", "ApprovalDate"]
    dollar_cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    for col in categorical_cols:
        data[col] = data[col].astype('category')
        if data[col].isnull().sum():
            data[col] = data[col].cat.add_categories("NAN").fillna("NAN")
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], format="%d-%b-%y")
        # add date cols
        data[col + "_year"] = pd.DatetimeIndex(data[col]).year
        data[col + "_month"] = pd.DatetimeIndex(data[col]).month
        data[col + "_day"] = pd.DatetimeIndex(data[col]).day
        data[col + "_daystamp"] = (data[col] - data[col].min()).dt.days
    for col in dollar_cols:
        data[col] = data[col].str.replace("[$,]", "", regex=True)
        data[col] = data[col].astype(float)

    ## I want to run Codes below but currently not possible due to access limit??
    # all_states = data['State'].to_numpy()
    # all_state_fips = [af.get_state_fips(item) for item in all_states]
    # data['State_FIPS'] = all_state_fips
    # county_fips = [county_FIPS(item['City'], item['State'], item['State_FIPS']) for i, item in data.iterrows()]
    # data['County_FIPS'] = county_fips

    # coarse franchise col
    data['FranchiseCode1'] = (data['FranchiseCode']==1).astype("category")
    data['FranchiseCode0'] = (data['FranchiseCode']==1).astype("category")
    
    return data
    
    
