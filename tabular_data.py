# %%
import pandas as pd
import numpy as np


def remove_rows_with_missing_ratings(df):
    df_no_nan = df.copy().dropna(subset=["Cleanliness_rating", "Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating", "Description"])
    # print(df_no_nan)
    combine_description_strings(df_no_nan)
    

def combine_description_strings(df_no_nan):
    combine_list = [",", "About this space", "[", "]"]
    df_no_nan["Description"] = df_no_nan["Description"].str.replace('"', '')
    df_no_nan["Description"] = df_no_nan["Description"].str.replace("'", "") 
    for combine_index in combine_list:   
        df_no_nan["Description"] = df_no_nan["Description"].str.replace(f"{combine_index}", "")         
    df_comb_string = df_no_nan
    # print(df_comb_string)

    set_default_feature_values(df_comb_string)

def set_default_feature_values(df_comb_string):
    column_list = ["guests", "beds", "bathrooms", "bedrooms"]

    for column_index in column_list:
        df_comb_string[column_index] = df_comb_string[column_index].fillna(1)
    clean_tabular_data = df_comb_string
    
    # print(clean_tabular_data)
    filepath = r"C:\Users\denni\Desktop\AiCore\Projects\tabular_data\clean_tabular_data.csv"
    clean_tabular_data.to_csv(filepath, index=True)

def clean_tabular_data(df):
    remove_rows_with_missing_ratings(df)

def load_airbnb(label):
    clean_df = pd.read_csv(r"C:\Users\denni\Desktop\AiCore\Projects\tabular_data\clean_tabular_data.csv")
    features = clean_df.select_dtypes(include=["int", "float"])
    labels = features[label]
    features = features.drop(columns=["Unnamed: 0", "Unnamed: 19", label])
    # print(labels)
    # print(features)
    data_tuple = (features, labels)
    # print(data_tuple)
    return(data_tuple)

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\denni\Desktop\AiCore\Projects\tabular_data\listing.csv")
    # clean_tabular_data(df)
    load_airbnb("Price_Night")



# %%
