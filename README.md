# Modelling AirBnBs Property Listings Dataset

## Milestone 3
For this milestone I was asked to load the tabular data containing the image folder and listings .csv file and clean the data set.

### Task 1
For this task I was asked to load the tabular data set into a python file `tabular_data.py`. I was asked to create a function `remove_rows_with_missing_ratings` which removed rows with missing rating values. To do this I used the `.drop` method and put it the column names as a subset.

I was asked to create another function `combine_description_strings` which combined the list items into a single string. I've done this by:
```
combine_list = [",", "About this space", "[", "]"]
    df_no_nan["Description"] = df_no_nan["Description"].str.replace('"', '')
    df_no_nan["Description"] = df_no_nan["Description"].str.replace("'", "") 
    for combine_index in combine_list:   
        df_no_nan["Description"] = df_no_nan["Description"].str.replace(f"{combine_index}", "")         
    df_comb_string = df_no_nan
```
From the code above you can see that I used the `str.replace()` method to replace the list items by iterating through the `combine_list`.

I was asked to create another function `set_default_feature_values` which replaces the missing rating values with the number 1. I have done this by:
```
column_list = ["guests", "beds", "bathrooms", "bedrooms"]

    for column_index in column_list:
        df_comb_string[column_index] = df_comb_string[column_index].fillna(1)
    clean_tabular_data = df_comb_string
```
From the code above you can see that I replace the missing entries by iterating through the different column names and using the `.fillna()` method to replace the missing entries. The last thing I was asked to do was to save the new data frame as `clean_tabular_data.csv` and to call the different functions in an `if __name__ = "__main__"` block.

### Task 2
For the second task I was asked to create a python file `prepare_image_data.py`. I was asked to create a function `download_images` which downloads the image folder from an S3 bucket. I've done this by:
```
alphabet = ["a", "b", "c", "d", "e"]
    img_uuid_list = df["ID"]
    for img_index in img_uuid_list[:1]:
        url = bucket_url + str(img_index)

        for letter_index in range(len(alphabet)):
            download_url = url + "-" + str(alphabet[letter_index]) + ".png"
```
From the code above you can see that I am creating the download url by iterating through different UUIDs of the images and each UUID having 5 different images labelled from `a-e`. After constructing the URL I would then use the `s3.download_file()` method to download each example from my S3 bucket.

I was also asked to create another function called `resize_images` which would resize all the images to the same height and width. I've done this by:
```
    img_uuid_list = df["ID"]
    alphabet = ["a", "b", "c", "d", "e"]
    for resize_index in img_uuid_list:
        for letter_index in range(len(alphabet)):
            img_name = str(resize_index) + "-" + str(alphabet[letter_index]) + ".png"
            try:
                resize_image = Image.open(f"c:\\Users\\denni\\Desktop\\AiCore\\Projects\\images\\{resize_index}\{img_name}")
                new_image = resize_image.resize((720, 480))
            except FileNotFoundError:
                print("Image does not exist")

            try:
                os.mkdir(f"c:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\processed_images\{resize_index}")
            
            except FileExistsError:
                print("Processed image folder already exists")

            if new_image.mode == "RGB":
                new_image.save(f"c:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\processed_images\{resize_index}\{img_name}")
            else:
                pass
```
From the code above you can see I am employing a similar approach as in the previous task when constructing the image names. After constructing the image name you can see that I am iterating through the entire image folder and using the `.resize()` method to resize the images iteratively. I've also had to add some `except` statements since some entries in the data set did not have corresponding images in the image folder and when creating the `processed_images` folder if the folder already exists. Another requirement for resizing the images was to check if the image was in RGB format and to discard it if it wasn't.

### Task 3
For the last task of the milestone I was asked to create a function `load_airbnb` which would create the features and labels of the data frame and return them as a tuple. I've done this by:
```
clean_df = pd.read_csv(r"C:\Users\denni\Desktop\AiCore\Projects\tabular_data\clean_tabular_data.csv")
    features = clean_df.select_dtypes(include=["int", "float"])
    labels = features[label]
    features = features.drop(columns=["Unnamed: 0", "Unnamed: 19", label])
    data_tuple = (features, labels) 
```
From the code above you can see that I am creating the features data frame by using the `select.dtypes()` method to select the columns that were integer and float types. With the labels variable (argument of the function) I also remove it from the  features data frame and pass it to its own called `label`. After creating the two data frames I create the tuple variable `data_tuple = (features, labels)`.

## Milestone 4
For this milestone I was asked to make several different regression machine learning models and to tune the hyperparameters for the different models. The first model I trained was a `SGDRegressor` model. To tune the hyperparameters I passed in a dictionary of many different hyperparameters and used the `GridSearchCV`. After running the grid search it returned the score of 0.039. I have also done a similar process for `DecisionTreeRegressor`, `GradientBoostingRegressor` and `RandomForestRegressor` which have returned scores of 0.522, 0.489 and 0.561 respectively. This shows that the `SGDRegressor` is the best model due to having the lowest score.