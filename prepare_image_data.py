# %%
import os
import pandas as pd
import boto3
from PIL import Image

s3 = boto3.client('s3',
                  aws_access_key_id="AKIAZ244M2OYUJU5QKFB",
                  aws_secret_access_key="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")

bucket_url = "images/"


def download_images():
    alphabet = ["a", "b", "c", "d", "e"]
    img_uuid_list = df["ID"]
    for img_index in img_uuid_list[:1]:
        url = bucket_url + str(img_index)

        for letter_index in range(len(alphabet)):
            download_url = url + "-" + str(alphabet[letter_index]) + ".png"
    # print(download_url)
    # s3.download_file("airbnb-imagesz", "images/f9dcbd09-32ac-41d9-a0b1-fdb2793378cf/f9dcbd09-32ac-41d9-a0b1-fdb2793378cf-e.png", r"C:\Users\denni\Desktop\AiCore\Projects\modelling-airbnbs-property-listing-dataset-\processed_images\image.png")
    # s3.download_file("airbnb-imagesz", "*", r"C:\Users\denni\Desktop\AiCore\Projects\images") > Will download entire S3 bucket. DO NOT RUN.
    # Line above will download entire S3 bucket. DO NOT RUN.

def resize_images():
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

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\denni\Desktop\AiCore\Projects\tabular_data\clean_tabular_data.csv")
    # download_images()
    resize_images()
    
# %%
