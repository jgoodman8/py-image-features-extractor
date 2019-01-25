import os, shutil
import pandas as pd


def change_validation_scaffolding(validation_base_route):
  validation_data = __load_validation_data(validation_base_route)
  for label in validation_data["label"].unique():
    os.mkdir(validation_base_route + "/" + label)
  
  for row in validation_data.iterrows():
    file = row[1]["file"]
    label = row[1]["label"]
    
    src = validation_base_route + "/images/" + file
    std = validation_base_route + "/" + label + "/" + file
    
    shutil.move(src, std)


def __load_validation_data(validation_base_route):
  headers = ["file", "label", "0", "1", "2", "3"]
  validation_data = pd.read_csv(
    validation_base_route + "/val_annotations.txt",
    sep='\t',
    header=None
  )
  
  validation_data.columns = headers
  
  return validation_data
