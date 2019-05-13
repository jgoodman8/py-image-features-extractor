import os, shutil
import pandas as pd


def change_validation_scaffolding(base_route, definition_file, separator):
    validation_data = _load_validation_data(definition_file, separator)
    
    for row in validation_data.iterrows():
        file = row[1]["file"]
        label = row[1]["label"]
        
        label_folder = os.path.join(base_route, label)
        
        if not os.path.exists(label_folder):
            os.mkdir(label_folder)
        
        shutil.move(os.path.join(base_route, file), os.path.join(label_folder, file))


def _load_validation_data(definition_file, separator):
    validation_data = pd.read_csv(
        definition_file,
        sep=separator,
        header=None
    )
    
    validation_data.columns = ["file", "label", "0", "1", "2", "3"]
    
    return validation_data
