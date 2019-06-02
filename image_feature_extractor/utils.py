import os, shutil
import pandas as pd


def change_validation_scaffolding(base_route: str, definition_file: str, separator: str, has_header: int):
    validation_data = _load_validation_data(definition_file, separator, has_header)
    
    for row in validation_data.iterrows():
        file = row[1]['file']
        label = row[1]['label']
        
        label_folder = os.path.join(base_route, label)
        
        if not os.path.exists(label_folder):
            os.mkdir(label_folder)
        
        shutil.move(os.path.join(base_route, file), os.path.join(label_folder, file))


def _load_validation_data(definition_file: str, separator: str, has_header: int):
    validation_data = pd.read_csv(
        definition_file,
        sep=separator,
        header=has_header
    )
    data_columns = validation_data.columns
    
    if len(data_columns) > 2:
        validation_data = validation_data[[data_columns[0], data_columns[1]]]
    elif len(data_columns) == 2:
        validation_data[data_columns[1]] = validation_data[data_columns[1]].apply(
            lambda content: content if ' ' not in content else content.split(' ')[0])
    
    validation_data.columns = ['file', 'label']
    
    return validation_data
