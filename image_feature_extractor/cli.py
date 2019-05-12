import click

from image_feature_extractor.extractors.deep_extractor import DeepExtractor
from image_feature_extractor.models.image_model import ImageModel
from image_feature_extractor.utils import change_validation_scaffolding


@click.command()
@click.option('--train', type=bool, help="If true, it performs models training and testing")
@click.option('--extract', type=bool, help="If true, it performs feature extraction")
@click.option('--scaff', type=bool, help="If true, it performs a change in validation scaffolding")
@click.option('--images-route', type=str, help="Input images base route")
@click.option('--model-name', type=str, help="[Train/Extract] Model used to train model or extract features")
@click.option('--output-file', type=str, help="[Extract] Output route for features csv file")
@click.option('--image-size', type=int, help="[Extract] Image size for (height, width)")
@click.option('--train-folder', type=str, default="train", help="[Train] Folder name for train data")
@click.option('--validation-folder', type=str, default="val", help="[Train] Folder name for validation data")
@click.option('--epochs', type=int, default=10, help="[Train] Epochs to train model")
@click.option('--fine-tune', type=bool, help="[Train] If true, it performs fine tuning")
@click.option('--definition-file', type=str, help="[Scaff] Validation definition file route")
@click.option('--separator', type=str, default='\t', help="[Scaff] Separator character")
def main(train: bool, extract: bool, scaff: bool,
         images_route: str, model_name: str,
         output_file: str, image_size: int,
         train_folder: str, validation_folder: str, epochs: int, fine_tune: bool,
         definition_file: str, separator: str):
    if train:
        model = ImageModel(base_route=images_route, epochs=epochs, train_folder=train_folder,
                           validation_folder=validation_folder, fine_tune=fine_tune)
        metrics = model.train()
        
        print(metrics)
    
    elif extract:
        extractor = DeepExtractor(base_route=images_route, model_name=model_name, size=image_size)
        extractor.extract_and_save(output_file=output_file)
    
    elif scaff:
        change_validation_scaffolding(images_route, definition_file, separator)
    
    else:
        print('Select an option: train, extract o scaff')
