import click

from image_feature_extractor.extractors.feature_extractor import FeatureExtractor
from image_feature_extractor.models.image_model import ImageModel
from image_feature_extractor.utils import change_validation_scaffolding


@click.command()
@click.option('--train', type=bool, help="If true, it performs models training and testing")
@click.option('--extract', type=bool, help="If true, it performs feature extraction")
@click.option('--scaff', type=bool, help="If true, it performs a change in validation scaffolding")
@click.option('--images-route', type=str, help="Input images base route")
@click.option('--model', type=str, help="[Train/Extract] Model used to train model or extract features")
@click.option('--output-route', type=str, help="[Extract] Output route for features csv file")
@click.option('--image-size', type=int, help="[Extract] Image size for (height, width)")
@click.option('--epochs', type=int, default=10, help="[Train] Epochs to train model")
@click.option('--fine-tune', type=bool, help="[Train] If true, it performs fine tuning")
@click.option('--fine-tune-epochs', type=int, default=100, help="[Train] Epochs to fine tune the model")
@click.option('--definition-file', type=str, help="[Scaff] Validation definition file route")
@click.option('--separator', type=str, default=',', help="[Scaff] Separator character")
def main(train: bool, extract: bool, scaff: bool,
         images_route: str, model: str,
         output_route: str, image_size: int,
         epochs: int, fine_tune: bool, fine_tune_epochs: int,
         definition_file: str, separator: str):
    if train:
        model = ImageModel(base_route=images_route, epochs=epochs, fine_tune_epochs=fine_tune_epochs,
                           fine_tune=fine_tune)
        metrics = model.train()
        
        print(metrics)
    
    elif extract:
        extractor = FeatureExtractor(images_route, model, output_route)
        if int(image_size) > 0:
            extractor.width = int(image_size)
            extractor.height = int(image_size)
        extractor.extract_and_store()
    
    elif scaff:
        change_validation_scaffolding(images_route, definition_file, separator)
    
    else:
        print('Select an option: train, extract o scaff')
