import click

from image_feature_extractor.extractors import BoWExtractor
from image_feature_extractor.extractors import DeepExtractor
from image_feature_extractor.extractors import LBPExtractor
from image_feature_extractor.models import ImageModel
from image_feature_extractor.utils import change_validation_scaffolding


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    if ctx is None:
        click.echo('An action should be specified. Try --help to check the options')


@main.command()
@click.option('--images-folder', type=str, help="Input images base route")
@click.option('--train-folder', type=str, default="train", help="Folder name for train data")
@click.option('--validation-folder', type=str, default="val", help="Folder name for validation data")
@click.option('--epochs', type=int, default=10, help="Epochs to train model")
@click.option('--fine-tune', type=bool, help="If true, it performs fine tuning")
def train(images_folder: str, train_folder: str, validation_folder: str, epochs: int, fine_tune: bool):
    model = ImageModel(base_route=images_folder, epochs=epochs, train_folder=train_folder,
                       validation_folder=validation_folder, fine_tune=fine_tune)
    metrics = model.train()
    
    print(metrics)


@main.command()
@click.option('--deep', is_flag=True, help="Feature extraction using deep learning models")
@click.option('--lbp', is_flag=True, help="Feature extraction using Local Binary Patterns")
@click.option('--bow', is_flag=True, help="Feature extraction using Bag of Visual Words technique")
@click.option('--src', type=str, help="Input images base route")
@click.option('--dst', type=str, help="Output route for features csv file")
@click.option('--cnn', type=click.Choice(['vgg19', 'inception_v3', 'inception_resnet_v2']),
              help="Model used to train model or extract features")
@click.option('--size', type=int, help="Image size for (height, width)")
@click.option('--points', type=int, help="Number of points used to use on the transformation")
@click.option('--radius', type=int, help="Radius used on the LBP transformation")
@click.option('--grid', type=int,
              help="Size of the image grid (number of chunks to divide every image) rather horizontally as vertically")
@click.option('--detector', type=click.Choice(['kaze', 'sift', 'surf']),
              help="Feature detection method used to perform the BoW extraction")
@click.option('--k', type=int, help="Number of centroids to use on the clustering step")
@click.option('--export', is_flag=True,
              help="If used, the kMeans centroids will be saved at the '--vocabulary-route' route")
@click.option('--export', is_flag=True, help="If used, the kMeans centroids will be saved at the '--vocabulary-route'")
@click.option('--load', is_flag=True, help="If used, the kMeans centroids will be loaded from the '--vocabulary-route'")
@click.option('--vocabulary-route', type=str, help="Route where to load/save the kMeans vocabulary")
def extract(deep: bool, lbp: bool, bow: bool, src: str, dst: str, cnn: str, size: int, points: int, radius: int,
            grid: int, detector: str, k: int, export: bool, load: bool, vocabulary_route: str):
    if deep:
        if has_required_parameters(src=src, deep_model=cnn, size=size, output=dst):
            extractor = DeepExtractor(base_route=src, model_name=cnn, size=size)
            extractor.extract_and_save(output_file=dst)
    elif lbp:
        if has_required_parameters(src=src, points=points, radius=radius, grid=grid, size=size, output=dst):
            extractor = LBPExtractor(base_route=src, points=points, radius=radius, grid_x=grid, grid_y=grid, size=size)
            extractor.extract_and_save(output_file=dst)
    elif bow:
        if has_required_parameters(src=src, bow_method=detector, k=k, output=dst):
            extractor = BoWExtractor(base_route=src, method=detector, k=k)
            extractor.setup()
            if load:
                extractor.load(vocabulary_route)
            else:
                extractor.fit()
            if export:
                extractor.export(vocabulary_route)
            extractor.extract_and_save(output_file=dst)
    
    else:
        click.echo('Extraction mode is required (--mode option)')


@main.command()
@click.option('--images-route', type=str, help="Input images base route")
@click.option('--definition-file', type=str, help="[Scaff] Validation definition file route")
@click.option('--separator', type=str, default='\t', help="[Scaff] Separator character")
def scaff(images_route: str, definition_file: str, separator: str):
    change_validation_scaffolding(images_route, definition_file, separator)


def has_required_parameters(**kwargs):
    required_parameters = True
    
    for key in kwargs:
        if kwargs[key] is None:
            click.echo('Missing required option {}'.format(key))
            required_parameters = False
    
    return required_parameters
