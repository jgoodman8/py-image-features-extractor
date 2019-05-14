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
@click.option('--mode', type=click.Choice(['deep', 'lbp', 'bow']), help="Feature extraction type")
@click.option('--src', type=str, help="Input images base route")
@click.option('--deep-model', type=str, help="Model used to train model or extract features")
@click.option('--output', type=str, help="Output route for features csv file")
@click.option('--size', type=int, help="Image size for (height, width)")
@click.option('--points', type=int, help="Number of points used to describe ")
@click.option('--radius', type=int, help="")
@click.option('--radius', type=int, help="")
@click.option('--grid', type=int, help="")
@click.option('--bow-method', type=click.Choice(['kaze']), help="")
@click.option('--k', type=int, help="")
def extract(mode: str, src: str, deep_model: str, size: int, output: str, points: int, radius: int,
            grid: int, bow_method: str, k: int):
    if mode == 'deep':
        if has_required_parameters(src=src, deep_model=deep_model, size=size, output=output):
            extractor = DeepExtractor(base_route=src, model_name=deep_model, size=size)
            extractor.extract_and_save(output_file=output)
    elif mode == 'lbp':
        if has_required_parameters(src=src, points=points, radius=radius, grid=grid, size=size, output=output):
            extractor = LBPExtractor(base_route=src, points=points, radius=radius, grid_x=grid, grid_y=grid, size=size)
            extractor.extract_and_save(output_file=output)
    elif mode == 'bow':
        if has_required_parameters(src=src, bow_method=bow_method, k=k, output=output):
            extractor = BoWExtractor(base_route=src, method=bow_method, k=k)
            extractor.fit()
            extractor.extract_and_save(output_file=output)
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
