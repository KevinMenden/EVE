import click
import sys
import rich
import rich.logging
import logging
import os

from EVE.training import train_EVE

"""
some comments
"""

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(
    rich.logging.RichHandler(
        level=logging.INFO,
        console=rich.console.Console(file=sys.stderr),
        show_time=False,
        markup=True,
    )
)


def main():
    cli()


if __name__ == "__main__":
    main()
"""
Set up the command line client with different commands to execute
"""


@click.group()
@click.version_option(EVE.__version__)
def cli():
    pass


"""
Train EVE model
"""


@cli.command()
@click.option(
    "--MSA_data_folder",
    type=str,
    help="Folder where MSAs are stored",
)
@click.option(
    "--MSA_list", type=str, help="List of proteins and corresponding MSA file name"
)
@click.option(
    "--protein_index", type=int, help="Row index of protein in input mapping file"
)
@click.option(
    "--MSA_weights_location",
    type=str,
    help="Location where weights for each sequence in the MSA will be stored",
)
@click.option(
    "--theta_reweighting", type=float, help="Parameters for MSA sequence re-weighting"
)
@click.option(
    "--VAE_checkpoint_location",
    type=str,
    help="Location where VAE model checkpoints will be stored",
)
@click.option(
    "--model_name_suffix",
    default="Jan1",
    type=str,
    help="model checkpoint name will be the protein name followed by this suffix",
)
@click.option(
    "--model_parameters_location", type=str, help="Location of VAE model parameters"
)
@click.option(
    "--training_logs_location", type=str, help="Location of VAE model parameters"
)
def train(
    MSA_data_folder,
    MSA_list,
    protein_index,
    MSA_weights_location,
    theta_reweighting,
    VAE_checkpoint_location,
    model_name_suffix,
    model_parameters_location,
    training_logs_location,
):
    """ Train a Scaden model """
    train_EVE(
        MSA_data_folder=MSA_data_folder,
        MSA_list=MSA_list,
        protein_index=protein_index,
        MSA_weights_location=MSA_weights_location,
        theta_reweighting=theta_reweighting,
        VAE_checkpoint_location=VAE_checkpoint_location,
        model_name_suffix=model_name_suffix,
        model_parameters_location=model_parameters_location,
        training_logs_location=training_logs_location,
    )
