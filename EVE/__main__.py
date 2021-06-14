from genericpath import exists
import click
import sys
import rich
import rich.logging
import logging
import os
import EVE
from EVE.training import train_EVE
from EVE.evol_indices import compute_evol_indices
from EVE.score import compute_eve_scores

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
# @click.version_option(EVE.__version__)
@click.version_option("1.0.0")
def cli():
    pass


"""
Train EVE model
"""


@cli.command()
@click.option("--out", type=str, default="eve_output", help="Output directory")
@click.option(
    "--msa_data_folder",
    "-d",
    type=str,
    default=".",
    help="Folder where MSAs are stored",
)
@click.option(
    "--msa_list",
    "-l",
    type=str,
    help="List of proteins and corresponding MSA file name",
)
@click.option(
    "--protein_index",
    type=int,
    default=0,
    help="Row index of protein in input mapping file",
)
@click.option(
    "--theta_reweighting",
    type=float,
    default=0.2,
    help="Parameters for MSA sequence re-weighting",
)
@click.option(
    "--model_name_suffix",
    default="protein_model",
    type=str,
    help="model checkpoint name will be the protein name followed by this suffix",
)
@click.option(
    "--model_parameters_location",
    "-p",
    type=str,
    default="default_model_params.json",
    help="Location of VAE model parameters",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Select device for training (cuda | cpu)",
)
def train(
    out,
    msa_data_folder,
    msa_list,
    protein_index,
    theta_reweighting,
    model_name_suffix,
    model_parameters_location,
    device,
):
    """ Train EVE VAE """
    # Generate the output folders if they don't exist already
    os.makedirs(out, exist_ok=True)
    training_logs_location = os.path.join(out, "logs")
    MSA_weights_location = os.path.join(out, "msa_weigths")
    VAE_checkpoint_location = os.path.join(out, "model")
    os.makedirs(MSA_weights_location, exist_ok=True)
    os.makedirs(VAE_checkpoint_location, exist_ok=True)
    os.makedirs(training_logs_location, exist_ok=True)

    train_EVE(
        MSA_data_folder=msa_data_folder,
        MSA_list=msa_list,
        protein_index=protein_index,
        MSA_weights_location=MSA_weights_location,
        theta_reweighting=theta_reweighting,
        VAE_checkpoint_location=VAE_checkpoint_location,
        model_name_suffix=model_name_suffix,
        model_parameters_location=model_parameters_location,
        training_logs_location=training_logs_location,
        device=device,
    )


"""
Compute evolutionary indices
"""


@cli.command()
@click.option("--out", type=str, default="eve_output", help="Output directory")
@click.option(
    "--msa_data_folder",
    "-d",
    type=str,
    default=".",
    help="Folder where MSAs are stored",
)
@click.option(
    "--msa_list",
    "-l",
    type=str,
    help="List of proteins and corresponding MSA file name",
)
@click.option(
    "--protein_index",
    type=int,
    default=0,
    help="Row index of protein in input mapping file",
)
@click.option(
    "--theta_reweighting",
    default=0.2,
    type=float,
    help="Parameters for MSA sequence re-weighting",
)
@click.option(
    "--model_name_suffix",
    default="protein_model",
    type=str,
    help="model checkpoint name will be the protein name followed by this suffix",
)
@click.option(
    "--model_parameters_location",
    "-p",
    type=str,
    default="default_model_params.json",
    help="Location of VAE model parameters",
)
@click.option(
    "--computation_mode",
    type=str,
    default="all_singles",
    help="Computes evol indices for all single AA mutations or for a passed in list of mutations (singles or multiples) [all_singles,input_mutations_list]",
)
@click.option(
    "--output_evol_indices_filename_suffix",
    default="evol_indices_out",
    type=str,
    help="(Optional) Suffix to be added to output filename",
)
@click.option(
    "--num_samples_compute_evol_indices",
    "-n",
    type=int,
    default=2000,
    help="Num of samples to approximate delta elbo when computing evol indices",
)
@click.option(
    "--batch_size",
    default=2048,
    type=int,
    help="Batch size when computing evol indices",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Select device for training (cuda | cpu)",
)
def evol_indices(
    out,
    msa_data_folder,
    msa_list,
    protein_index,
    theta_reweighting,
    model_name_suffix,
    model_parameters_location,
    computation_mode,
    output_evol_indices_filename_suffix,
    num_samples_compute_evol_indices,
    batch_size,
    device,
):
    """ Compute evolutionary indices """

    # Generate paths to output folders and make new folders if necessary
    MSA_weights_location = os.path.join(out, "msa_weigths")
    VAE_checkpoint_location = os.path.join(out, "model")
    all_singles_mutations_folder = os.path.join(out, "mutations")
    output_evol_indices_location = os.path.join(out, "evol_indices")

    os.makedirs(all_singles_mutations_folder, exist_ok=True)
    os.makedirs(output_evol_indices_location, exist_ok=True)

    compute_evol_indices(
        MSA_data_folder=msa_data_folder,
        MSA_list=msa_list,
        protein_index=protein_index,
        MSA_weights_location=MSA_weights_location,
        theta_reweighting=theta_reweighting,
        VAE_checkpoint_location=VAE_checkpoint_location,
        model_name_suffix=model_name_suffix,
        model_parameters_location=model_parameters_location,
        computation_mode=computation_mode,
        all_singles_mutations_folder=all_singles_mutations_folder,
        mutations_location=all_singles_mutations_folder,
        output_evol_indices_location=output_evol_indices_location,
        output_evol_indices_filename_suffix=output_evol_indices_filename_suffix,
        num_samples_compute_evol_indices=num_samples_compute_evol_indices,
        batch_size=batch_size,
        device=device,
    )


"""
Train GMM and compute EVE scores
"""


@cli.command()
@click.option(
    "--input_evol_indices_location",
    type=str,
    help="Folder where all individual files with evolutionary indices are stored",
)
@click.option(
    "--input_evol_indices_filename_suffix",
    type=str,
    default="",
    help="Suffix that was added when generating the evol indices files",
)
@click.option(
    "--protein_list", type=str, help="List of proteins to be included (one per row)"
)
@click.option(
    "--output_eve_scores_location",
    type=str,
    help="Folder where all EVE scores are stored",
)
@click.option(
    "--output_eve_scores_filename_suffix",
    default="",
    type=str,
    help="(Optional) Suffix to be added to output filename",
)
@click.option(
    "--load_GMM_models",
    default=False,
    help="If True, load GMM model parameters. If False, train GMMs from evol indices files",
)
@click.option(
    "--GMM_parameter_location",
    default=None,
    type=str,
    help="Folder where GMM objects are stored if loading / to be stored if we are re-training",
)
@click.option(
    "--GMM_parameter_filename_suffix",
    default=None,
    type=str,
    help="Suffix of GMMs model files to load",
)
@click.option(
    "--protein_GMM_weight",
    default=0.3,
    type=float,
    help="Value of global-local GMM mixing parameter",
)
@click.option(
    "--compute_EVE_scores",
    default=False,
    help="Computes EVE scores and uncertainty metrics for all input protein mutations",
)
@click.option(
    "--recompute_uncertainty_threshold",
    default=False,
    help="Recompute uncertainty thresholds based on all evol indices in file. Otherwise loads default threhold.",
)
@click.option(
    "--default_uncertainty_threshold_file_location",
    default="./utils/default_uncertainty_threshold.json",
    type=str,
    help="Location of default uncertainty threholds.",
)
@click.option(
    "--plot_histograms",
    default=False,
    help="Plots all evol indices histograms with GMM fits",
)
@click.option(
    "--plot_scores_vs_labels",
    default=False,
    help="Plots EVE scores Vs labels at each protein position",
)
@click.option(
    "--labels_file_location",
    default=None,
    type=str,
    help="File with ground truth labels for all proteins of interest (e.g., ClinVar)",
)
@click.option(
    "--plot_location", default=None, type=str, help="Location of the different plots"
)
@click.option("--verbose", default=True, help="Print detailed information during run")
def score(
    input_evol_indices_location,
    input_evol_indices_filename_suffix,
    protein_list,
    output_eve_scores_location,
    output_eve_scores_filename_suffix,
    load_GMM_models,
    GMM_parameter_location,
    GMM_parameter_filename_suffix,
    protein_GMM_weight,
    compute_EVE_scores,
    recompute_uncertainty_threshold,
    default_uncertainty_threshold_file_location,
    plot_histograms,
    plot_scores_vs_labels,
    labels_file_location,
    plot_location,
    verbose,
):
    """ Train GMM and compute EVE scores"""
    compute_eve_scores(
        input_evol_indices_location,
        input_evol_indices_filename_suffix,
        protein_list,
        output_eve_scores_location,
        output_eve_scores_filename_suffix,
        load_GMM_models,
        GMM_parameter_location,
        GMM_parameter_filename_suffix,
        protein_GMM_weight,
        compute_EVE_scores,
        recompute_uncertainty_threshold,
        default_uncertainty_threshold_file_location,
        plot_histograms,
        plot_scores_vs_labels,
        labels_file_location,
        plot_location,
        verbose,
    )