import os, sys
import json
import argparse
import pandas as pd
import torch

from EVE.model import VAE_model
from EVE.utils import data_utils


def compute_evol_indices(
    MSA_data_folder,
    MSA_list,
    protein_index,
    MSA_weights_location,
    theta_reweighting,
    VAE_checkpoint_location,
    model_name_suffix,
    model_parameters_location,
    computation_mode,
    all_singles_mutations_folder,
    mutations_location,
    output_evol_indices_location,
    output_evol_indices_filename_suffix,
    num_samples_compute_evol_indices,
    batch_size,
):

    mapping_file = pd.read_csv(MSA_list)
    protein_name = mapping_file["protein_name"][protein_index]
    msa_location = (
        MSA_data_folder + os.sep + mapping_file["msa_location"][protein_index]
    )
    print("Protein name: " + str(protein_name))
    print("MSA file: " + str(msa_location))

    if theta_reweighting is not None:
        theta = theta_reweighting
    else:
        try:
            theta = float(mapping_file["theta"][protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: " + str(theta))

    data = data_utils.MSA_processing(
        MSA_location=msa_location,
        theta=theta,
        use_weights=True,
        weights_location=MSA_weights_location
        + os.sep
        + protein_name
        + "_theta_"
        + str(theta)
        + ".npy",
    )

    if computation_mode == "all_singles":
        data.save_all_singles(
            output_filename=all_singles_mutations_folder
            + os.sep
            + protein_name
            + "_all_singles.csv"
        )
        mutations_location = (
            all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv"
        )

    model_name = protein_name + "_" + model_name_suffix
    print("Model name: " + str(model_name))

    if os.path.exists(model_parameters_location):
        model_params = json.load(open(model_parameters_location))
    else:
        print("Using default model parameters")
        model_params = json.load(
            open(
                os.path.join(
                    os.path.dirname(__file__), "model", "default_model_params.json"
                )
            )
        )

    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42,
    )
    model = model.to(model.device)

    try:
        checkpoint_name = str(VAE_checkpoint_location) + os.sep + model_name + "_final"
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
    except:
        print("Unable to locate VAE model checkpoint")
        sys.exit(0)

    list_valid_mutations, evol_indices, _, _ = model.compute_evol_indices(
        msa_data=data,
        list_mutations_location=mutations_location,
        num_samples=num_samples_compute_evol_indices,
        batch_size=batch_size,
    )

    df = {}
    df["protein_name"] = protein_name
    df["mutations"] = list_valid_mutations
    df["evol_indices"] = evol_indices
    df = pd.DataFrame(df)

    evol_indices_output_filename = (
        output_evol_indices_location
        + os.sep
        + protein_name
        + "_"
        + str(num_samples_compute_evol_indices)
        + "_samples"
        + output_evol_indices_filename_suffix
        + ".csv"
    )
    try:
        keep_header = os.stat(evol_indices_output_filename).st_size == 0
    except:
        keep_header = True
    df.to_csv(
        path_or_buf=evol_indices_output_filename,
        index=False,
        mode="a",
        header=keep_header,
    )
