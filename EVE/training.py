import os, sys
import pandas as pd
import json
import logging

from EVE.model import VAE_model
from EVE.utils import data_utils


logger = logging.getLogger(__name__)


def train_EVE(
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

    model_name = protein_name + "_" + model_name_suffix
    print("Model name: " + str(model_name))

    model_params = json.load(open(model_parameters_location))

    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42,
    )
    model = model.to(model.device)

    model_params["training_parameters"][
        "training_logs_location"
    ] = training_logs_location
    model_params["training_parameters"][
        "model_checkpoint_location"
    ] = VAE_checkpoint_location

    print("Starting to train model: " + model_name)
    model.train_model(
        data=data, training_parameters=model_params["training_parameters"]
    )

    print("Saving model: " + model_name)
    model.save(
        model_checkpoint=model_params["training_parameters"][
            "model_checkpoint_location"
        ]
        + os.sep
        + model_name
        + "_final",
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        training_parameters=model_params["training_parameters"],
    )