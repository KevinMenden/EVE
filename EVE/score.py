import os
import numpy as np
import pandas as pd
import pickle
import tqdm
import json
from sklearn import mixture, linear_model, svm, gaussian_process

from EVE.utils import performance_helpers as ph, plot_helpers


def compute_eve_scores(
    input_evol_indices_location,
    protein_list,
    output_eve_scores_location,
    load_GMM_models,
    GMM_parameter_location,
    protein_GMM_weight,
    compute_EVE_scores,
    recompute_uncertainty_threshold,
    default_uncertainty_threshold,
    plot_histograms,
    plot_scores_vs_labels,
    labels_file_location,
    plot_location,
    verbose,
):

    mapping_file = pd.read_csv(protein_list, low_memory=False)
    protein_list = np.unique(mapping_file["protein_name"])
    list_variables_to_keep = ["protein_name", "mutations", "evol_indices"]
    all_evol_indices = pd.concat(
        [
            pd.read_csv(
                input_evol_indices_location + os.sep + protein + ".csv",
                low_memory=False,
            )[list_variables_to_keep]
            for protein in protein_list
            if os.path.exists(input_evol_indices_location + os.sep + protein + ".csv")
        ],
        ignore_index=True,
    )

    all_evol_indices = all_evol_indices.drop_duplicates()
    X_train = np.array(all_evol_indices["evol_indices"]).reshape(-1, 1)
    if verbose:
        print("Training data size: " + str(len(X_train)))
        print(
            "Number of distinct proteins in protein_list: "
            + str(len(np.unique(all_evol_indices["protein_name"])))
        )

    if load_GMM_models:
        dict_models = pickle.load(
            open(
                GMM_parameter_location + os.sep + "GMM_model_dictionary_" "rb",
            )
        )
        dict_pathogenic_cluster_index = pickle.load(
            open(
                GMM_parameter_location
                + os.sep
                + "GMM_pathogenic_cluster_index_dictionary_"
                "rb",
            )
        )
    else:
        dict_models = {}
        dict_pathogenic_cluster_index = {}
        if not os.path.exists(GMM_parameter_location + os.sep):
            os.makedirs(GMM_parameter_location + os.sep)
        GMM_stats_log_location = (
            GMM_parameter_location + os.sep + os.sep + "GMM_stats_" + ".csv"
        )
        with open(GMM_stats_log_location, "a") as logs:
            logs.write(
                "protein_name,weight_pathogenic,mean_pathogenic,mean_benign,std_dev_pathogenic,std_dev_benign\n"
            )

        main_GMM = mixture.GaussianMixture(
            n_components=2, covariance_type="full", max_iter=1000, n_init=30, tol=1e-4
        )
        main_GMM.fit(X_train)

        dict_models["main"] = main_GMM
        pathogenic_cluster_index = np.argmax(
            np.array(main_GMM.means_).flatten()
        )  # The pathogenic cluster is the cluster with higher mean value
        dict_pathogenic_cluster_index["main"] = pathogenic_cluster_index
        if verbose:
            inferred_params = main_GMM.get_params()
            print(
                "Index of mixture component with highest mean: "
                + str(pathogenic_cluster_index)
            )
            print("Model parameters: " + str(inferred_params))
            print("Mixture component weights: " + str(main_GMM.weights_))
            print("Mixture component means: " + str(main_GMM.means_))
            print("Cluster component cov: " + str(main_GMM.covariances_))
        with open(GMM_stats_log_location, "a") as logs:
            logs.write(
                ",".join(
                    str(x)
                    for x in [
                        "main",
                        np.array(main_GMM.weights_).flatten()[
                            dict_pathogenic_cluster_index["main"]
                        ],
                        np.array(main_GMM.means_).flatten()[
                            dict_pathogenic_cluster_index["main"]
                        ],
                        np.array(main_GMM.means_).flatten()[
                            1 - dict_pathogenic_cluster_index["main"]
                        ],
                        np.sqrt(
                            np.array(main_GMM.covariances_).flatten()[
                                dict_pathogenic_cluster_index["main"]
                            ]
                        ),
                        np.sqrt(
                            np.array(main_GMM.covariances_).flatten()[
                                1 - dict_pathogenic_cluster_index["main"]
                            ]
                        ),
                    ]
                )
                + "\n"
            )

        if protein_GMM_weight > 0.0:
            for protein in tqdm.tqdm(protein_list, "Training all protein GMMs"):
                X_train_protein = np.array(
                    all_evol_indices["evol_indices"][
                        all_evol_indices.protein_name == protein
                    ]
                ).reshape(-1, 1)
                if (
                    len(X_train_protein) > 0
                ):  # We have evol indices computed for protein on file
                    protein_GMM = mixture.GaussianMixture(
                        n_components=2,
                        covariance_type="full",
                        max_iter=1000,
                        tol=1e-4,
                        weights_init=main_GMM.weights_,
                        means_init=main_GMM.means_,
                        precisions_init=main_GMM.precisions_,
                    )
                    protein_GMM.fit(X_train_protein)
                    dict_models[protein] = protein_GMM
                    dict_pathogenic_cluster_index[protein] = np.argmax(
                        np.array(protein_GMM.means_).flatten()
                    )
                    with open(GMM_stats_log_location, "a") as logs:
                        logs.write(
                            ",".join(
                                str(x)
                                for x in [
                                    protein,
                                    np.array(protein_GMM.weights_).flatten()[
                                        dict_pathogenic_cluster_index[protein]
                                    ],
                                    np.array(protein_GMM.means_).flatten()[
                                        dict_pathogenic_cluster_index[protein]
                                    ],
                                    np.array(protein_GMM.means_).flatten()[
                                        1 - dict_pathogenic_cluster_index[protein]
                                    ],
                                    np.sqrt(
                                        np.array(protein_GMM.covariances_).flatten()[
                                            dict_pathogenic_cluster_index[protein]
                                        ]
                                    ),
                                    np.sqrt(
                                        np.array(protein_GMM.covariances_).flatten()[
                                            1 - dict_pathogenic_cluster_index[protein]
                                        ]
                                    ),
                                ]
                            )
                            + "\n"
                        )
                else:
                    if verbose:
                        print(
                            "No evol indices for the protein: "
                            + str(protein)
                            + ". Skipping."
                        )

        pickle.dump(
            dict_models,
            open(
                GMM_parameter_location + os.sep + os.sep + "GMM_model_dictionary_" "wb",
            ),
        )
        pickle.dump(
            dict_pathogenic_cluster_index,
            open(
                GMM_parameter_location
                + os.sep
                + os.sep
                + "GMM_pathogenic_cluster_index_dictionary_"
                "wb",
            ),
        )

    if plot_histograms:
        if not os.path.exists(plot_location + os.sep + "plots_histograms" + os.sep):
            os.makedirs(plot_location + os.sep + "plots_histograms" + os.sep)
        plot_helpers.plot_histograms(
            all_evol_indices,
            dict_models,
            dict_pathogenic_cluster_index,
            protein_GMM_weight,
            plot_location + os.sep + "plots_histograms" + os.sep + protein_list,
        )

    if compute_EVE_scores:
        if protein_GMM_weight > 0.0:
            all_scores = all_evol_indices.copy()
            all_scores["EVE_scores"] = np.nan
            all_scores["EVE_classes_100_pct_retained"] = ""
            for protein in tqdm.tqdm(protein_list, "Scoring all protein mutations"):
                try:
                    test_data_protein = all_scores[all_scores.protein_name == protein]
                    X_test_protein = np.array(
                        test_data_protein["evol_indices"]
                    ).reshape(-1, 1)
                    mutation_scores_protein = ph.compute_weighted_score_two_GMMs(
                        X_pred=X_test_protein,
                        main_model=dict_models["main"],
                        protein_model=dict_models[protein],
                        cluster_index_main=dict_pathogenic_cluster_index["main"],
                        cluster_index_protein=dict_pathogenic_cluster_index[protein],
                        protein_weight=protein_GMM_weight,
                    )
                    gmm_class_protein = ph.compute_weighted_class_two_GMMs(
                        X_pred=X_test_protein,
                        main_model=dict_models["main"],
                        protein_model=dict_models[protein],
                        cluster_index_main=dict_pathogenic_cluster_index["main"],
                        cluster_index_protein=dict_pathogenic_cluster_index[protein],
                        protein_weight=protein_GMM_weight,
                    )
                    gmm_class_label_protein = pd.Series(gmm_class_protein).map(
                        lambda x: "Pathogenic" if x == 1 else "Benign"
                    )

                    all_scores.loc[
                        all_scores.protein_name == protein, "EVE_scores"
                    ] = np.array(mutation_scores_protein)
                    all_scores.loc[
                        all_scores.protein_name == protein,
                        "EVE_classes_100_pct_retained",
                    ] = np.array(gmm_class_label_protein)
                except:
                    print("Issues with protein: " + str(protein) + ". Skipping.")
        else:
            all_scores = all_evol_indices.copy()
            mutation_scores = dict_models["main"].predict_proba(
                np.array(all_scores["evol_indices"]).reshape(-1, 1)
            )
            all_scores["EVE_scores"] = mutation_scores[
                :, dict_pathogenic_cluster_index["main"]
            ]
            gmm_class = dict_models["main"].predict(
                np.array(all_scores["evol_indices"]).reshape(-1, 1)
            )
            all_scores["EVE_classes_100_pct_retained"] = np.array(
                pd.Series(gmm_class).map(
                    lambda x: "Pathogenic"
                    if x == dict_pathogenic_cluster_index["main"]
                    else "Benign"
                )
            )

        len_before_drop_na = len(all_scores)
        all_scores = all_scores.dropna(subset=["EVE_scores"])
        len_after_drop_na = len(all_scores)

        if verbose:
            scores_stats = ph.compute_stats(all_scores["EVE_scores"])
            print("Score stats: " + str(scores_stats))
            print(
                "Dropped mutations due to missing EVE scores: "
                + str(len_after_drop_na - len_before_drop_na)
            )
        all_scores["uncertainty"] = ph.predictive_entropy_binary_classifier(
            all_scores["EVE_scores"]
        )

        if recompute_uncertainty_threshold:
            uncertainty_cutoffs_deciles, _, _ = ph.compute_uncertainty_deciles(
                all_scores
            )
            uncertainty_cutoffs_quartiles, _, _ = ph.compute_uncertainty_quartiles(
                all_scores
            )
            if verbose:
                print(
                    "Uncertainty cutoffs deciles: " + str(uncertainty_cutoffs_deciles)
                )
                print(
                    "Uncertainty cutoffs quartiles: "
                    + str(uncertainty_cutoffs_quartiles)
                )
        else:
            uncertainty_thresholds = json.load(
                open(default_uncertainty_threshold_file_location)
            )
            uncertainty_cutoffs_deciles = uncertainty_thresholds["deciles"]
            uncertainty_cutoffs_quartiles = uncertainty_thresholds["quartiles"]

        for decile in range(1, 10):
            classification_name = "EVE_classes_" + str((decile) * 10) + "_pct_retained"
            all_scores[classification_name] = all_scores["EVE_classes_100_pct_retained"]
            all_scores.loc[
                all_scores["uncertainty"] > uncertainty_cutoffs_deciles[str(decile)],
                classification_name,
            ] = "Uncertain"
            if verbose:
                print("Stats classification by uncertainty for decile #:" + str(decile))
                print(all_scores[classification_name].value_counts(normalize=True))
        if verbose:
            print("Stats classification by uncertainty for decile #:" + str(10))
            print(
                all_scores["EVE_classes_100_pct_retained"].value_counts(normalize=True)
            )

        for quartile in [1, 3]:
            classification_name = "GMM_class_" + str((quartile) * 25) + "_pct_retained"
            all_scores[classification_name] = all_scores["EVE_classes_100_pct_retained"]
            all_scores.loc[
                all_scores["uncertainty"]
                > uncertainty_cutoffs_quartiles[str(quartile)],
                classification_name,
            ] = "Uncertain"
            if verbose:
                print(
                    "Stats classification by uncertainty for quartile #:"
                    + str(quartile)
                )
                print(all_scores[classification_name].value_counts(normalize=True))

        all_scores.to_csv(
            output_eve_scores_location + os.sep + "all_EVE_scores_" + ".csv",
            index=False,
        )

    if plot_scores_vs_labels:
        labels_dataset = pd.read_csv(labels_file_location, low_memory=False)
        all_scores_mutations_with_labels = pd.merge(
            all_scores,
            labels_dataset[["protein_name", "mutations", "ClinVar_labels"]],
            how="inner",
            on=["protein_name", "mutations"],
        )
        all_PB_scores = all_scores_mutations_with_labels[
            all_scores_mutations_with_labels.ClinVar_labels != 0.5
        ].copy()
        if not os.path.exists(
            plot_location + os.sep + "plots_scores_vs_labels" + os.sep
        ):
            os.makedirs(plot_location + os.sep + "plots_scores_vs_labels" + os.sep)
        for protein in tqdm.tqdm(protein_list, "Plot scores Vs labels"):
            plot_helpers.plot_scores_vs_labels(
                score_df=all_PB_scores[all_PB_scores.protein_name == protein],
                plot_location=plot_location
                + os.sep
                + "plots_scores_vs_labels"
                + os.sep,
                output_eve_scores_filename_suffix="" + "_" + protein,
                mutation_name="mutations",
                score_name="EVE_scores",
                label_name="ClinVar_labels",
            )
