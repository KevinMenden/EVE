# Variant effect prediction with EVE

This repository was forked and adapted from: https://github.com/OATML/EVE

The code was adapted to make the algorithm easier usable in a pipeline setting.

## Usage

To generate variant effect predictions, the following data is required:

* A copy of the [Uniref100 protein database](https://www.uniprot.org/uniref/?query=&fil=identity:1.0) 
* the sequence of your query protein (in Fasta format)

The procedure then consists of the following steps:

* Generate multiple sequence alignment (MSA) with [jackhmmer](http://hmmer.org/) using your query protein and the Uniref100 DB
* Train an EVE model on the MSA
* Compute evolutionary indices for all possible mutations using the trained EVE model
* Train a GMM and score evolutionary indices to classify mutations into three categories (pathogenic, benign, uncertain)

## MSA generation with EVcouplings

The input MSA is generated with the `align` stage of the [EVcouplings](https://github.com/debbiemarkslab/EVcouplings) pipeline.

The complete pipeline is configured via a config file, like this one: [example config](https://github.com/debbiemarkslab/EVcouplings/blob/develop/config/sample_config_monomer.txt)

To run EVcouplings, you simply have to the following command from withing the EVE docker image:

```bash
evcouplings my_config.txt
```

A config file to be used with this pipeline is provided here: TODO ADD THE EXAMPLE CONFIG FILE
You simple have to change the uniref ID in the config file to specify the protein that should be used for alignment. The align pipeline will be run several times with different sequence threshold, to produce different MSAs with differing leniency.

The best MSA then has to be selected according to the criteria of the EVE paper. With `L` as the length of the sequence and `N` being the number of sequences in the MSA, the MSA should fulfil the following criteria:

* `Lcov >= 0.8L`
* `100,000 >= N >= 10L`

If these cannot be fulfilled, they can be relaxed to:

* `Lcov >= 0.7L`
* `200,000 >= N >= 10L`

and so on. Using these criteria the best MSA is selected and used for EVE stage.
### MSA Format

This command generates a MSA in Stockholm format (`.sth`), however we need the A2M format (`.a2m`). So we need to convert the format using `esl-reformat` which is included in the HMMER suite of tools:

`esl-reformat a2m protein_msa.sth > protein_msa.a2m`

Finally, we need to reformat the name of our query sequence in the MSA to be compatible with the EVE pre-processing functions. Specifically, we need a header that is modified like so:

`>PROTEIN_NAME/start-end`

So in our example:

`>PROTEIN/1-819`

The query protein should be the first entry in the MSA, so this line should be the first one in the file.

### Further comments

In the EVE publication, the authors furthermore make sure the the MSA adheres to the following quality standards:

... TO BE FILLED OUT ...

## Run EVE

### Train EVE model

The first step is to train an EVE model.

As input, you currently need three files:

* your MSA
* a file pointing to your MSA (`example_mapping.csv`)
* a file with the model parameters (`default_model_parameters.json`)

Examples of the latter two files can be found in the assets folder. If you have those files, you can run the model:

`eve train -l example_mapping.csv -p default_model_params.json`

By default, this will use the GPU for training. You can have a look at other parameters using `eve train --help`

### Compute Evolutionary Indices

Next, we need to compute the evolutionary indices:

`eve evol-indices -l example_mapping.csv`

### Train GMM and score mutations

To score mutations using a GMM, use the following command:

`eve score -l example_mapping.csv`

### Notes

This workflow is not entirely ideal but we might put this into a Nextflow pipeline anyway so it might not be worth it to update.

## License

This project is available under the MIT license.

## Reference

If you use this code, please cite the following paper:

```bash
Large-scale clinical interpretation of genetic variants using evolutionary data and deep learning
Jonathan Frazer, Pascal Notin, Mafalda Dias, Aidan Gomez, Kelly Brock, Yarin Gal, Debora S. Marks
bioRxiv 2020.12.21.423785
doi: https://doi.org/10.1101/2020.12.21.423785
```

#### Archive

First we have to generate the MSA using jackhmmer according to the method specified in the paper:

* bitscore (`--incT` and `--incdomT`) of 0.3 * length_of_sequence (set both parameters to the threshold)
* 5 iterations (`-N 5`)

Example for protein of length 819, which gives a bitscore of 246 (round and use integers).

`jackhmer --cpu 8 -A protein_msa.sth --incT 246 -N 5 protein.fasta uniref100.fasta`