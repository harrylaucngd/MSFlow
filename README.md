# MSFlow: a flow matching framework for guided de-novo drug elucidation from MassSpec data
This is the codebase for our preprint: [MSFlow: De novo molecular structure elucidation from mass spectra via flow matching](https://arxiv.org/).
For running the repo please follow the instructions:
## Environment installiation:
   - Install conda/miniconda if needed
   - Use [flow.yml](flow.yml) to create the necessary conda environment for using this codebase:
```
    conda env create -f flow.yml
    conda activate flow
```
## Data download/preprocessing
- To download data used for training MSFlow, please follow the same steps for download/preprocessing of data as illustrared in the [DiffMS] repository (https://github.com/coleygroup/DiffMS). You need to clone DiffMS repository into the [diffms_scripts](diffms_scripts) directory for obtaining identical train/validation and also test benchmarks for inference and for running [scripts](scripts). 
- After downloading the necessary training data, you can use [convert_smiles_to_safe.py](convert_smiles_to_safe.py) script for pre-processing training and validation datasets and converting smiles into SAFE representation.
- For training the pipeline of encoder/decoder using CDDDs representation, you need to first extract CDDDs for all training/validation datasets, as illustrated in the repository [CDDDs](https://github.com/jrwnter/cddd)

## Running the code
- For training the flow decoder, you can run [cfg_pretrain.py](cfg_pretrain.py). You will need to set the paths in [config.py](configs/data.py) to match the data directory. 
- For training MIST encoder you can run [train_enc_cddd.py]([scripts/train_enc_cddd.py)).
## Inference with model weights
We  provide weights for our encoder-decoder pipeline for running inference [here](https://zenodo.org).
After downloading, you can use [condition_inference.py](diffms_scripts/condition_inference.py) to save MS embeddings to a temporary dataframe. Then you can set the checkpoint path and the temporary dataframe path for running decoding using [compute_spec_parallel.py](compute_spec_parallel.py) script.

## License

MSFlow is released under the MIT license.

## Contact
If you have any inquiries, please reach out to ghaith.mqawass@tum.de
