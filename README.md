# MSFlow: a flow matching framework for guided de-novo drug elucidation from MassSpec data
This is the codebase for our preprint: [MSFlow: flow matching framework for guided de-novo drug elucidation from MassSpec data](https://arxiv.org/).
For running the repo please follow the instructions:
## Environment installiation:
   - Install conda/miniconda if needed
   - Use flow.yml to create the necessary conda environment for using this codebase:
```
    conda env create -f flow.yml
    conda activate flow
```
## Data download/preprocessing
- To download data used for training MSFlow, please follow the same steps for download/preprocessing of data as illustrared in the [DiffMS] repository (https://github.com/coleygroup/DiffMS). You need to clone DiffMS repository into the [diffms_scripts](diffms_scripts) directory for obtaining identical train/validation and also test benchmarks for inference and for running [diffms_scripts](diffms_scripts). 
- After downloading the necessary training data, you can use [convert_smiles_to_safe.py](convert_smiles_to_safe.py) script for preproccessing training and validation datasets and converting smiles into SAFE representation.
- For training the pipeline of encoder/decoder using CDDDs representation, you need to first extract cddds for all training/validation/test datasets as illustrated in the repository [CDDDs](https://github.com/jrwnter/cddd)
- For training the (MS-Molecule) decoder, you need to use [detect_bounds.py](diffms_scripts/detect_bounds.py) and then [extract_ms.py](diffms_scripts/extract_ms.py) to embed MS into vectorized form. You can then use [convert_smiles_to_safe.py](convert_smiles_to_safe.py) script for preproccessing the resulting data and converting smiles into SAFE representation.
## Running the code
For pretraining (ECFP-molecule/CDDDs-molecule) decoder you can run [cfg_pretrain.py](cfg_pretrain.py). You will need to set the paths in [config.py](configs/data.py) to match the location of preprocessed data directory. 
For pretraining or the MIST encoder (MS-ECFPS/MS-CDDDs) you can run ([train_tune_enc_fp.py]([diffms_scripts/train_tune_enc_fp.py)
/[train_enc_cddd.py]([diffms_scripts/train_enc_cddd.py)).
For pretraining the (MS-Molecule)

## Inference with checkpoints
We also provide weights for the all MIST encoder/MSFlow modules for rnnung inference [here](https://zenodo.org).
After downloading, you can use [condition_inference.py](diffms_scripts/condition_inference.py) to save MS embeddings to a temp dataframe. Then you can set checkpoint path and temp dataframe path for running inference using [compute_spec_parallel.py](compute_spec_parallel.py) script.

## License

MSFlow is released under the MIT license.

## Contact
If you have any inquiries, please reach out to ghaith.mqawass@pfizer.com
