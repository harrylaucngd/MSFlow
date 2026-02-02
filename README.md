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
- To download data used for training MSFlow, please follow the same steps for download/preprocessing of data as illustrared in the repository [DiffMS](https://github.com/coleygroup/DiffMS). You need to clone DiffMS repository into the [ms_scripts](ms_scripts) directory for obtaining identical train/validation and also test sets.
- Then, you can derive CDDD representations for all datasets as illustrated in the repository [CDDDs](https://github.com/jrwnter/cddd)
### Encoder training:
- You can use CANOPUS and MassSpyGym training and validation data for training MS-CDDD decoder.
- You can check the original repository for retraining MIST using the provided script [src/train_mist.py]([https://github.com/samgoldman97/mist/blob/main_v2/src/mist/train_mist.py) but with CDDD representations.

### Decoder training:
- After downloading the necessary training data, you can use [convert_smiles_to_safe.py](convert_smiles_to_safe.py) script for pre-processing decoder training and validation datasets and converting smiles into SAFE representation.
- For training the flow decoder, you can run [cfg_pretrain.py](cfg_pretrain.py). You will need to set the paths in [config.py](configs/data.py) to match the data directory.

## Inference with model weights
We  provide weights for our encoder-decoder pipeline for running inference [here](https://zenodo.org).
- For MS-to-CDDD inference:  you need to use [condition_inference.py](ms_scripts/condition_inference.py). This script depends on DiffMS repo, so make sure it is cloned inside and working. We advice to create a seperate conda environment for encoder inference following the authors instructions but using our provided encoder checkpoints. You can use this script to save MS embeddings to an output dataframe.
- Additionally, we provide some examples for running decoder inference using [inference.py](inference.py) that can be used after downloading the checkpoint and storing it in the existing checkpoints placeholder directory.

## License

MSFlow is released under the MIT license.

## Contact
If you have any inquiries, please reach out to ghaith.mqawass@tum.de
