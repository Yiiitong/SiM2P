# Diffusion Bridge Networks Simulate Clinical-grade PET from MRI for Dementia Diagnostics

Official Implementation of Diffusion Bridge Networks Simulate Clinical-grade PET from MRI for Dementia Diagnostics.

<p align="center">
  <img src="img/archi_sim2p.svg" width="100%"/>
</p>


## Installation

1. Create environment: `conda env create -n sim2p --file requirements.yaml`
2. Activate environment: `conda activate sim2p`


## Data

We used public datasets from [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/) and [Japanese Alzheimer's Disease Neuroimaging Initiative (J-ADNI)](https://pubmed.ncbi.nlm.nih.gov/29753531/). Since we are not allowed to share our data, you would need to process the data yourself. Data for training, validation, and testing should be stored in separate [HDF5](https://docs.h5py.org/en/latest/quick.html) files, using the following hierarchical format:

1. First level: A unique identifier, e.g. image ID.
2. The second level always has the following entries:
    1. A group named `MRI/T1`, containing the T1-weighted 3D MRI data.
    2. A group named `PET/FDG`, containing the 3D FDG PET data.
    3. A dataset named `tabular` of size 13, containing a list of non-image clinical data, including age, gender, education level, MRI brain segmentation volumes obtained by Freesurfer including cerebrospinal fluid volume, the total grey matter volume, cortical white matter volume, left hippocampus volume, right hippocampus volume, left entorhinal thickness, right entorhinal thickness, cognitive examination scores MMSE, ADAS-Cog-13, and genetic risk factor ApoE4.
    4. A string attribute `DX` containing the diagnosis labels: `CN`, `Dementia` or `MCI`, if available.


Finally, the HDF5 file should also contain the following meta-information in a separate group named `stats`:
```bash
/stats/tabular           Group
/stats/tabular/columns   Dataset {13}
/stats/tabular/mean      Dataset {13}
/stats/tabular/stddev    Dataset {13}
```
They are the names of the features in the clinical data, their mean, and standard deviation.


## Model training and evaluation

We provide bash files [train_sim2p.sh](train_sim2p.sh) and [test_sim2p.sh](test_sim2p.sh) for model training and evaluation. Important model variables can be set in the bash file [args.sh](args.sh). For example, the length of the clincial data input can be accustomized and assigned to `TAB_DIM`.

To train, run
```
bash train_sim2p.sh

# to resume, set CKPT to your checkpoint, or it will automatically resume from your last checkpoint based on your experiment name.

bash train_sim2p.sh $CKPT
```

For evaluation, you need to be set `MODEL_PATH` for your checkpoint to be evaluated. Setting `--save_syn_scans True` will save the generated synthetic scans into your experiment folder. To evaluate, run
```
bash test_sim2p.sh $MODEL_PATH --save_syn_scans True
```
This script will print and also save the evaluation scores into `.txt` and `.csv` files into your experiment folder.



## Acknowlegements

The codebase is inspired by [alexzhou907/DDBM](https://github.com/alexzhou907/DDBM) and [DiT-3D/DiT-3D](https://github.com/DiT-3D/DiT-3D). Thanks for their wonderful works.


## Citation

If you find this method and/or code useful, please consider giving a star 🌟 and citing the paper:

