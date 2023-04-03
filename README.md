# CoNFies: Controllable Neural Face Avatars
[![arXiv](https://img.shields.io/badge/arXiv-2211.08610-red.svg)](https://arxiv.org/abs/2211.08610)
[![Website](https://img.shields.io/badge/website-up-yellow.svg)](https://confies.github.io/)

This is the official implementation for FG 2023 paper "CoNFies: Controllable Neural Face Avatars" 

* [Project Page](https://confies.github.io/)
* [Paper](https://arxiv.org/abs/2211.08610)
* [Video](https://www.youtube.com/watch?v=DLgfEofnaoA)

The codebase is based on [CoNeRF](https://github.com/kacperkan/conerf)
implemente in [JAX](https://github.com/google/jax), building on
[JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).


## Setup
The code uses the same environment as [CoNeRF](https://github.com/kacperkan/conerf). We test tested it using Python 3.8.

Set up an environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

    conda create --name XXX python=3.8

Install the required packages:

    pip install -r requirements.txt

For more details, please refer to [CoNeRF](https://github.com/kacperkan/conerf).


## Dataset
### Basic structure
The dataset uses the [same format as Nerfies](https://github.com/google/nerfies#datasets) for the image extraction and camera estimation.

### Annotation
The format of annotations is the same as [CoNeRF](https://github.com/kacperkan/conerf). Annotation files include `annotations.yml`, `[frame_id].json` and `mapping.yml`. Please refer to [CoNeRF](https://github.com/kacperkan/conerf) for more details. We use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) to generate the keypoints and Facial Action Units.

## Running
After preparing a dataset, you can train using command similar to [CoNeRF](https://github.com/kacperkan/conerf):

    export DATASET_PATH=/path/to/dataset
    export EXPERIMENT_PATH=/path/to/save/experiment/to
    python train.py --base_folder $EXPERIMENT_PATH --gin_bindings="data_dir='$DATASET_PATH'" --gin_configs configs/baselines/ours.gin

After training the model, you can do rendering using:

    python render_changing_attributes.py --base_folder $EXPERIMENT_PATH --gin_bindings="data_dir='$DATASET_PATH'" --gin_configs /path/to/experiment/config.gin


## Note
It is not the final version code and we will keep updating!

## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{yu2022confies,
  title={CoNFies: Controllable Neural Face Avatars},
  author={Yu, Heng and Niinuma, Koichiro and Jeni, Laszlo A},
  journal={arXiv preprint arXiv:2211.08610},
  year={2022}
}
```

