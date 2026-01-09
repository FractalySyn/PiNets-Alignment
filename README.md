# PiNets and Explanatory Alignment


<a href="https://arxiv.org/abs/2601.04378"><img src="https://img.shields.io/badge/arXiv%20-%20preprint-%23FF0000?logo=arxiv&logoColor=white" alt="arXiv - preprint"></a>
<a href="https://github.com/FractalySyn/PiNets-Alignment?tab=MIT-1-ov-file"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License - MIT"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>



Official implementation of the paper: **"Aligned explanations in neural networks"**

```bibtex
@article{pinets2026,
  title={Aligned explanations in neural networks},
  author={Lobet, Corentin and Chiaromonte, Francesca},
  journal={arXiv preprint arXiv:2601.04378},
  year={2026}
}
```


## Usage

We provide two **standalone Jupyter notebooks** (`.ipynb`) to reproduce our main experiments. Each notebook is self-contained and includes a startup code chunk to download the necessary helper scripts and model weights located in the associated folder in the repo.

<a href="https://colab.research.google.com/github/FractalySyn/PiNets-Alignment/blob/main/1.%20Toyshapes.ipynb"><img src="https://img.shields.io/badge/ToyShapes-Open%20in%20Colab-orange?logo=googlecolab&logoColor=white" alt="ToyShapes - Open in Colab"></a>
<a href="https://colab.research.google.com/github/FractalySyn/PiNets-Alignment/blob/main/2.%20Floods.ipynb"><img src="https://img.shields.io/badge/Floods-Open%20in%20Colab-orange?logo=googlecolab&logoColor=white" alt="Floods - Open in Colab"></a>


### Reproducibility

Pretrained models and results files are provided in the repo. If the `RETRAIN` and `REEVALUATE` parameters are set to `False` in the notebooks these files will be loaded to skip most computations. 

To reproduce the models and results from scratch, simply set these parameters to `True` and make sure to use the random seeds and hyperparameters provided in the notebooks and config files. 

As of January 2026, the notebooks run flawlessly on both Colab and Kaggle servers, with both CPU and GPU (cuda) devices for Pytorch models. The notebooks automatically use `cuda` if available.

### Data

In the ToyShapes experiments (`1. Toyshapes.ipynb` notebook) the images are generated on the fly and thus no dataset is required. The data generating functions are located in the `toyshapes/data.py` script. 

As for the segmentation task — implemented in the `2. Floods.ipynb` notebook — the code will download the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset from Google Cloud Storage (using `gsutil`) and the [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M) foundation model from HuggingFace (using `git`). 

### Playground

To conduct experiments with different settings, the hyperparameters can be edited in the notebooks. The parameters for the generation of the synthetic ToyShapes datasets can be edited both in the `1. Toyshapes.ipynb` notebook and in the `toyshapes/data.json` config file directly. 

   


<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>