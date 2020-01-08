# Modeling emotion in complex stories: the Stanford Emotional Narratives Dataset

Codebase for the publication of the first version of the Stanford Emotional Narratives Dataset (SEND) (for Transactions on Affective Computing paper. Link: [on Arxiv](https://arxiv.org/pdf/1912.05008.pdf))

**Citation**: If you want to refer our work, please consider cite (citation is required if you want to use our SEND dataset): 

    @article{ong2019modeling,
      title={Modeling emotion in complex stories: the Stanford Emotional Narratives Dataset},
      author={Ong, Desmond and Wu, Zhengxuan and Tan, Zhi-Xuan and Reddan, Marianne and Kahhale, Isabella and Mattek, Alison and Zaki, Jamil},
      journal={IEEE Transactions on Affective Computing},
      year={2019},
      publisher={IEEE}
	}


## Stanford Emotional Narratives Dataset

Here, We first introduce the first version of the Stanford Emotional Narratives Dataset (SENDv1): a set of rich, multimodal videos of self-paced, unscripted emotional narratives, annotated for emotional valence over time. The complex narratives and naturalistic expressions in this dataset provide a challenging test for contemporary time-series emotion recognition models.

## SEND Usage
Researchers are welcomed to request the dataset. You can view the dataset here at [SEND homepage](https://github.com/StanfordSocialNeuroscienceLab/SEND)).

## Provided Models
In this section, we present several time-series approaches to
model valence ratings on the SENDv1. We implement:

### SVR 
a baseline (non-time-series) discriminative model, a Support Vector Regression (SVR)
### HMM 
a baseline generative model, a Hidden Markov Model (HMM)
### LSTM
a state-of-the-art discriminative Long Short-Term Memory (LSTM) model
### VRNN 
a state-of-the-art (deep) generative Variational Recurrent Neural Network (VRNN) model.


## Requirement

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the required packages mentioned in the ***requirement.txt***.

```bash
pip install -r requirements.txt
```

## Usage
You will need to go to the model's subdirectory, and run the following command.

For all the supported command,
```python
cd models/lstm
python train.py -H
```
With the default settings,
```python
cd models/lstm
python train.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
