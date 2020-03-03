# Generative Adversarial Imputation Networks (GAIN)
Title: GAIN: Missing Data Imputation using Generative Adversarial Nets

Authors: Jinsung Yoon, James Jordon, Mihaela van der Schaar

Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," International Conference on Machine Learning (ICML), 2018.

Paper Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf

Appendix Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf

Description of the code

This code shows the implementation of GAIN on MNIST dataset.

1. Introducing 50% of missingness on MNIST dataset.

2. Recover missing values on MNIST datasets using GAIN.

3. Show the multiple imputation results on MNIST with GAIN.

------------------------------------

Add source codes for UCI Letter and Spam datasets (02/12/2019)

# Prerequsites

### Python 2 (tested on Python 2.7.15)

### Tensorflow 1 (tested on 1.13.1)

### Required packages: {tqdm, matplotlib}:

```
pip3 install {package}
```

The code implementing MICE requires Python 3 (tested on 3.6.6) and scikit-learn (tested on 0.21.2).

# Train GAIN

Create directory for each dataset; e.g., 

```
mkdir news
```

Run with arguments *dataname* *data file*; e.g., 

```
python train_GAIN.py news News.csv
```
# Benchmark estimators

Run with arguments *dataname*; e.g., 

```
python3 train_MICE.py news
```