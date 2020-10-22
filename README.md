# Approximate Cross-Validation for Structured Models
This repository contains code for approximate cross-validation for structred models based on the infinitesimal jackknife as developed in [1].


### Requirements
```setup
conda env create -f environment.yml
conda activate approxcv
```

### Illustration
To perform LWCV on a AR(0) hidden Markov model trained on synthetic data, 

```run
python ./leave_within_structure_hmm.py
```

To perform LWCV on a discrete MRF, 

```run
python ./leave_within_structure_mrf.py
```

### References
[1] Ghosh, Soumya*, William T. Stephenson*, Tin D. Nguyen, Sameer K. Deshpande, and Tamara Broderick. [Approximate Cross-Validation for Structured Models](https://arxiv.org/abs/2006.12669). NeurIPS 2020. 

<sup>*</sup> Equal contribution
