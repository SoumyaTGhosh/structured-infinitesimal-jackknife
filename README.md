# Approximate Cross-Validation for Structured Model
We provide code for performing leave-within-structure cross-validation (LWCV). 
Code has been tested on Python 3.8.3

### Requirements
```setup
conda env create -f environment.yml
```

### Illustration
To perform LWCV on a AR(0) hidden Markov model trained on synthetic data, run,

```run
python ./leave_within_structure_hmm.py
```

To perform LWCV on a discrete MRF, run,

```run
python ./leave_within_structure_mrf.py
```