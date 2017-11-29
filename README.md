
            ====================================
              WELD BREAK PREDICTION CHALLENGE
            ====================================

**Author**
- [Bereket Abera Yilma](bereketabera.yilma@studenti.unitn.it)

* * *

## Requirements

- [numpy](www.numpy.org/)
- [scipy](www.scipy.org/)
- [scikit-learn](scikit-learn.org/)
- [Pandas](pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

This code  works on Python 2.7 or later.

* * *

## Running the  experiments

Run the `Run.sh` runner script from a terminal, like this:
```
$ bash Run.sh
```
 This script calls the `preprocess.py` script to preprocess datasets from OK and WB folders,
 the  preprocessing results will be dumped into the directory as pickle files
 then it calls the `Classyfy.py` which loads the preprocessed data and performs the classification task
 and displays results

* * *
# Weld-break-prediction
