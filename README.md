
            ====================================
              WELD BREAK PREDICTION CHALLENGE
            ====================================
In a production line a steel strip comes to the cold rolling machine in the form of a coil. When a coil is about to be finished, its end is welded with the head of the next coil. This results in a weld in the steel strip. The weld is
smoothed in what is known as scarfing process. The resulting steel strip is pickled on the pickling line and then thrown onto tandem mill. At tandem mill the strip is pressurized to reduce the thickness. In this particular scenario, we address a weld
break prediction for a company that welds the end of steel coils together in their production process. Often their production line succeeds in doing this. However, there are cases after the welding, in a subsequent step of the production line, the weld breaks. This is unfortunate because the company has to stop the production line, get the steel coil out of a machine and fix the broken steel. Visualizing one csv file which corresponds to one coil that might or might not have broken and the additional description files; one can understand that each file contains static parameters and a temporal information of dynamic variables measured during the welding process. We can also understand each row is one time step and each time step contains measurements from different sensors. The names corresponding to the values can be found at the top of each csv file. The “OK” folder containing measurements of healthy coils has 2970 items where the “WB” folder containing measurements of broken coils has 200 items, providing a total of 3170 samples. The provided dataset is highly skewed (exhibits a large imbalance) in the distribution of the target classes.

The full project report can be found here https://www.researchgate.net/publication/321360695_Recommender_system_for_Weld_break_prediction

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
