# Informatics applications in medicine

### Technology stack

**Back-end**

   - Python 3.5 ([Read the docs](http://docs.python.org/3.5/))
   - Scikit Learn 0.17.1 ([Read the docs](http://http://scikit-learn.org/0.17/documentation.html))
   - Joblib 0.9.4 ([Read the docs](https:/pythonhosted.org/joblib/))
   - Pylint 1.5.4 ([Read the docs](http://www.pylint.org))
   - Numpy 1.11.0 ([Read the docs](http://www.numpy.org))
   - Scipy 0.17.0 ([Read the docs](http://www.scipy.org))
   - Openpyxl 2.3.3 ([Read the docs](https://openpyxl.readthedocs.org/en/default/))
   - Python MNIST 0.3 ([Read the docs](https://github.com/sorki/python-mnist))
   - Patool 1.12 ([Read the docs](http://wummel.github.io/patool/))

## Accuracy Results

Testing method  : Cross Validation

Number of folds : 10
### Ensemble Methods

### Multiclass Methods
#### Data Set = Hypertension
|                      | Threshold 0% | Threshold 5% | Threshold 10% |
|:--------------------:|:------------:|:------------:|:-------------:|
| **Algorithm**        | **Accuracy [%]** | **Accuracy [%]** |  **Accuracy [%]** |
| One Vs. One          |    70.894    |      n/a     |      n/a      |
| Dynamic One Vs. One  |    71.381    |    71.942    |     72.781    |
| One Vs. Rest         |    61.260    |      n/a     |      n/a      |
| Dynamic One Vs. Rest |    62.819    |    62.102    |     65.492    |
