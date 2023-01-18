# Learning Proper Multiclass Losses with the LegendreTron algorithm.

This repository contains code to learn proper multiclass losses by learning canonical links with the LegendreTron and Multinomial Logistic Regression algorithms.

## Environment Setup
Create a python3.10.6 virtualenv and start a terminal. Uncomment lines in the requirements.txt file if installing PyTorch with CUDA support.

Run 

```
pip install -r requirements.txt
```

## Directory Setup
Setup the data folder at the root of this repository with the below structure
```
/data/aloi/
/data/dna/
/data/fmnist/
/data/glass/
/data/iris/
/data/kmnist/
/data/letter/
/data/mnist/
/data/news20/
/data/satimage/
/data/sector/
/data/segment/
/data/Sensorless/
/data/svmguide2/
/data/usps/
/data/vehicle/
/data/vowel/
/data/wine/
```

Download the associated datasets from the below sources and move them into the matching subfolder in the above:
* https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
* http://yann.lecun.com/exdb/mnist/
* http://codh.rois.ac.jp/kmnist/index.html.en
* https://github.com/zalandoresearch/fashion-mnist

## Experiment run setup
cd to the root of this repository before running any of the below experiments.

## LIBSVM demo
Refer to legendretron_demo.ipynb to reproduce accuracy statistics.

## LIBSVM examples
Run implementation of LegendreTron:
```
python -m experiments.lt_libsvm --dataset "segment" --seed 123
```

Run implementation of LegendreTron with label noise:
```
python -m experiments.lt_libsvm --dataset "segment" --seed 123 --flip_labels --eta 0.2
```

Run implementation of Multinomial Logistic Regression:
```
python -m experiments.mlr_libsvm --dataset "segment" --seed 123
```

Run implementation of Multinomial Logistic Regression with label noise:
```
python -m experiments.mlr_libsvm --dataset "segment" --seed 123 --flip_labels --eta 0.2
```

## MNIST examples
Run implementation of LegendreTron on multiclass problem with 10 classes:
```
python -m experiments.lt_mnist --dataset "fmnist" --seed 123
```

Run implementation of Multinomial Logistic Regression on multiclass problem with 10 classes:
```
python -m experiments.mlr_mnist --dataset "fmnist" --seed 123
```

Run implementation of LegendreTron to classify odd (1,3,5,7,9) vs even numbers (0,2,4,6,8):
```
python -m experiments.lt_mnist --dataset "fmnist" --seed 123 --binary --pos 1 3 5 7 9 --neg 0 2 4 6 8
```

Run implementation of Multinomial Logistic Regression to classify odd (1,3,5,7,9) vs even numbers (0,2,4,6,8):
```
python -m experiments.mlr_mnist --dataset "fmnist" --seed 123 --binary --pos 1 3 5 7 9 --neg 0 2 4 6 8
```

## Acknowledgements
This repository contains code from the following sources:

* https://gitlab.com/cwalder/linkgistic 
* https://github.com/CW-Huang/CP-Flow
* https://github.com/giorgiop/loss-correction
