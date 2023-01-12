# Learning Proper Multiclass Losses with the LegendreTron algorithm.

This repository contains code to learn proper multiclass losses by learning canonical links with the LegendreTron and Multinomial Logistic Regression algorithms.

## Environment Setup
Create a python3.10.6 virtualenv and start a terminal. Uncomment lines in the requirements.txt file if installing PyTorch with CUDA support.

Run 

```
pip install -r requirements.txt
```

## Directory Setup
Setup the data subdirectory at the root of this reposity with the below folder structure
├── data
│   ├── segment
│   │      └── segment.scale

Datasets are available at:
* https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
* http://yann.lecun.com/exdb/mnist/
* http://codh.rois.ac.jp/kmnist/index.html.en
* https://github.com/zalandoresearch/fashion-mnist

## LIBSVM examples
Run implementation of LegendreTron:
```
python lt_libsvm.py --dataset "segment" --seed 0
```

Run implementation of LegendreTron with label noise:
```
python lt_libsvm.py --dataset "segment" --seed 0 --flip_labels --eta 0.2
```

Run implementation of Multinomial Logistic Regression:
```
python mlr_libsvm.py --dataset "segment" --seed 0
```

Run implementation of Multinomial Logistic Regression with label noise:
```
python mlr_libsvm.py --dataset "segment" --seed 0 --flip_labels --eta 0.2
```

## MNIST examples
Run implementation of LegendreTron:
```
python lt_mnist.py --dataset "kmnist" --seed 0
```

Run implementation of Multinomial Logistic Regression:
```
python mlr_mnist.py --dataset "kmnist" --seed 0
```

## Acknowledgements
This repository contains code from the following sources:

* https://gitlab.com/cwalder/linkgistic 
* https://github.com/CW-Huang/CP-Flow
* https://github.com/giorgiop/loss-correction
