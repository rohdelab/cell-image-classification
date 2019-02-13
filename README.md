# Cell Image Classification

Code for training and testing of a set of statistical machine learning models on the 2D HeLa dataset (https://ome.irp.nia.nih.gov/iicbu2008/hela/index.html).

## Dependencies

* TensorFlow https://www.tensorflow.org/
* Keras https://keras.io/
* WND-CHARM Python API https://github.com/wnd-charm/wnd-charm

## Usage

```
usage: main.py [-h] --space {raw,wndchrm,rcdt} --model
               {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3}
               [--transfer-learning] [--PCA-comps PCA_COMPS]
               [--PLDA-comps PLDA_COMPS] [--PLDA-alpha PLDA_ALPHA]
               [--SVM-kernel SVM_KERNEL]

P1 Cell Image Classification

optional arguments:
  -h, --help            show this help message and exit
  --space {raw,wndchrm,rcdt}
  --model {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3}
  --transfer-learning   neural network use pretrained weights instead of
                        training from scratch
  --PCA-comps PCA_COMPS
                        number of PCA components if use PCA
  --PLDA-comps PLDA_COMPS
                        number of components if use PLDA
  --PLDA-alpha PLDA_ALPHA
                        PLDA alpha
  --SVM-kernel SVM_KERNEL
```

**Examples**

* Train A logistic regression model on image space: `python main.py --space raw --model LR`

* Train A logistic regression model on WND-CHARM feature space: `python main.py --space wndchrm --model LR`

* Train InceptionV3 on image space: `python main.py --space raw --model InceptionV3`

* Train InceptionV3 on image space by fine-tuning a pre-trained model (transfer learning): `python main.py --space image --model InceptionV3 --transfer-learning`

## Model Performances (10-fold cross validation accuracy)
* Image Space (before preprocessing)
    * SVM with linear kernel: 0.46 (+/- 0.07)
    * Logistic Regression: 0.44 (+/- 0.09)
    * KNN: 0.44 (+/- 0.09)
    * Random Forest: 0.44 (+/- 0.09)
* Image Space (after preprocessing)
    * SVM with linear kernel: 0.54 (+/- 0.10)
    * Logistic Regression: 0.51 (+/- 0.13)
    * KNN: 0.52 (+/- 0.07)
    * Random Forest: 0.53 (+/- 0.08)
    * LDA: 0.51 (+/- 0.06)
* WND-CHRM feature space (before preprocessing)
    * SVM with linear kernel: 0.92 (+/- 0.04)
    * Logistic Regression: 0.90 (+/- 0.07)
    * KNN: 0.83 (+/- 0.05)
    * Random Forest: 0.74 (+/- 0.06)
* WND-CHRM feature space (after preprocessing)
    * SVM with linear kernel: 0.87 (+/- 0.09)
    * Logistic Regression: 0.85 (+/- 0.05)
    * KNN: 0.76 (+/- 0.04)
    * Random Forest: 0.71 (+/- 0.06)
    * LDA: 0.86 (+/- 0.08)
    