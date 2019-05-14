# Cell Image Classification

Code for training and testing of a set of statistical machine learning models on the 2D HeLa dataset (https://ome.irp.nia.nih.gov/iicbu2008/hela/index.html).

## Dependencies

* TensorFlow 1.13.1 https://www.tensorflow.org/
* Keras https://keras.io/
* WND-CHARM, and its Python API https://github.com/wnd-charm/wnd-charm
* scikit-learn 0.18.1 <https://scikit-learn.org/stable/>

## Usage

```
usage: main.py [-h] [--dataset DATASET] --space {image,wndchrm,rcdt} --model
               {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3}
               [--transfer-learning] [--SVM-kernel {rbf,linear}] [--reproduce]

P1 Cell Image Classification

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET
  --space {image,wndchrm,rcdt}
  --model {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3}
  --transfer-learning   neural network use pretrained weights instead of
                        training from scratch
  --SVM-kernel {rbf,linear}
  --reproduce           reproduce the results on Hela dataset reported in the
                        paper
```

**Examples**

* Train A logistic regression model on image space: `python main.py --space image --model LR`

* Train A logistic regression model on WND-CHARM feature space: `python main.py --space wndchrm --model LR`

* Train InceptionV3 on image space: `python main.py --space image --model InceptionV3`

* Train InceptionV3 on image space by fine-tuning a pre-trained model (transfer learning): `python main.py --space image --model InceptionV3 --transfer-learning`

**Reproduce Hela Results**

We provide the data used for producing the Hela results as reported in the paper. The preprocessed data is  located in `data/hela_reproduce`. To reproduce the results, for example, using PLDA and wndchrm features, simple run `python main.py --dataset hela --space wndchrm --model PLDA --reproduce`.
