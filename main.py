import argparse
from models import *
from dataset import *

sklearn_models = ['PLDA', 'KNN', 'RF', 'LR', 'SVM']
neural_network_models = ['MLP', 'ShallowCNN', 'VGG16', 'InceptionV3', 'DenseNet']
parser = argparse.ArgumentParser(description='P1 Cell Image Classification')
parser.add_argument('--space', choices=['raw', 'wndchrm', 'rcdt'], required=True)
parser.add_argument('--model', choices=sklearn_models+neural_network_models, required=True)
parser.add_argument("--transfer-learning",
                    help='neural network use pretrained weights instead of training from scratch',
                    action='store_true')
parser.add_argument("--PCA-comps", type=int, default=-1, help='number of PCA components if use PCA')
parser.add_argument("--PLDA-comps", type=int, default=100, help='number of components if use PLDA')
parser.add_argument("--PLDA-alpha", type=float, default=100.0, help='PLDA alpha')
parser.add_argument("--SVM-kernel", type=str, default='linear')
parser.add_argument("--CV", help='perform cross validation', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.space == 'wndchrm' and args.model in ['ShallowCNN', 'VGG16', 'InceptionV3']:
        raise ValueError("not able to use {} on {}".format(args.model, args.space))

    print("classification on {} space using {}...".format(args.space, args.model))
    dataset = load_dataset(args.space)

    if args.model in neural_network_models:
        nn_clf(args.model, dataset, args)
    else:
        sklearn_clf(args.model, dataset, args)
