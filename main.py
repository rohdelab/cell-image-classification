import argparse
from models import *
from dataset import *

sklearn_models = ['RF', 'KNN', 'SVM', 'LR', 'LDA', 'PLDA']
neural_network_models = ['MLP', 'ShallowCNN', 'VGG16', 'InceptionV3']
parser = argparse.ArgumentParser(description='P1 Cell Image Classification')
parser.add_argument('--dataset', default='hela')
parser.add_argument('--space', choices=['image', 'wndchrm', 'rcdt'], required=True)
parser.add_argument('--model', choices=sklearn_models+neural_network_models, required=True)
parser.add_argument("--transfer-learning",
                    help='neural network use pretrained weights instead of training from scratch',
                    action='store_true')
parser.add_argument("--SVM-kernel", type=str, choices=['rbf', 'linear'], default='linear')
parser.add_argument("--reproduce",
                    help='reproduce the results reported in the paper',
                    action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.space == 'wndchrm' and args.model in ['ShallowCNN', 'VGG16', 'InceptionV3']:
        raise ValueError("not able to use {} on {}".format(args.model, args.space))

    print("classification on {} space using {}...".format(args.space, args.model))
    if args.reproduce:
        dataset = load_dataset_reproduce(args.dataset, args.space)
    else:
        dataset = load_dataset(args.dataset, args.space)

    if args.model in neural_network_models:
        nn_clf(args.model, dataset, args)
    else:
        sklearn_clf(args.model, dataset, args)
