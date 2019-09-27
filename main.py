import argparse
from models import *
from dataset import *

sklearn_models = ['RF', 'KNN', 'SVM', 'LR', 'LDA', 'PLDA']
neural_network_models = ['MLP', 'ShallowCNN', 'VGG16', 'InceptionV3', 'ResNet']
parser = argparse.ArgumentParser(description='P1 Cell Image Classification')
parser.add_argument('--dataset', default='hela')
parser.add_argument('--space', choices=['image', 'wndchrm', 'rcdt'], required=True)
parser.add_argument('--model', choices=sklearn_models+neural_network_models, required=True)
parser.add_argument("-T", "--transfer-learning",
                    help='neural network use pretrained weights instead of training from scratch',
                    action='store_true')
parser.add_argument("-U", "--data_augmentation",
                    help='use data augmentation for neural network based approaches',
                    action='store_true')
parser.add_argument("--splits", type=int, choices=range(2, 11), default=10, help='number of splits for cross-validation')
parser.add_argument("--SVM-kernel", type=str, choices=['rbf', 'linear'], default='linear')
parser.add_argument("--preprocessed",
                    help='reproduce the results on Hela dataset reported in the paper',
                    action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.space == 'wndchrm' and args.model in ['ShallowCNN', 'VGG16', 'InceptionV3']:
        raise ValueError("not able to use {} on {}".format(args.model, args.space))

    print("classification on {} space using {}...".format(args.space, args.model))
    if args.preprocessed:
        dataset = load_dataset_preprocessed(args.dataset, args.space)
    else:
        dataset = load_dataset(args.dataset, args.space, image_size=(256, 256))

    if args.model in neural_network_models:
        nn_clf(args.model, dataset, args)
    else:
        sklearn_clf(args.model, dataset, args)
