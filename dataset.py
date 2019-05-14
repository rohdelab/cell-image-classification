import sys
sys.path.append('optimaltransport')

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wndcharm.FeatureVector import FeatureVector
from wndcharm.PyImageMatrix import PyImageMatrix
from optimaltransport.optrans.continuous import RadonCDT
from optimaltransport.optrans.utils import signal_to_pdf
from itertools import zip_longest
from sklearn.decomposition import PCA

# image_target_size = 224  # original size 382x382
# image_target_size = (447, 382)  # width, height
image_target_size = (224, 224)  # width, height


def extract_wndchrm_feats_single(gray_img):
    # grayscale image
    assert gray_img.ndim == 2
    matrix = PyImageMatrix()
    matrix.allocate(gray_img.shape[1], gray_img.shape[0])
    numpy_matrix = matrix.as_ndarray()
    numpy_matrix[:] = gray_img
    fv = FeatureVector(name='FromNumpyMatrix', long=True, original_px_plane=matrix)
    fv.GenerateFeatures(write_to_disk=False)
    return fv.values.astype(np.float32)


def extract_wndchrm_feats_batch(gray_imgs):
    batch_feats = [extract_wndchrm_feats_single(im) for im in gray_imgs]
    return np.array(batch_feats)


def extract_wndchrm_feats(gray_imgs):
    import multiprocessing
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    splits = np.array_split(gray_imgs, multiprocessing.cpu_count())
    results = p.map(extract_wndchrm_feats_batch, splits)
    result = np.vstack(results)
    return result


def save_wndchrm_feats(dataset, filename):
    x, y = dataset['x'], dataset['y']
    assert x.max() <= 1.0
    x = (x * 255).astype(np.uint8)
    x = extract_wndchrm_feats(x)
    dataset = {
        'x': x, 'y': y,
        'classnames': dataset['classnames']
    }
    np.savez(filename, **dataset)


def rcdt_transform(x, template=None, nprocesses=40):
    from multiprocessing import Pool
    p = Pool(nprocesses)
    splits = np.array_split(x, nprocesses)
    if template is None:
        template = np.ones(x.shape[1:]).astype('float32')
        template = template / template.sum()
    results = p.starmap(rcdt_transform_single, zip_longest(splits, [template], fillvalue=template))
    result = np.vstack(results)
    return result


def rcdt_transform_single(x, template):
    x_trans = []
    for i in range(x.shape[0]):
        img = signal_to_pdf(x[i])
        radoncdt = RadonCDT()
        rcdt = radoncdt.forward(template, img)
        x_trans.append(rcdt.astype('float32'))
    x_trans = np.array(x_trans)
    return x_trans


def save_rcdt_feats(dataset, filename):
    x, y = dataset['x'], dataset['y']
    x = rcdt_transform(x)
    dataset = {
        'x': x, 'y': y,
        'classnames': dataset['classnames']
    }
    np.savez(filename, **dataset)


def load_images(root, target_size):
    x = []
    y = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            img_path = os.path.join(path, name)
            try:
                img = Image.open(img_path)
                if img.size != target_size:
                    img = img.resize(target_size)
                x.append(np.asarray(img))
                y.append(path.split('/')[-1])
            except IOError:
                pass
    classnames = sorted(list(set(y)))
    classname2idex = {name: i for i, name in enumerate(classnames)}
    y = np.array([classname2idex[name] for name in y])
    x = np.array(x).astype('float32')
    x = x / x.max()
    return {'x': x, 'y': y, 'classnames': classnames}

def vis_data(x, y):
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 20))
    for img, label, ax in zip(x, y, axes.ravel()):
        ax.matshow(img[:, :], cmap='gray')
        ax.set_title(label)
        ax.set_axis_off()
    plt.savefig('hela.pdf')
    plt.show()


def load_dataset(dataset, space='image'):
    image_dir = os.path.join('data', dataset)
    if not os.path.isdir(image_dir):
        raise ValueError('Directory "{}" does not exit'.format(image_dir))

    wndchrm_feat_file = os.path.join(image_dir, '{}_wndchrm_feats{}.npz'.format(dataset, *image_target_size))
    rcdt_feat_file = os.path.join(image_dir, '{}_rcdt_feats{}.npz'.format(dataset, *image_target_size))

    if space == 'image':
        dataset = load_images(image_dir, target_size=image_target_size)
        print('loaded raw images')
    elif space == 'wndchrm':
        if not os.path.isfile(wndchrm_feat_file):
            print('precomputed wndchrm features not found, computing and saving {}...'.format(wndchrm_feat_file))
            save_wndchrm_feats(load_images(image_dir, target_size=image_target_size), wndchrm_feat_file)
        dataset = np.load(wndchrm_feat_file)
        print('loaded wndchrm features')
    elif space == 'rcdt':
        if not os.path.isfile(rcdt_feat_file):
            print('precomputed RCDT features not found, computing and saving {}...'.format(rcdt_feat_file))
            save_rcdt_feats(load_images(image_dir, target_size=image_target_size), rcdt_feat_file)
        dataset = np.load(rcdt_feat_file)
        print('loaded RCDT features')

    return dataset


def load_dataset_reproduce(dataset, space='image'):
    if dataset != 'hela':
        raise ValueError('Reproduction of experimental results only support Hela dataset')
    from scipy.io import loadmat
    data_space = {'image': 'raw1', 'wndchrm': 'wnd', 'rcdt': 'rcdt'}[space]
    prefix = {'image': 'I', 'wndchrm': 'W', 'rcdt': 'R'}[space]
    datadir = 'data/hela_reproduce/data1'
    y = loadmat(os.path.join(datadir, 'labels'))
    y = np.squeeze(y['label'])
    datadir = os.path.join(datadir, data_space, 'bcls')
    x = []
    for i in range(y.size):
        x.append(loadmat('{}/{}{}.mat'.format(datadir, prefix, i + 1))['xx'])
    return {'x': np.array(x), 'y': y}


if __name__ == '__main__':
    dataset = load_dataset(dataset='hela', space='image')
    print("dataset stats:")
    print("images {}, dimension: {}, "
          "number classes: {}".format(dataset['x'].shape[0],
                                      dataset['x'].shape[1:],
                                      len(dataset['classnames'])))
    print("classes: {}".format(dataset['classnames']))

    dataset = load_dataset(dataset='hela', space='wndchrm')

    dataset = load_dataset(dataset='hela', space='rcdt')


