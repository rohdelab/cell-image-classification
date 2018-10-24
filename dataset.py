import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wndcharm.FeatureVector import FeatureVector
from wndcharm.PyImageMatrix import PyImageMatrix


def extract_wndchrm_feats(gray_img):
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
    batch_feats = [extract_wndchrm_feats(im) for im in gray_imgs]
    return np.array(batch_feats)


def extract_wndchrm_feats_parallel(gray_imgs, nprocesses=40):
    from multiprocessing import Pool
    p = Pool(nprocesses)
    splits = np.array_split(gray_imgs, nprocesses)
    results = p.map(extract_wndchrm_feats_batch, splits)
    result = np.vstack(results)
    return result


def save_wndchrm_feats(dataset):
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_test, y_test = dataset['x_test'], dataset['y_test']
    assert x_train.max() <= 1.0
    assert x_test.max() <= 1.0
    x_train = (x_train * 255).astype(np.uint8)
    x_test = (x_test * 255).astype(np.uint8)
    x_train = extract_wndchrm_feats_parallel(x_train)
    x_test = extract_wndchrm_feats_parallel(x_test)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    dataset = {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'classnames': dataset['classnames']
    }
    np.savez('hela_wndchrm_feats224.npz', **dataset)


def load_data(root, target_size):
    x = []
    y = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            img_path = os.path.join(path, name)
            try:
                img = Image.open(img_path)
                if img.size != (target_size, target_size):
                    img = img.resize((target_size, target_size))
                x.append(np.asarray(img))
                y.append(path.split('/')[-1])
            except IOError:
                pass
    classnames = list(set(y))
    classname2idex = {name: i for i, name in enumerate(classnames)}
    y = np.array([classname2idex[name] for name in y])
    x = np.array(x).astype('float32')
    x = x / x.max()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return {'x_train': x_train, 'y_train': y_train,
            'x_test': x_test, 'y_test': y_test,
            'classnames': classnames}


def vis_data(x, y):
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 20))
    for img, label, ax in zip(x, y, axes.ravel()):
        ax.matshow(img[:, :], cmap='gray')
        ax.set_title(label)
        ax.set_axis_off()
    plt.savefig('hela.pdf')
    plt.show()


def load_dataset(space='image'):
    if space == 'image':
        dataset = load_data('data/hela', target_size=224)
    elif space == 'wndchrm':
        dataset = np.load('hela_wndchrm_feats224.npz')
    return dataset