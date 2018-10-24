import keras
import numpy as np
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.applications import nasnet
from keras.applications import densenet
from keras.applications import resnet50
from keras.applications import vgg16


def report(classnames, clf, x_test, x_train, y_test, y_train):
    eval_f = clf.score if hasattr(clf, 'score') else clf.evaluate
    print("train accuracy: {}".format(eval_f(x_train, y_train)))
    print("test accuracy: {}".format(eval_f(x_test, y_test)))
    y_pred = clf.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    y_test = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    print(classification_report(y_test, y_pred, target_names=classnames))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def build_model(model_name, input_shape, num_classes):
    if model_name == 'MLP':
        model = Sequential()
        model.add(Dense(256, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    else:
        include_top = True
        if model_name == 'VGG16':
            model = vgg16.VGG16(weights='imagenet', include_top=include_top, input_shape=input_shape)
        elif model_name == 'DenseNet':
            model = densenet.DenseNet121(weights='imagenet', include_top=include_top, input_shape=input_shape)
        x = model.output
        if not include_top:
            x = Flatten()(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(input=model.input, output=predictions)

    return model


def nn_clf(model_name, dataset):
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_test, y_test = dataset['x_test'], dataset['y_test']
    if model_name == 'MLP':
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
    else:
        if x_train.ndim == 3:
            x_train = np.repeat(x_train[..., np.newaxis], 3, axis=3)
            x_test = np.repeat(x_test[..., np.newaxis], 3, axis=3)
        else:
            assert x_train.ndim == 4
            assert x_train.shape[-1] == 3

    classnames = dataset['classnames']

    y_train = keras.utils.to_categorical(y_train, num_classes=len(classnames))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(classnames))

    model = build_model(model_name, x_train.shape[1:], len(classnames))

    opt = keras.optimizers.RMSprop(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    early_stop = EarlyStopping(patience=20)
    model.fit(x_train, y_train, verbose=2, batch_size=32, epochs=500,
              validation_split=0.1, shuffle=True, callbacks=[early_stop])

    report(classnames, model, x_test, x_train, y_test, y_train)


def sklearn_clf(model_name, dataset):
    clf = {
        'RF': RandomForestClassifier(),
        'LR': LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=300),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(),
    }[model_name]

    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_test, y_test = dataset['x_test'], dataset['y_test']
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    classnames = dataset['classnames']
    clf.fit(x_train, y_train)

    report(classnames, clf, x_test, x_train, y_test, y_train)

    # scores = cross_val_score(clf, np.vstack([x_train, x_test]), np.hstack((y_train, y_test)), cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
