import sys
sys.path.append('optimaltransport')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from optimaltransport.optrans.decomposition import PLDA


def report(classnames, clf, x_test, x_train, y_test, y_train):
    eval_f = clf.score if hasattr(clf, 'score') else clf.evaluate
    train_acc, test_acc = eval_f(x_train, y_train), eval_f(x_test, y_test)
    if isinstance(train_acc, list):
        train_acc, test_acc = train_acc[-1], test_acc[-1]
    print("train accuracy: {}, test accuracy {}".format(train_acc, test_acc))

    y_pred = clf.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    y_test = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    print(classification_report(y_test, y_pred, target_names=classnames))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def build_model(model_name, input_shape, num_classes, transfer_learning):
    from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
    from keras.models import Sequential
    from keras.applications import densenet
    from keras.applications import vgg16
    from keras.applications import InceptionV3
    from keras.models import Model

    weights = 'imagenet' if transfer_learning else None
    if model_name == 'MLP':
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
    elif model_name == 'ShallowCNN':
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=[11, 5], strides=[3, 3], input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=[3, 3], strides=[2, 2]))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
    elif model_name == 'VGG16':
        base_model = vgg16.VGG16(weights=weights, include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights=weights, include_top=False, input_shape=input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(4096, activation='relu')(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    elif model_name == 'DenseNet':
        model = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        x = Flatten()(model.output)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=model.input, outputs=predictions)

    return model


def nn_clf(model_name, dataset, args):
    import keras
    from keras.callbacks import EarlyStopping
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_test, y_test = dataset['x_test'], dataset['y_test']
    input_shape = x_train.shape[1:]
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if model_name != 'MLP':
        x_train = np.reshape(x_train, [-1, *input_shape])
        x_test = np.reshape(x_test, [-1, *input_shape])
        if x_train.ndim == 3:
            x_train = np.repeat(x_train[..., np.newaxis], 3, axis=3)
            x_test = np.repeat(x_test[..., np.newaxis], 3, axis=3)
        else:
            assert x_train.ndim == 4
            assert x_train.shape[-1] == 3

    classnames = dataset['classnames']

    y_train = keras.utils.to_categorical(y_train, num_classes=len(classnames))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(classnames))

    if args.transfer_learning and model_name in ['VGG16', 'InceptionV3']:
        print("using pretrained weights {}".format(model_name))
    else:
        print("training {} from scratch...".format(model_name))

    model = build_model(model_name, x_train.shape[1:], len(classnames), args.transfer_learning)

    opt = keras.optimizers.Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    model.fit(x_train, y_train, verbose=2, batch_size=32, epochs=100,
              validation_split=0.1, shuffle=True, callbacks=[early_stop])

    report(classnames, model, x_test, x_train, y_test, y_train)


def sklearn_clf(model_name, dataset, args):
    clf = {
        'RF': RandomForestClassifier(),
        'LR': LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=300),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(kernel=args.SVM_kernel, class_weight='balanced', probability=True, decision_function_shape='ovr'),
        'PLDA': PLDA(args.PLDA_alpha, args.PLDA_comps)
    }[model_name]

    print("normalizing features...")
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_test, y_test = dataset['x_test'], dataset['y_test']
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if args.PCA_comps > 0:
        print("computing PCA {}-> {}...".format(x_train.shape[1], args.PCA_comps))
        pcai = PCA(n_components=args.PCA_comps)
        x_train = pcai.fit_transform(x_train)
        x_test = pcai.transform(x_test)

    classnames = dataset['classnames']
    clf.fit(x_train, y_train)
    report(classnames, clf, x_test, x_train, y_test, y_train)

    # scores = cross_val_score(clf, np.vstack([x_train, x_test]), np.hstack((y_train, y_test)), cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
