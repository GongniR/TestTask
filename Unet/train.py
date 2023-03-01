import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Unet.model_Unet import *

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K

def read_image(path):
    """Чтение изображения"""
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask(path):
    """Чтение маски"""
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)

    return x


def tf_parse(x, y):
    """Парсер данных"""

    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y


def tf_dataset(x, y, batch):
    """Получения датасета для обучения"""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


def iou(y_true, y_pred):
    """Метрика сходимости"""

    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float64)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float64)


def load_data(path_data: str) -> [list[str], list[str]]:
    """Загрузка наборов данных для обучения"""
    data_x = sorted([os.path.join(path_data, f"images\\{path}") for path in os.listdir(os.path.join(path_data, "images\\"))], key=lambda a: int(a.split('_')[0].split('\\')[-1]))
    data_y = sorted([os.path.join(path_data, f"masks\\{path}") for path in os.listdir(os.path.join(path_data, "masks\\"))], key=lambda a: int(a.split('_')[0].split('\\')[-1]))
    return data_x, data_y


path_train = 'E:\\GitHub\\TestTask\\GenerateDataSet\\DataSet\\train'
path_valid = 'E:\\GitHub\\TestTask\\GenerateDataSet\\DataSet\\valid'

# Загрузка тренировочного и валидационного набора
train_x, train_y = load_data(path_train)
valid_x, valid_y = load_data(path_valid)


# Гиперпараметры
batch = 32
lr = 1e-4
epochs = 1000

# dataset
train_dataset = tf_dataset(train_x, train_y, batch=batch)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth = 10e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)
# Модель
model = build_model()
opt = tf.keras.optimizers.Adam(lr)
metrics = ["acc", iou]
model.compile(loss=dice_coef_loss, optimizer=opt, metrics=metrics)



callbacks = [
    ModelCheckpoint("files/model.h5"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    CSVLogger("files/data_TrainValid.csv"),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

train_steps = len(train_x)//batch
valid_steps = len(valid_x)//batch

if len(train_x) % batch != 0:
    train_steps += 1
if len(valid_x) % batch != 0:
  valid_steps += 1

print(train_steps)
print(valid_steps)

if __name__ == "__main__":
    model.fit(train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            steps_per_epoch = len(train_x),
            validation_steps = len(valid_x),
            callbacks=callbacks)
