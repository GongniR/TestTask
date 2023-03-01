from Unet.train import *
from tensorflow.keras.utils import CustomObjectScope
with CustomObjectScope({'iou': iou, 'dice_coef_loss': dice_coef_loss}):
  model = tf.keras.models.load_model("dice_19_epoch/files/model.h5")


def read_image1(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask1(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x
path_image = "E:\\GitHub\\TestTask\\GenerateDataSet\\DataSet\\valid\\images\\56_img.jpg"
path_mask =  "E:\\GitHub\\TestTask\\GenerateDataSet\\DataSet\\valid\\masks\\56_mask.jpg"

x = read_image1(path_image)
y = read_mask1(path_mask)
y_pred = model.predict(np.expand_dims(x, axis=0))[0]
cv2.imshow('image', x)
cv2.imshow('mask', y)
cv2.imshow('pred',y_pred*255 )
cv2.waitKey(0)