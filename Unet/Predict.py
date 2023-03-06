import argparse

from Unet.train import *
from tensorflow.keras.utils import CustomObjectScope

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float64)
    w, h, _ = x.shape
    x = x / 255.0
    x_old = x.copy()
    x = cv2.resize(x, (256, 256))

    return x,x_old, (w,h)

def draw_mask(image: np.array, mask: np.array, alpha: float = 0.001) -> np.array:
    """Отобразить маску на изображении"""

    def mask2red(mask):
        """Перекрасить маску в красный цвет"""
        zeros = np.zeros_like(mask)
        mask = np.dstack((zeros, zeros, mask)).astype(np.float64)
        return mask

    return cv2.addWeighted(mask2red(mask), alpha, image, 1.0 - alpha, 0)


def predict_lung(path_image: str, loss_function: str = 'dice_coef_loss') -> None:
    """Сегментировать легкие на изображении"""
    with CustomObjectScope({'iou': iou, f'{loss_function}': dice_coef_loss}):
        model = tf.keras.models.load_model("E:\\GitHub\\TestTask\\Unet\\dice_19_epoch\\files\\model.h5")

    new_size_image, image, size = read_image(path_image)
    mask = model.predict(np.expand_dims(new_size_image, axis=0))[0] * 255
    image_with_mask = draw_mask(image, cv2.resize(mask, size))

    return image_with_mask



# parser = argparse.ArgumentParser(description='Prepare data for preprocessing')
# parser.add_argument('--imagePath', type=str)
# parser.add_argument('--lossFunction', type=str)
#
# args = parser.parse_args()
#
# imagePath = args.imagePath
# lossFunction = args.lossFunction
