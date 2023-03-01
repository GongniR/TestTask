import os
import SimpleITK as sitk
import numpy as np
import cv2
from tqdm import tqdm
import time
'''
Вспомогательные функции 
'''


#  просмотр файлов для генерации датасета
def view_path_dataSet(list_path_files: list[str], list_path_masks: list[str]) -> None:
    """Вывести список исследований и масок сегментации"""
    for i in range(len(list_path_files)):
        print(list_path_files[i], '   ', list_path_masks[i])


def normalize(x):
    """Нормализация изображений"""
    min_in = np.min(x)
    max_in = np.max(x)
    return (x - min_in) / (max_in - min_in + 1e-8)


def read_nii(path_file: str, binary: bool = False) -> np.array:
    """Прочитать файл .nii и преобразовать в массив """
    images = sitk.ReadImage(path_file)
    images_array = sitk.GetArrayFromImage(images)
    if binary:
        for i in range(len(images_array)):
            (thresh, image) = cv2.threshold(images_array[i], 0, 255, cv2.THRESH_BINARY)

            images_array[i, ...] = image

    return images_array


def save_image_for_folder(path_save: str, list_images: list[str], list_masks: list[str]) -> None:
    """Сохранить изображения и маски исследования в директорию"""
    save_number = 0
    for i in range(len(list_images)):
        image = read_nii(list_images[i])
        mask = read_nii(list_masks[i], binary=True)
        print(list_images[i])
        time.sleep(1)
        for j in tqdm(range(len(image))):
            if (np.count_nonzero(mask[j])>0):
                cv2.imwrite(os.path.join(path_save, f'images\\{save_number}_img.jpg'), normalize(image[j])*255)
                cv2.imwrite(os.path.join(path_save, f'masks\\{save_number}_mask.jpg'), mask[j])
                save_number+=1



def create_folder(name_dataset: str = None) -> None:
    if not os.path.exists('DataSet'):
        os.makedirs('DataSet')
        os.makedirs('DataSet/train')
        os.makedirs('DataSet/valid')
        os.makedirs('DataSet/test')
