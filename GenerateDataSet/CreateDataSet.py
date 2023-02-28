import os
import SimpleITK as sitk
import numpy as np
import cv2

import GenerateDataSet.utils as ult
'''
Файл для создания датасета на основе .nii файлов с кт с размеченными легкими 
https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans
'''

# пути к файлам датасета

# путь к директории с КТ исследованиям
path_folder_CT = "E:\\DataSet\\ct_scans"
# Путь к директории с маскам легких
path_folder_lung_mask = "E:\\DataSet\\lung_mask"


'''
1. Считать файлы КТ
2. Считать файлы масок
3. Создать директории для тренировочной и валидационной выборки  
4. Сохранить туда кт и маски 
'''

# пути для файлов внутри директории -> фильтрация на совпадения с "coronacases", чтобы исключить не нужные
# файлы для обработки

# пути для исследований КТ
list_path_CTs = [os.path.join(path_folder_CT, path_file_CT) for path_file_CT in
                 filter(lambda a: not (a.find('coronacases', 0)), os.listdir(path_folder_CT))]

# пути для масок КТ исследований
list_path_masks = [os.path.join(path_folder_lung_mask, path_file_mask) for path_file_mask in
                   filter(lambda a: not (a.find('coronacases', 0)), os.listdir(path_folder_lung_mask))]

# просмотр файлов и масок
# ult.view_path_dataSet(list_path_CTs, list_path_masks)

# разбить датасет на выборки
count = int(len(list_path_CTs)*0.75)
# 75 % - train
print('--------------------------<train>----------------------------------')
list_path_CTs_train = [path for path in list_path_CTs[:count]]
list_path_masks_train = [path for path in list_path_masks[:count]]
ult.view_path_dataSet(list_path_CTs_train, list_path_masks_train)

# Сохранить исследования для тренировочного набора данных
ult.save_image_for_folder('E:\\GitHub\\TestTask\\GenerateDataSet\\DataSet\\train\\', list_path_CTs_train, list_path_masks_train)

# 25 % - valid
print('--------------------------<valid>----------------------------------')
list_path_CTs_valid = [path for path in list_path_CTs[count:]]
list_path_masks_valid = [path for path in list_path_masks[count:]]
ult.view_path_dataSet(list_path_CTs_valid, list_path_masks_valid)

# Сохранить исследования для валидационного набора данных
ult.save_image_for_folder('E:\\GitHub\\TestTask\\GenerateDataSet\\DataSet\\valid\\', list_path_CTs_valid, list_path_masks_valid)
