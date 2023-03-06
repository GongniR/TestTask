# TestTask
Найти датасет внутренних органов. 
   Для обучения был выбран датасет: https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans
   
   ######а) Понятность, простота кода. Визуализация этапов. Умение объяснить результаты.
   Задача была разделена на 3 этапа:
    1. Получения датасета из набора файлов .nii (Директория GenerateDataSet)
      * * Выборки составлялись из разных пациентов, удалялись срезы без маски сегментации * *
    2. Обучение модели U-net и формирования файлов для предсказания и визуализации (Директория Unet)
    3. Веб просмотр результата (Директория Web)
    
  ######б) Анализ полученного решения: где хорошо работает, где плохо, анализ подходов для улучшения.
    Была выбрана стандартная модель U-net для входного изображения 256x256. Модель обучалась 19 эпох с batch = 32, обучения приостановилось в результате срабатывания         EarlyStopping при уменьшения valid_loss.
    Процес обучения представлен на графиках: 
    Функция потерь dice_loss:
 
    Точность обучения Accuracy: 
  
    Метрика IOU:
    
    
    Результат работы сети:
    ![Image alt](https://github.com/GongniR/TestTask/blob/main/image/result.jpg)
    Входе тестов было обнаружена, что сеть теряет некоторые полигоны легких с матовым стеклом. 
    Лучше сего сеть сегментирует легкие  в случаях 6-8, хуже всего в случае 3 и срезах с большим кол-во матового стекла.
    
    Для улучшения сегментации:
      1. В выборку необходимо добавить здоровые легкие, без матового стекла 
      2. В выборку можно добавить срезы без легких
      3. Также рекомендуется реализовать кастомный генератор для загрузки датасета, с добавлением shuffle
      4. Добавление аугментации данных с использованием аффинных преобразований
      5. Подобрать функцию потерь 
      6. Изменить кол-во фильтров в архитектуре U-net
      
   ###### в) Представление результата. Наличие Flask API или другого web-сервинга модели.
      Платформа для просмотра:
      ![Image alt](https://github.com/GongniR/TestTask/blob/main/image/Web%20Image.png)
  Ожидаемый результат тестового задания: на вход фотография среза, на выход изображежние с сегментационной маской.
