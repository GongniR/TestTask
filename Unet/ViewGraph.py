import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

csv_loger = "dice_19_epoch/files/data_TrainValid.csv"

data = pd.read_csv(csv_loger)



fig, ax = plt.subplots(figsize=(15, 15), linewidth = 3)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.plot(data["epoch"],data["loss"],label="loss")
plt.plot(data["epoch"],data["val_loss"], label="val_loss")

#  Устанавливаем интервал основных делений:
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
#  Устанавливаем интервал вспомогательных делений:
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

#  Тоже самое проделываем с делениями на оси "y":
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

"""plt.plot(data["epoch"],data["acc"])
plt.plot(data["epoch"],data["iou"])"""
ax.legend(loc = 1)


fig, ax = plt.subplots(figsize=(15, 15), linewidth = 3)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("acc", fontsize=14)
plt.plot(data["epoch"],data["acc"],label="acc")
plt.plot(data["epoch"],data["val_acc"], label="val_acc")

#  Устанавливаем интервал основных делений:
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
#  Устанавливаем интервал вспомогательных делений:
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

#  Тоже самое проделываем с делениями на оси "y":
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.001))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.0005))

"""plt.plot(data["epoch"],data["acc"])
plt.plot(data["epoch"],data["iou"])"""
ax.legend(loc = 1)


fig, ax = plt.subplots(figsize=(15, 15), linewidth = 3)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("iou", fontsize=14)
plt.plot(data["epoch"],data["iou"],label="iou")
plt.plot(data["epoch"],data["val_iou"], label="val_iou")

#  Устанавливаем интервал основных делений:
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
#  Устанавливаем интервал вспомогательных делений:
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

#  Тоже самое проделываем с делениями на оси "y":
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

"""plt.plot(data["epoch"],data["acc"])
plt.plot(data["epoch"],data["iou"])"""
ax.legend(loc = 1)
plt.show()