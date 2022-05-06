import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from predict import Predict


def distrib(df):
	plt.title('Исходные данные')
	plt.scatter(df.km, df.price, c='r', marker='x')
	plt.show()


def history(df_history):
	plt.title('График изменения функции стоимости от номера итерации')
	history = df_history.losses
	plt.plot(range(len(history)), history)
	plt.show()


def trained(df, df_params, predict=False):
	pr = Predict()
	pr.params_from_files(df_params)

	if predict:
		x = pr.enter_value()
		plt.plot(x, pr.prediction(x), c='g', marker='o')
	plt.title('Обученая модель')
	plt.scatter(df.km, df.price, c='r', marker='x')
	plt.plot(df.km.values, pr.prediction(df.km.values))
	plt.show()


if __name__ == "__main__":
	try:
		df = pd.read_csv("data/data.csv")
	except:
		print("file with dataframe is not valid")
		exit()

	try:
		df_params = pd.read_csv("params.csv")
		df_history = pd.read_csv("loss_history.csv")
	except:
		print("Use train.py before use visualize.py")
		exit()

	print("What to visualize?\n 1. Distribution of data\n 2. History of losses\n 3. Trained model\n 4. Predict value")
	x = int(input("Enter the number:"))
	while x < 1 or x > 4:
		x = int(input("Enter the correct number:"))

	if x == 1:
		distrib(df)
	elif x == 2:
		history(df_history)
	elif x == 3:
		trained(df, df_params)
	elif x == 4:
		trained(df, df_params, predict=True)
