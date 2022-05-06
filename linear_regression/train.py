import numpy as np
import pandas as pd


class LinearRegression:
	def __init__(self, scale=True):
		self.scale = scale
		self.theta = [0, 0]
		self.history = []
		self.mean_value = 0
		self.std_value = 0

	def compute_hypothesis(self, x):
		return self.theta[0] + self.theta[1] * x

	def compute_cost(self, x, y):
		m = x.shape[0]
		return 1 / (2 * m) * np.sum((self.compute_hypothesis(x) - y) ** 2)

	def scale_features(self, x):
		self.mean_value = np.mean(x)
		self.std_value = np.std(x)
		return (x - self.mean_value) / self.std_value

	def fit(self, x, y, lr=0.01, tolerance=0.0001):
		if self.scale:
			x = self.scale_features(x)

		self.history.append(self.compute_cost(x, y))
		curr_epoch = 0
		while True:

			theta_temp = self.theta.copy()
			theta_temp[0] = theta_temp[0] - lr * np.mean(self.compute_hypothesis(x) - y)
			theta_temp[1] = theta_temp[1] - lr * np.mean((self.compute_hypothesis(x) - y) * x)
			self.theta = theta_temp

			self.history.append(self.compute_cost(x, y))

			curr_epoch += 1
			loss_diff = np.fabs(self.history[-2] - self.history[-1])
			if loss_diff <= tolerance:
				break

	def r_squared(self, x, y):
		if self.scale:
			x = ((x - self.mean_value) / self.std_value)

		rss = np.sum((y - self.compute_hypothesis(x)) ** 2)
		tss = np.sum((y - y.mean()) ** 2)
		return 1 - rss / tss

	def prediction(self, x):
		x = ((x - self.mean_value) / self.std_value)
		return self.theta[0] + self.theta[1] * x


if __name__ == "__main__":
	try:
		df = pd.read_csv("data/data.csv")
	except:
		print("file with dataframe is not valid")

	regr = LinearRegression()
	regr.fit(df.km.values, df.price.values)

	losses_history = pd.DataFrame(data={'losses': regr.history})
	losses_history.to_csv('loss_history.csv', index=False)

	df_export = pd.DataFrame(data={'w0': [regr.theta[0]],
									'w1': [regr.theta[1]],
									'mean_value': [regr.mean_value],
									'std_value': [regr.std_value]})
	df_export.to_csv('params.csv', index=False)

	print(regr.r_squared(df.km.values, df.price.values))