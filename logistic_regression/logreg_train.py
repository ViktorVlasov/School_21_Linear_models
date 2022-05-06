import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt

EPS = 1e-15


def logistic(z):
	return 1. / (1 + np.exp(-z))


class LogReg:
	def __init__(self, lr=0.1, tolerance=0.00001, max_iters=10000, fit_intercept=True, batch_size=32, solver='SGD'):
		"""
		lr : float, default 0.001
			Скорость обучения.
		tolerance : float, default 0.0001
			Критерий остановки градиентного спуска. Если разница между значениями функции стоимость на
			двух последних итерациях меньше, чем данное значение, то градиентный спуск завершается.
		max_iters : int, default 10000
			Максимальное количество эпох.
		fit_intercept : bool, default True
			Добавлние фиктивного признака (bias)
		batch_size : int, default 32
			Используется только при solver == MBSG, размер мини-батча
		solver : str, default 'SGD'
			Метод градиентного спуска.
			SGD - стохастический градиентый спуск
			BGD - батчевый градиентый спуск
			MBGD - мини-батчевый градиентый спуск
		"""
		self.tolerance = tolerance
		self.lr = lr
		self.max_iters = max_iters
		self.errors = []
		self.theta = []
		self.fit_intercept = fit_intercept
		self.mean_val = None
		self.std_val = None
		self.batch_size = batch_size
		self.solver = solver

	@staticmethod
	def _add_intercept(X: np.ndarray):
		b = np.ones([X.shape[0], 1])
		return np.concatenate([b, X], axis=1)

	def scale_features(self, X: np.ndarray):
		Z = X.copy()
		self.mean_val = Z[:, 1:].mean(axis=0)
		self.std_val = Z[:, 1:].std(axis=0)

		Z[:, 1:] = (Z[:, 1:] - self.mean_val) / self.std_val
		return Z

	def compute_cost(self, X, y, theta):
		predicted = np.clip(logistic(X @ theta), EPS, 1 - EPS)
		cost = np.mean(-(y * np.log(predicted) + (1 - y) * np.log(1 - predicted)))

		return cost

	def compute_cost_grad(self, X, y, theta):
		predicted = logistic(X @ theta)
		grad = ((predicted - y).reshape(-1, 1) * X).mean(axis=0)
		return grad

	def fit(self, X, y=None):
		self.X = X
		self.y = y

		if self.fit_intercept:
			self.X = self._add_intercept(self.X)

		self.X = self.scale_features(self.X)

		# Initialize weights + bias term if fit_intercept = True
		self.theta = np.random.normal(size=(self.X.shape[1], ), scale=0.5)
		# self.theta = np.zeros((self.X.shape[1],))

		self._gradient_descent()

	def iterate_minibatches(self, batchsize, shuffle=False):
		X = self.X
		y = self.y

		if shuffle:
			indices = np.arange(X.shape[0])
			np.random.shuffle(indices)
		for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
			if shuffle:
				excerpt = indices[start_idx:start_idx + batchsize]
			else:
				excerpt = slice(start_idx, start_idx + batchsize)
			yield X[excerpt], y[excerpt]

	def _minibatch_gradient_descent(self, theta):
		errors = self.errors

		for batch in self.iterate_minibatches(self.batch_size, shuffle=True):
			x_batch, y_batch = batch
			delta = self.compute_cost_grad(x_batch, y_batch, theta)
			theta -= self.lr * delta

			errors.append(self.compute_cost(self.X, self.y, theta))

			error_diff = np.linalg.norm(errors[-2] - errors[-1])
			if error_diff < self.tolerance:
				break
		return theta

	def _batch_gradient_descent(self, theta):
		delta = self.compute_cost_grad(self.X, self.y, theta)
		theta -= self.lr * delta

		self.errors.append(self.compute_cost(self.X, self.y, theta))
		return theta

	def _stochastic_gradient_descent(self, theta):
		errors = self.errors

		for k in range(len(self.X)):
			index_rand = np.random.randint(0, len(self.X))
			predicted = logistic(self.X[index_rand, :] @ theta)
			grad = ((predicted - self.y[index_rand]) * self.X[index_rand, :])
			theta -= self.lr * grad

			errors.append(self.compute_cost(self.X, self.y, theta))

			error_diff = np.linalg.norm(errors[-2] - errors[-1])
			if error_diff < self.tolerance:
				break
		return theta

	def _gradient_descent(self):
		self.errors = [self.compute_cost(self.X, self.y, self.theta)]

		for epoch in range(1, self.max_iters + 1):
			if self.solver == 'SGD':
				self.theta = self._stochastic_gradient_descent(self.theta)
			if self.solver == 'MBGD':
				self.theta = self._minibatch_gradient_descent(self.theta)
			if self.solver == 'BGD':
				self.theta = self._batch_gradient_descent(self.theta)
			error_diff = np.linalg.norm(self.errors[-2] - self.errors[-1])
			if error_diff < self.tolerance:
				print(f"Convergence has reached. Iteration: {len(self.errors)}, error: {self.errors[-1]}")
				break


def preproc(df):
	X = df.copy()
	X = X.iloc[:, 5:]
	X = X.fillna(X.median())

	y = df['Hogwarts House']
	y_lst = []
	for i in y.unique():
		y_lst.append((y == i).astype(int).to_list())

	return X.to_numpy(), np.array(y_lst)


def train(X, y, solver):
	thetas = []
	scale = []

	clf = None
	for one_class_y in y:
		clf = LogReg(solver=solver)
		clf.fit(X, one_class_y)
		thetas.append(clf.theta)
		plt.plot(clf.errors)
	plt.title('График изменения функции стоимости от номера итерации')
	plt.legend(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
	plt.savefig('data/Cost_plot')
	scale.extend([clf.mean_val, clf.std_val])

	return thetas, scale


def main(path, solver):
	df_train = pd.read_csv(path, index_col=0)

	X, y = preproc(df_train)
	thetas, scale = train(X, y, solver)

	df_theta = pd.DataFrame({'theta0': thetas[0],
							 'theta1': thetas[1],
							 'theta2': thetas[2],
							 'theta3': thetas[3]
							 })

	df_scale = pd.DataFrame({'mean': scale[0],
							 'std': scale[1]})

	df_theta.to_csv('data/df_theta.csv', index=False)
	df_scale.to_csv('data/df_scale.csv', index=False)


if __name__ == '__main__':
	try:
		if len(sys.argv) != 2 and len(sys.argv) != 3:
			raise ValueError('Wrong amount args!')

		path = sys.argv[1]
		f = open(path)
		f.close()

		solver = 'BGD'
		if len(sys.argv) == 3:
			solver = sys.argv[2]
			if solver not in ('SGD', 'BGD', 'MBGD'):
				raise ValueError('Wrong value of second argument!')

		main(path, solver)
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print(f'File not found!')


