import pandas as pd
import numpy as np
import sys


def logistic(z):
	return 1. / (1 + np.exp(-z))


def get_X(df_test, df_scale):
	X = df_test.iloc[:, 5:]
	X = X.fillna(X.median())
	X = X.to_numpy()

	mean_val = df_scale['mean'].to_numpy()
	std_val = df_scale['std'].to_numpy()

	# добавляем bias
	b = np.ones([X.shape[0], 1])
	X = np.concatenate([b, X], axis=1)

	# scale
	X[:, 1:] = (X[:, 1:] - mean_val) / std_val
	return X


def main(df_test_path, df_theta_path, df_scale_path):
	df_test = pd.read_csv(df_test_path, index_col=0)
	df_theta = pd.read_csv(df_theta_path)
	df_scale = pd.read_csv(df_scale_path)

	X = get_X(df_test, df_scale)

	pred_lst = []
	for theta in df_theta.T.to_numpy():
		pred = logistic(X @ theta)
		pred_lst.append(pred)
	houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
	pred_houses = [houses[i] for i in np.argmax(pred_lst, axis=0)]

	df_houses = pd.DataFrame({'Index': list(range(len(pred_houses))),
				  'Hogwarts House': pred_houses})
	df_houses.to_csv('data/houses.csv', index=False)



if __name__ == '__main__':
	try:
		if len(sys.argv) != 4:
			raise ValueError('Wrong amount args!')
		for p in sys.argv[1:]:
			f = open(p)
			f.close()
		main(sys.argv[1], sys.argv[2], sys.argv[3])
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print(f'File "{p}" not found!')
