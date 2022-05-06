import pandas as pd
import sys
from matplotlib import pyplot as plt


def get_hist(df_train):
	houses = df_train['Hogwarts House'].unique()
	courses = df_train.columns[5:]

	plt.figure(figsize=(15, 20))
	for idx, course in enumerate(courses):
		plt.subplot(4, 4, idx + 1)
		for house in houses:
			plt.hist(df_train[df_train['Hogwarts House'] == house][course], alpha=0.6, label=house)
			plt.title(course)
			plt.legend()
		plt.savefig('data/hist1')

	plt.figure(figsize=(12, 12))
	for house in houses:
		plt.hist(df_train[df_train['Hogwarts House'] == house]['Arithmancy'], alpha=0.6, label=house)
		plt.title('Arithmancy')
		plt.legend()
	plt.savefig('data/hist2')


if __name__ == '__main__':
	try:
		if len(sys.argv) != 2:
			raise ValueError('Wrong amount args!')
		path = sys.argv[1]
		f = open(path)
		f.close()
		df = pd.read_csv(path, index_col=0)
		get_hist(df)
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print(f'File "{path}" not found!')