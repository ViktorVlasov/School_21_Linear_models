import pandas as pd
import sys
from matplotlib import pyplot as plt
import seaborn as sns


def get_scatter(df_train):
	plt.figure(figsize=(10, 10))
	sns.scatterplot(data=df_train, x='Astronomy',
					y='Defense Against the Dark Arts',
					hue='Hogwarts House').figure.savefig('data/scatter')
	plt.figure(figsize=(10, 10))
	sns.scatterplot(data=df_train, x='Flying',
					y='History of Magic',
					hue='Hogwarts House').figure.savefig('data/scatter1')

if __name__ == '__main__':
	try:
		if len(sys.argv) != 2:
			raise ValueError('Wrong amount args!')
		path = sys.argv[1]
		f = open(path)
		f.close()
		df = pd.read_csv(path, index_col=0)
		get_scatter(df)
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print(f'File "{path}" not found!')