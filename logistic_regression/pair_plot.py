import pandas as pd
import sys
import seaborn as sns


def get_pair_plot(df_train):
	list_of_features = [df_train.columns[0]] + df_train.columns[5:].to_list()
	sns.pairplot(df_train[list_of_features], hue='Hogwarts House').figure.savefig("data/pair_plot")


if __name__ == '__main__':
	try:
		if len(sys.argv) != 2:
			raise ValueError('Wrong amount args!')
		path = sys.argv[1]
		f = open(path)
		f.close()
		df = pd.read_csv(path, index_col=0)
		get_pair_plot(df)
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print(f'File "{path}" not found!')

