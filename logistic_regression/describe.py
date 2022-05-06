import pandas as pd
import sys

def percentile_alpha(Column: pd.Series, alpha) -> float:
	percentile = 0

	Column = Column.dropna()
	Column = Column.sort_values(ignore_index=True)
	N = (len(Column))

	index = (N - 1) * alpha
	if int(index) == index:  # has no fractional part
		return Column[index]
	fraction = index - int(index)
	left = int(index)
	right = left + 1
	i, j = Column[left], Column[right]
	return i + (j - i) * fraction


def get_min(Column: pd.Series):
	minim = Column[0]
	for val in Column:
		if val < minim:
			minim = val
	return minim


def get_max(Column: pd.Series):
	maxim = Column[0]
	for val in Column:
		if val > maxim:
			maxim = val
	return maxim


def describe(X: pd.DataFrame) -> pd.DataFrame:
	quantitative = [f for f in X.columns if X.dtypes[f] != 'object']

	X_len = len(X)

	count_lst = []
	mean_lst = []
	std_lst = []
	min_lst = []
	percentile_25 = []
	percentile_50 = []
	percentile_75 = []
	max_lst = []
	interquartile_range = []
	range = []

	for idx, col in enumerate(quantitative):
		# count
		Column = X[col].dropna()
		count_lst.append(float(len(Column)))

		# mean
		sum_values = sum(Column)
		mean_lst.append(sum_values / count_lst[idx])

		# std
		sum_square = 0
		mean_val = mean_lst[idx]
		for val in Column:
			sum_square += (val - mean_val) ** 2
		std_lst.append((sum_square / (count_lst[idx] - 1)) ** (1 / 2))

		# percentile
		percentile_25.append(percentile_alpha(X[col], 0.25))
		percentile_50.append(percentile_alpha(X[col], 0.5))
		percentile_75.append(percentile_alpha(X[col], 0.75))

		# min
		min_lst.append(get_min(X[col]))
		max_lst.append(get_max(X[col]))

		# Interquartile range
		interquartile_range.append(percentile_75[idx] - percentile_25[idx])

		# range
		range.append(max_lst[idx] - min_lst[idx])

	df_ = pd.DataFrame(columns=quantitative)
	df_.loc['count'] = count_lst
	df_.loc['mean'] = mean_lst
	df_.loc['std'] = std_lst
	df_.loc['min'] = min_lst
	df_.loc['25%'] = percentile_25
	df_.loc['50%'] = percentile_50
	df_.loc['75%'] = percentile_75
	df_.loc['max'] = max_lst
	df_.loc['intq_range'] = interquartile_range
	df_.loc['range'] = range

	return df_

if __name__ == '__main__':
	try:
		if len(sys.argv) != 2:
			raise ValueError('Wrong amount args!')
		path = sys.argv[1]
		f = open(path)
		f.close()
		df = pd.read_csv(path, index_col=0)
		print(describe(df))
	except ValueError:
		print("Требуется 1 аргумент - путь к файлу")
	except FileNotFoundError:
		print(f'Файл "{path}" не существует!')



