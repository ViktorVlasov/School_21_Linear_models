import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import sys

if len(sys.argv) == 3:
	y_true = pd.read_csv(sys.argv[1])['Hogwarts House']
	y_pred = pd.read_csv(sys.argv[2])['Hogwarts House']
	print(accuracy_score(y_true, y_pred))
else:
	print("Wrong amount args!")