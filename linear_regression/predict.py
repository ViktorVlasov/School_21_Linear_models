import pandas as pd
import numpy as np

class Predict:
    def __init__(self):
        self.theta = [0, 0]
        self.mean_value = 0
        self.std_value = 1

    def enter_value(self):
        x = int(input('Enter a mileage: '))
        while x < 0 or x > 350000:
            if x < 0:
                print("Mileage should be nonnegative number.")
            elif x > 350000:
                print("You enter too big number.")
            x = int(input('Enter a correct mileage: '))
        return x

    def params_from_files(self, df):
        self.theta[0] = df.iloc[0, 0]
        self.theta[1] = df.iloc[0, 1]
        self.mean_value = df.iloc[0, 2]
        self.std_value = df.iloc[0, 3]

    def prediction(self, x):
        x = (x - self.mean_value) / self.std_value
        return self.theta[0] + self.theta[1] * x

if __name__ == '__main__':
    pr = Predict()
    x = pr.enter_value()
    try:
        df = pd.read_csv("params.csv")
        pr.params_from_files(df)
    except:
        print("Use train.py before use predict.py")

    print(f"Your predict is {round(pr.prediction(x), 2)}")