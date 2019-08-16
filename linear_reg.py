#!usr/bin/python
#Linear regression model
#Code source: Irina Calic

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.formula.api as sm
import seaborn as sns

from pandas import DataFrame, Series
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.graphics.regressionplots import *
from statsmodels.iolib.summary2 import summary_col
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

sns.set(color_codes=True)

df1 = pd.read_csv('/path/path/filename.csv')

df1.dropna()
y=len(df1.columns)

for x in range(0, y-1):
	a=df1.columns[x]
	result1 = smf.ols(formula = 'fitness ~ '+a, data=df1).fit()
	print(result1.summary())
	result2 = smf.ols(formula = "fitness ~ "+a+" + np.power("+a+", 2)", data=df1).fit()
	print(result2.summary())

diag1=df1.min()
with pd.option_context('display.max_rows', 16000, 'display.max_columns', 3):
	print(diag1)

diag_a = df1[df1.apply(lambda x: np.abs(x-x.mean()) /x.std() <3).all(axis=1)]


with pd.option_context('display.max_rows', 400, 'display.max_columns', 16000):
	print(diag_a)
	
f = open('/path/path/mydat.txt', 'w')
z = str(result1.summary())
i = str(result2.summary())
b = str(diag1)
v = str(diag_a)
f.write(z)
f.write(i)
f.write(b)
f.write(v)
f.close()

