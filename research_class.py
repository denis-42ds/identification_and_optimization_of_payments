# импорт модулей
import phik
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import timedelta
from catboost import CatBoostClassifier
from phik.report import plot_correlation_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class DatasetExplorer:
	def __init__(self, dataset, target=None):
		self.dataset = dataset
		self.target = target

	def explore_dataset(self):
		# Вывод информации о датасете
		self.dataset.info()

		# Вывод случайных примеров из датасета
		display(self.dataset.sample(5))

		# Количество полных дубликатов строк
		print(f"количество полных дубликатов строк: {self.dataset.duplicated().sum()}")

		# Круговая диаграмма для количества полных дубликатов
		if self.dataset.duplicated().sum() > 0:
			sizes = [self.dataset.duplicated().sum(), self.dataset.shape[0]]
			fig1, ax1 = plt.subplots()
			ax1.pie(sizes, labels=['duplicate', 'not a duplicate'], autopct='%1.0f%%')
			plt.title('Количество полных дубликатов в общем количестве строк', size=12)
			plt.show()

		print(f"""количество пропущенных значений:\n{self.dataset.isnull().sum()}""")
		if self.dataset.isnull().values.any():
			sns.heatmap(self.dataset.isnull(), cmap=sns.color_palette(['#000099', '#ffff00']))
			plt.xticks(rotation=90)
			plt.title('Визуализация количества пропущенных значений', size=12, y=1.02)
			plt.show()

		# Вывод признаков с пропущенными значениями
		missing_values_ratios = {}
		for column in self.dataset.columns[self.dataset.isna().any()].tolist():
			missing_values_ratio = self.dataset[column].isna().sum() / self.dataset.shape[0]
			missing_values_ratios[column] = missing_values_ratio

		print("Процент пропущенных значений в признаках:")
		for column, ratio in missing_values_ratios.items():
		    print(f"{column}: {ratio*100:.2f}%")

		# Выбираем только признаки, у которых в названии есть 'id'
		id_columns = [col for col in self.dataset.columns if 'id' in col]

		# Выводим информацию для каждого выбранного признака
		for col in id_columns:
			print(f"Количество уникальных значений в столбце '{col}': {self.dataset[col].nunique()}")
			print(f"Соотношение уникальных значений и общего количества записей в столбце '{col}': {self.dataset.shape[0] / self.dataset[col].nunique():.2f}")
			print()

		if self.target:
			print(f"""Соотношение классов целевой переменной:
			{self.dataset[self.target].value_counts().sort_index(ascending=False)}""")

			labels = 'False', 'True'
			sizes = [self.dataset[self.target].value_counts()[0],
					 self.dataset[self.target].value_counts()[1]]
			fig1, ax1 = plt.subplots()
			ax1.pie(sizes, labels=labels, autopct='%1.0f%%')
			plt.title('Соотношение классов целевой переменной', size=12)
			plt.show()
