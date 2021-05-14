import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from enum import IntEnum, unique
from sklearn.ensemble import GradientBoostingRegressor


@unique
class Phase(IntEnum):
    DATA_EXPLORATION = 1,
    PARAMETER_EXTRACTION = 2,
    MODEL_TRAINING = 3,
    PREDICTION = 4,
    MODEL_EVALUATION = 5


class Explorer:
    def __init__(self, filename):
        self._filename = filename

        pd.set_option('display.max.columns', None)
        self._dataset = pd.read_csv(self._filename, low_memory=False)
        print(self._dataset.corr().to_csv(filename.replace('.', '.corr.')))

        self._dataset_stats = self._dataset.describe()
        self._dataset_stats.to_csv(filename.replace('.', '.stats.'))

    def get_dataset(self):
        return self._dataset

    def get_dataset_stats(self):
        return self._dataset_stats

    def scatter_2d(self, x_axis, y_axis, index):
        self._dataset.plot(x=x_axis, y=y_axis, style='o')
        plt.title(f'{x_axis} vs {y_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.savefig(f'plots/data_exploration/{index:02d}_{x_axis}Vs{y_axis}.png', bbox_inches='tight')
        plt.show()

    def hist(self, feature, index):
        sns.histplot(data=self._dataset, x=feature, kde=True)
        plt.title(f'{feature} Distribution')
        plt.savefig(f'plots/data_exploration/{index:02d}_{feature}Distribution.png', bbox_inches='tight')
        plt.show()

    def density_dist(self, feature, index):
        sns.kdeplot(data=self._dataset, x=feature)
        plt.title(f'{feature} Density Distribution')
        plt.savefig(f'plots/data_exploration/{index:02d}_{feature}Distribution.png', bbox_inches='tight')
        plt.show()

    def explore_data(self, feature_name, label_name):
        self.hist(label_name, 1)
        self.density_dist(label_name, 2)
        self.scatter_2d(feature_name, label_name, 3)

        return self._dataset, self._dataset_stats


class ParamExtractor:
    def __init__(self, feature_name, label_name, test_size, random_state):
        self._feature_name = feature_name
        self._label_name = label_name
        self._test_size = test_size
        self._random_state = random_state

    def extract(self, dataset):
        # define the attribute and the label
        # it was decided that the attribute is the min temp and the label is the maxTemp
        x = dataset[[self._feature_name]].values  # .reshape(-1,1)
        y = dataset[self._label_name].values  # .reshape(-1,1)

        # polynomial_features = PolynomialFeatures(degree=2)
        # x = polynomial_features.fit_transform(x)

        return train_test_split(x, y, test_size=self._test_size, random_state=self._random_state)


class Trainer:
    def __init__(self):
        self._models = {
            'linear_regression': LinearRegression(),
            'gradient_boosting': GradientBoostingRegressor(learning_rate=0.3, n_estimators=102)
        }

    def train(self, x_train, y_train, algorithm):
        # create a trainer using linear regression algorithm
        print(self._models['gradient_boosting'])
        model = self._models['linear_regression']
        model.fit(x_train, y_train)

        return model


class ModelEvaluator:
    def __init__(self, inputs, actual_values, predicted_values):
        self._inputs = inputs
        self._actual_values = actual_values
        self._predicted_values = predicted_values

    def evaluate(self):
        df = pd.DataFrame({'Actual': self._actual_values.flatten(), 'Predicted': self._predicted_values.flatten()})

        # show first row_count predictions vs actual
        row_count = 25
        print(df.head(row_count))

        # graph the above comparison
        df1 = df.head(row_count)
        df1.plot(kind='bar')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.savefig(f'plots/model_evaluation/01_actualVsPrediction.png', bbox_inches='tight')
        plt.show()

        print(f'Feature: {self._inputs.shape}, actual: {self._actual_values.shape}, predicted:{self._predicted_values.shape}')
        plt.scatter(self._inputs, self._actual_values, color='gray')
        print(self._inputs)
        plt.plot(self._inputs, self._predicted_values, color='red', linewidth=2)
        plt.savefig(f'plots/model_evaluation/02_regressionLineOnScatterPlot.png', bbox_inches='tight')
        plt.show()

        print('Mean Absolute Error:', metrics.mean_absolute_error(self._actual_values, self._predicted_values))
        print('Mean Squared Error:', metrics.mean_squared_error(self._actual_values, self._predicted_values))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self._actual_values, self._predicted_values)))
