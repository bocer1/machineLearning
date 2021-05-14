#! /Users/tizzdale27/opt/anaconda3/bin/python
import sys
from tp_machine_learning import (Explorer, ParamExtractor, Trainer, ModelEvaluator, Phase)


class MLRunner:
    def __init__(self, filename='data/Weather.csv'):
        self._filename = filename
        self._label_name = 'MaxTemp'
        self._phase = Phase.MODEL_EVALUATION

    # main program
    def run(self):
        try:
            # read data and perform exploration
            e = Explorer(self._filename)
            feature_name = 'MeanTemp'
            dataset, _ = e.explore_data(feature_name, self._label_name)

            # decide on the features and label to use for training
            if self._phase >= Phase.PARAMETER_EXTRACTION:
                pe = ParamExtractor(feature_name, self._label_name, test_size=0.2, random_state=0)
                training_input, inputs, training_output, actual_outputs = pe.extract(dataset)

            # split the data into testing and training sets then train the model (using linear regression)
            if self._phase >= Phase.MODEL_TRAINING:
                t = Trainer()
                model = t.train(training_input, training_output, 'linear_regression')

            # Make predictions, feed test attributes to retrieve the predicted labels
            if self._phase >= Phase.PREDICTION:
                predicted_outputs = model.predict(inputs)
                print(f'Inputs shape: {inputs.shape}')
                print(f'Prediction shape: {predicted_outputs.shape}')

            # Evaluate model by viewing, comparing and analyzing predicted vs actual
            if self._phase >= Phase.MODEL_EVALUATION:
                me = ModelEvaluator(inputs, actual_outputs, predicted_outputs)
                me.evaluate()
        except FileNotFoundError as e:
            print(f'Input file \'{self._filename}\' is not available.\nDetails: {e!r}')


if __name__ == "__main__":
    file_name = 'data/Weather.csv'
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    r = MLRunner(file_name)
    r.run()
