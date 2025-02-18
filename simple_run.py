# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import argparse


# todo https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='1. Time series forecasting with AutoGluon Chronos')
  parser.add_argument('input_file', type=str, help='Path to the file with data in CSV format')
  parser.add_argument('output_file', type=str, help='Path to the file for saving predictions in CSV format')
  args = parser.parse_args()
  # Загрузка данных
  try:
    df = pd.read_csv(args.input_file, skipinitialspace=True)
  except FileNotFoundError:
    print(f"Error: File not found: {args.input_file}")
    exit(1)
  except pd.errors.ParserError:
    print(f"Error: Incorrect CSV file format: {args.input_file}")
    exit(1)

  # Преобразуем столбец 'timestamp' в datetime
  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
  # Создаем копию столбца 'timestamp'
  df['ds'] = df['timestamp']
  # Устанавливаем 'ds' в качестве индекса
  df = df.set_index('ds')
  # Удаляем столбцы 'close_timestamp', 'quoted_volume'
  df.drop(columns=['close_timestamp', 'quoted_volume'], inplace=True)
  # Добавляем столбец item_id со значением 0 для всех строк
  df['item_id'] = 0

  # Создаем TimeSeriesDataFrame
  data = TimeSeriesDataFrame(df)

  print(data.tail())
  prediction_length = 5

  # train_data, test_data = data.train_test_split(prediction_length)
  train_data = data
  models_dir = "./ag_models"  # Или любой другой путь
  os.makedirs(models_dir, exist_ok=True)  # Создаем каталог, если он не существует
  model_name = os.path.basename(args.input_file).split('_')[0]
  model_path = os.path.join(models_dir, model_name)  # Уникальное имя для каждой модели

  predictor = TimeSeriesPredictor(target='close', prediction_length=prediction_length, path=model_path, freq='D').fit(
      train_data=train_data,
      hyperparameters={
          "Chronos": [
              {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
              # {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
          ]
      },
      # features=['open', 'volume'],
      # hyperparameters={"Chronos": {"fine_tune": True, "fine_tune_lr": 1e-4, "fine_tune_steps": 2000}}
      time_limit=60,  # time limit in seconds (for fine-tuning?)
      enable_ensemble=False,
  )

  print("Starting inference")

  # predictions = predictor.predict(train_data, model="ChronosFineTuned[bolt_small]")
  predictions = predictor.predict(train_data)

  # Сохранение предсказаний в CSV
  try:
      predictions.to_csv(args.output_file)
      print(f"Predictions exported to file: {args.output_file}")
  except Exception as e:
      print(f"Error exporting predictions to file: {e}")
      exit(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
