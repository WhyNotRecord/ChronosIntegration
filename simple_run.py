import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import argparse


# todo https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='1. Time series forecasting with AutoGluon Chronos')
  parser.add_argument('input_file', type=str, help='Path to the file with data in CSV format')
  parser.add_argument('output_file', type=str, help='Path to the file for saving predictions in CSV format')
  parser.add_argument('-lookback', type=int, default=0, required=False,
                      help='Length of the lookback period for prediction')
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
  prediction_length = 3

  train_data = data.tail(500)
  models_dir = "./ag_models/one-shot"  # Или любой другой путь
  os.makedirs(models_dir, exist_ok=True)  # Создаем каталог, если он не существует
  model_name = os.path.basename(args.input_file).split('_')[0]
  model_path = os.path.join(models_dir, model_name)  # Уникальное имя для каждой модели

  print(f"Calling fit on sequence with length {train_data.shape[0]}")
  predictor = TimeSeriesPredictor(target='close', prediction_length=prediction_length, freq='D', path=model_path,
                                  eval_metric='MSE', quantile_levels=[0.25, 0.5, 0.75]).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_base", "ag_args": {"name_suffix": "ZeroShot"}},
        ],
        "AutoETS": {},
    },
    # features=[],
    num_val_windows=50, val_step_size=2,
    time_limit=600,  # time limit in seconds (for tuning)
    enable_ensemble=True,
  )

  print("Starting inference")

  predict_data = train_data if args.lookback == 0 else train_data.tail(args.lookback)
  print(f"Calling predict on sequence with length {predict_data.shape[0]}")
  # predictions = predictor.predict(predict_data, model="ChronosFineTuned[bolt_small]")
  predictions = predictor.predict(predict_data)
  # Удаляем item_id из индекса, если он там есть
  if 'item_id' in predictions.index.names:
    predictions = predictions.reset_index(level='item_id')  # Переносим item_id в колонки
  predictions.drop(columns=['item_id'], inplace=True)  # Удаляем столбец item_id

  # Преобразование индекса в Unix-время в миллисекундах (long)
  predictions.index = predictions.index.astype("int64") // 10 ** 6

  # Сохранение предсказаний в CSV
  try:
      # Создаем каталоги по пути выходного файла, если они отсутствуют
      dirname = os.path.dirname(args.output_file)
      if dirname:
        os.makedirs(dirname, exist_ok=True)
      predictions.to_csv(args.output_file)
      print(f"Predictions exported to file: {args.output_file}")
  except Exception as e:
      print(f"Error exporting predictions to file {args.output_file}: {e}")
      exit(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
