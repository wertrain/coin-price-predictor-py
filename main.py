from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
import pickle
import gluonts
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution.student_t import StudentTOutput
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import coindata

def create_prices_data(symbols, start_date, end_date):
    ''' 与えられたシンボルの価格を作成
    '''
    cache_file_name = start_date.strftime('%Y-%m-%d') + '_' + end_date.strftime('%Y-%m-%d') + '_' + '_'.join(symbols) + '.csv'
    if os.path.exists(cache_file_name):
        return pd.read_csv(cache_file_name)
    else:
        dfs = []
        for symbol in symbols:
            dfs.append(coindata.create_data_frame_span(start_date, end_date, symbol))
        df = pd.concat(dfs)
        df.to_csv(cache_file_name)
    return df

def create_dataset(df, prediction_length):
    ''' 学習用のデータセットを作成
    '''
    # 価格データを整形する
    stock_dataset = df.pivot(index='symbol', columns='time', values='price')
    # データの開始時間を銘柄数分作成
    dates = [pd.Timestamp(START_DATE.strftime('%Y-%m-%d'), freq='1min') for _ in range(stock_dataset.shape[0])]
    # 学習は総データ数の半分を除いたデータ
    train_target_values = [ts[:-prediction_length] for ts in stock_dataset.values]
    # テストは全て含まれたデータ
    test_target_values = stock_dataset.copy().values

    # 学習 ListDataset を作成
    train_ds = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start,
            FieldName.ITEM_ID: code,
        }
        for (target, start, code) in zip(train_target_values, dates, stock_dataset.index)
    ], freq='1min')

    # テスト ListDataset を作成
    test_ds = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start,
            FieldName.ITEM_ID: code,
        }
        for (target, start, code) in zip(test_target_values, dates, stock_dataset.index)
    ], freq='1min')

    return train_ds, test_ds

def training(train_ds, prediction_length):
    ''' 学習させる
    '''
    cache_file_name = 'predictor.bin'
    if os.path.exists(cache_file_name):
        with open(cache_file_name, 'rb') as f:
            return pickle.load(f)
    else:
        # 学習の開始
        estimator = DeepAREstimator(
            freq='1min', # 必須
            prediction_length=prediction_length, # 必須
            trainer = Trainer(batch_size=32,
                            clip_gradient=10.0,
                            ctx='cpu',
                            epochs=100,
                            hybridize=True,
                            init="xavier",
                            learning_rate=0.001,
                            learning_rate_decay_factor=0.5,
                            minimum_learning_rate=5e-05,
                            num_batches_per_epoch=50,
                            patience=10,
                            weight_decay=1e-08),
            context_length = prediction_length,
            num_layers = 2,
            num_cells = 40,
            cell_type = 'lstm',
            dropout_rate = 0.1,
            use_feat_dynamic_real = False,
            use_feat_static_cat = False,
            use_feat_static_real = False,
            cardinality = None,
            embedding_dimension = None,
            distr_output = StudentTOutput(),
            scaling = True,
            lags_seq = None,
            time_features = None,
            num_parallel_samples = 100,
        )
        predictor = estimator.train(train_ds)
        with open(cache_file_name,'wb') as f:
            pickle.dump(predictor, f)
        return predictor

def evaluating(test_ds, predictor, num_samples):
    '''評価を実行する
    '''
    # 学習結果から推論を実行する
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=num_samples
    )
    # 時系列条件付け値の取得
    tss = list(tqdm(ts_it, total=len(test_ds)))
    # 時系列予測の取得
    forecasts = list(tqdm(forecast_it, total=len(test_ds)))
    # 評価を実行する
    evaluator = gluonts.evaluation.Evaluator(quantiles=[0.5])
    return tss, forecasts, evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))

def plot_prob_forecasts(ts_entry, forecast_entry, path, prediction_length, inline=True):
    # プロットの長さ
    plot_length = 60 * 5
    # サンプリングの 50% が含まれる区間、サンプリングの 90% が含まれる区間
    prediction_intervals = (50, 90)
    legend = ["実価格", "予測価格中央値"] + [f"{k}% 予測区間" for k in prediction_intervals][::-1]
    plt.rcParams['font.family'] = 'BIZ UDGothic'
    _, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    ax.axvline(ts_entry.index[-prediction_length], color='r')
    plt.legend(legend, loc="upper left")
    plt.title(forecast_entry.item_id)
    if inline:
        plt.show()
        plt.clf()
    else:
        # 出力先 フォルダを作成
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('{}forecast_{}.png'.format(path, forecast_entry.item_id))
        plt.close()

# 対象の銘柄名
SYMBOLS = ['BNBUSDT', "DOGEUSDT"]
# 学習データの開始期間・終了期間
START_DATE = datetime(year=2021, month=4, day=10)
END_DATE = datetime(year=2021, month=4, day=11)
# 予測期間（分）
PREDICTION_LENGTH = 120

# 価格データを作成
df = create_prices_data(SYMBOLS, START_DATE, END_DATE)
# 学習用データ、テスト用データを作成
train_ds, test_ds = create_dataset(df, PREDICTION_LENGTH)
# 学習を開始
predictor = training(train_ds, PREDICTION_LENGTH)
# 評価を実行
tss, forecasts, (agg_metrics, item_metrics) = evaluating(test_ds, predictor, 100)

# 結果をダンプしてみる
#print(json.dumps(agg_metrics, indent=4))
# 先頭のデータを表示してみる
#print(item_metrics.head())

# 結果の可視化
for i in tqdm(range(len(SYMBOLS))):
    ts_entry = tss[i]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry, './plots/', PREDICTION_LENGTH, inline=True)
