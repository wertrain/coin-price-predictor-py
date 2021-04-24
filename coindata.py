from datetime import datetime, timedelta
import json
import pandas as pd

def create_data_frame(date, symbol):
    ''' 指定された時刻の指定されたシンボルの価格をデータフレームで返す
    '''
    all_symbol_data = read_symbols_prices(date)
    return pd.DataFrame({
        'symbol': symbol,
        'time': all_symbol_data[symbol].times,
        'price': all_symbol_data[symbol].prices,
    })

def create_data_frame_all(date):
    ''' 指定された時刻の指定されたシンボルの価格をデータフレームで返す
    '''
    symbols = times = prices = []
    for k, v in read_symbols_prices(date).items():
        symbols.append(k)
        times.append(v.times)
        prices.append(v.prices)

    return pd.DataFrame({
        'symbol': symbols,
        'time': times,
        'price': prices,
    })

def create_data_frame_all_span(start_date, end_date):
    ''' 指定された日付の範囲のすべてのシンボルの価格をデータフレームで返す
    '''
    merged_symbol_data = {}
    i_date = start_date
    while i_date <= end_date:
        for k, v in read_symbols_prices(i_date).items():
            symbol_data = None
            if (k in merged_symbol_data):
                symbol_data = merged_symbol_data[k]
            else:
                # 必要なデータを格納するオブジェクトを作成
                symbol_data = type("SymbolData", (object,), {
                    'name': k,
                    'times': [],
                    'prices': []
                })
                merged_symbol_data[k] = symbol_data
            symbol_data.times.append(v.times)
            symbol_data.prices.append(v.prices)
        i_date = i_date + timedelta(days=1)

    symbols = times = prices = []
    for k, v in merged_symbol_data.items():
        symbols.append(k)
        times.append(v.times)
        prices.append(v.prices)
    return pd.DataFrame({
        'symbol': symbols,
        'time': times,
        'price': prices,
    })

def get_prices_file_name(date, hour):
    ''' ファイル名を作成する
    '''
    base_name = date.strftime('%Y-%m-%d')
    return './data/' + base_name + '/' + base_name + '_' + format(hour, '02d') + '.txt'

def read_symbols_prices(date):
    ''' 指定された日付の価格データを辞書にして返す
    '''
    all_symbol_data = {}
    # 毎時間ごとに 1 ファイル存在する前提で処理する
    for i in range(24):
        # 日付 ＋ 時間からファイル名を決定
        with open(get_prices_file_name(date, i), 'r') as fh:

            line_count = 0 # ファイル内の行数（そのまま "分" を表す）
            for line in fh:
                # その時間における分の価格が json フォーマットで記載されている
                json_data = json.loads(line)
                # json データのパース
                for data in json_data:
                    symbol_data = None
                    # シンボル名をそのままキーとして扱う
                    # キーが存在すればオブジェクトを取り出し
                    if (data['symbol'] in all_symbol_data):
                        symbol_data = all_symbol_data[data['symbol']]
                    else:
                        # 必要なデータを格納するオブジェクトを作成
                        symbol_data = type("SymbolData", (object,), {
                            'name': data['symbol'],
                            'times': [],
                            'prices': []
                        })
                        # あらかじめ辞書にセット
                        all_symbol_data[data['symbol']] = symbol_data
                    # シンボル名/時間/価格をオブジェクトに設定
                    symbol_data.name = data['symbol']
                    symbol_data.times.append(date.strftime('%Y-%m-%d') + ' ' + format(i, '02d') + ':' + format(line_count, '02d') + ':00')
                    symbol_data.prices.append(data['price'])
                # 行数カウント
                line_count = line_count + 1
    # 作成した辞書を返す
    return all_symbol_data

if __name__ == "__main__":
    ''' テスト
    '''
    tdatetime = datetime(year=2021, month=4, day=10)
    spd = create_data_frame_all_span(datetime(year=2021, month=4, day=10), datetime(year=2021, month=4, day=11))
    print (spd)
    spd.to_csv('alldata.csv')
    #print (create_data_frame_all(tdatetime))
    #print (read_symbols_prices(tdatetime))
    # CSV 出力
    #symbol_name = 'BNBUSDT'
    #df = create_data_frame(tdatetime, symbol_name)
    #df.to_csv(tdatetime.strftime('%Y-%m-%d') + '_' + symbol_name + '.csv')
