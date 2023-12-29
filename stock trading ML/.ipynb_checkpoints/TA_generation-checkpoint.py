import talib

def calculate_SMA(df, column_name, timeperiod):
    df[column_name] = talib.SMA(df['Close'], timeperiod=timeperiod)

def calculate_RSI(df, column_name, timeperiod):
    df[column_name] = talib.RSI(df['Close'], timeperiod=timeperiod)

def calculate_MACD(df, macd_column, signal_column, fastperiod, slowperiod, signalperiod):
    macd, signal, _ = talib.MACD(df['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    df[macd_column] = macd
    df[signal_column] = signal

def calculate_BollingerBands(df, upper_column, middle_column, lower_column, timeperiod):
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=timeperiod)
    df[upper_column] = upper
    df[middle_column] = middle
    df[lower_column] = lower

def calculate_MOM(df, column_name, timeperiod):
    df[column_name] = talib.MOM(df['Close'], timeperiod=timeperiod)

def calculate_STOCH(df, slowk_column, slowd_column):
    slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'])
    df[slowk_column] = slowk
    df[slowd_column] = slowd

def calculate_ADX(df, column_name, timeperiod):
    df[column_name] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=timeperiod)

def add_prev_close(df):
    df['prev_close'] = df['Close'].shift(1)

def calculate_RC(df, column_name, periods):
    df[column_name] = df['Close'].transform(lambda x: x.pct_change(periods=periods))

def get_TA(df):
    calculate_SMA(df, 'SMA_10', 10)
    calculate_SMA(df, 'SMA_50', 50)
    
    calculate_RSI(df, 'RSI_14', 14)
    
    calculate_MACD(df, 'MACD', 'MACD_Signal', 12, 26, 9)
    
    calculate_BollingerBands(df, 'BB_Upper', 'BB_Middle', 'BB_Lower', 20)
    
    calculate_MOM(df, 'MOM_10', 10)
    
    calculate_STOCH(df, 'SlowK', 'SlowD')
    
    calculate_ADX(df, 'ADX_14', 14)
    
    add_prev_close(df)
    
    calculate_RC(df, 'RC', 15)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df
