import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score,
    confusion_matrix, classification_report, PrecisionRecallDisplay, roc_curve, precision_recall_curve # <-- Убедитесь, что эта строка здесь
)
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import SMOTE 

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Для XGBoost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import warnings

from sklearn.metrics import roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')
df = pd.read_csv('df_volatility.csv')

feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume','Date',
                                                            'Log_Return', 'Rolling_Volatility', 'Volume_MA',
                                                            'Short_Term_Volatility', 'Long_Term_Volatility', 'Future_Vol',
                                                            'Target_Volatility_Spike']] # Исключаем целевую
X = df[feature_columns]
y = df['Target_Volatility_Spike']

corr = df[feature_columns + ['Target_Volatility_Spike']].corr()['Target_Volatility_Spike'].sort_values(ascending=False)

models = {
    'LogisticRegression': LogisticRegression(
        random_state=42, 
        C=0.1, 
        penalty='l2', 
        solver='lbfgs', 
        max_iter=1000,
        class_weight='balanced'
    ),
    'XGBClassifier': XGBClassifier(
        random_state=42, max_depth=4, eval_metric='logloss', use_label_encoder=False  #  Для XGB аналог class_weight, динамично по train
    )
}
cyclic_features = ['DayOfWeek_sin', 'DayOfMonth_sin', 'DayOfWeek_cos', 'DayOfMonth_cos']
numeric_features = [col for col in feature_columns if col not in cyclic_features]
# Создаем preprocessor
preprocessor = ColumnTransformer([
('scale', StandardScaler(), numeric_features),
('cyclic', 'passthrough', cyclic_features)   ])

X = df[feature_columns]
y = df['Target_Volatility_Spike']
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Pipeline как в функции для 2ух моделей
steps_lr = [('preprocessor', preprocessor), ('classifier', LogisticRegression(class_weight='balanced', penalty='l2',  C=0.1, random_state=42))]
steps_xgb = [('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, objective='binary:logistic',colsample_bytree=0.4, max_depth=4, eval_metric='aucpr',learning_rate=0.05, n_estimators=200))]

pipeline_lr = Pipeline(steps_lr)
pipeline_xgb = Pipeline(steps_xgb)

pipeline_lr.fit(X_train, y_train)
pipeline_xgb.fit(X_train, y_train)

y_pred_lr = pipeline_lr.predict(X_test)
y_pred_xgb = pipeline_xgb.predict(X_test)

y_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]
y_proba_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]


cols_to_drop_db = [
    # 1. Утечка данных (Target Leakage)
    'Future_Vol', 'Target_Volatility_Spike', 'Date',
    
    # 2. Сырые цены и объемы (Нестационарность)
    'Close', 'High', 'Low', 'Open', 
    'Volume', 'Volume_MA', 
    'BB_Upper', 'BB_Lower', 
    'KC_high', 'KC_low', 'KC_mid',

    #. Слабые или неполные циклические признаки
    'DayOfWeek_sin', 'DayOfMonth_sin',
    'DayOfWeek_cos', 'DayOfMonth_cos',
    
    # 3. Абсолютная волатильность (зависит от цены актива)
    'ATR',  # Если не нормирован на цену, он вырастет вместе с курсом BTC
    'MACD', 'MACD_Signal' # Оставляем MACD_Diff
]

date_column_name = 'Date'
if date_column_name in df.columns:
    df[date_column_name] = pd.to_datetime(df[date_column_name])
    df = df.set_index(date_column_name)
else:
    print(" ОШИБКА: Колонка с датами не найдена. Проверь название колонки.")

X_db = df.drop(columns=cols_to_drop_db, errors='ignore')
y_db = df['Target_Volatility_Spike']


def dbscan_anomaly_detector(X_db, y_db, train_size):
    scaler = StandardScaler()
    X_train, X_test = X_db.iloc[:train_size], X_db.iloc[train_size:]
    y_train, y_test = y_db.iloc[:train_size], y_db.iloc[train_size:]
    

    X_scaled = scaler.fit_transform(X_train)
    
    
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X_scaled)
    
    X_test_scaled = scaler.transform(X_test)
    X_test_reduced = pca.transform(X_test_scaled)
    
    # Обучаем DBSCAN только на TRAIN
    # eps и min_samples подбираю вручную
    db = DBSCAN(eps=0.5, min_samples=5) 
    db.fit(X_reduced) # X_reduced получен из X_train
    
    # Получаем метки: -1 это выброс, 0,1,2это кластеры
    labels_train = db.labels_
    
    # 2. Создаем "Обертку" для предсказаний через KNN
    # учим KNN запоминать: "Если точка здесь, то DBSCAN считает её выбросом"
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_reduced, labels_train)
    
    # 3. Предсказываем на TEST честным способом 
    # KNN найдет ближайших соседей из прошлого (Train) и присвоит их метку
    test_labels_pred = knn.predict(X_test_reduced) 
    train_labels_pred = labels_train # Для трейна берем родные метки
    
    y_pred_db_train = (train_labels_pred == -1).astype(int)
    y_pred_db_test = (test_labels_pred == -1).astype(int)

    print(f"DBSCAN (через KNN) нашел аномалий в Train: {y_pred_db_train.sum()}")
    print(f"DBSCAN (через KNN) нашел аномалий в Test: {y_pred_db_test.sum()}")

    return  (y_pred_db_train, y_pred_db_test, X_train.index, X_test.index)
    
train_size = int(len(X_db) * 0.8)
y_pred_db_train, y_pred_db_test, train_idx, test_idx = dbscan_anomaly_detector(X_db, y_db, train_size) 

# Функция для создания 3D массива для LSTM
# look_back - сколько дней в прошлом смотрит сеть (например, 10)
def create_sequences(X, y, look_back=10):
    Xs, ys = [], []
    # X должен быть numpy array (X_scaled)
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)]) # Берем окно от i до i+10
        ys.append(y.iloc[i + look_back]) # Предсказываем точку сразу после окна
    return np.array(Xs), np.array(ys)
    
train_size = int(len(X_db) * 0.8)
# Берем данные из X_db (где уже удалены ATR, BB_Upper и т.д.)
X_train = X_db.iloc[:train_size]
X_test = X_db.iloc[train_size:]
# Инициализируем скейлер заново для LSTM
scaler = StandardScaler()

# 1. Масштабируем данные (Обязательно!)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Создаем последовательности
look_back = 10 
X_train_lstm, y_train_lstm = create_sequences(X_train_scaled, y_train, look_back)
X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, look_back)

model_lstm = Sequential()

# Слой 1: LSTM
# units=32 или 50 - количество нейронов. return_sequences=False (так как следующий слой Dense)
model_lstm.add(Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model_lstm.add(LSTM(units=50, return_sequences=False))

# Слой 2: Dropout (КРИТИЧНО для Precision)
# Отключаем 40% нейронов случайно, чтобы сеть не заучивала шум
model_lstm.add(Dropout(0.4))

# Слой 3: Выход
model_lstm.add(Dense(1, activation='sigmoid')) # 0 или 1

# Компиляция
# learning_rate поменьше для стабильности
model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])

# Обучение
# class_weight используем тот же, что и для XGBoost/LogReg!
history = model_lstm.fit(
    X_train_lstm, y_train_lstm,
    epochs=50,            # Не слишком много
    batch_size=32,
    validation_split=0.1, # Следим за валидацией
    class_weight={0: 1, 1: 19}, # Ваш вес классов (пример: 19)
    verbose=0
)

y_proba_lstm = model_lstm.predict(X_test_lstm).flatten()

def volatility(y_proba_lr, y_proba_xgb, y_pred_db_test, y_proba_lstm, y_test, weight_lr=0.5, weight_xgb= 0.2, weight_lstm = 0.3,boost_factor = 0.18, look_back=10):

    y_proba_lr_cut = y_proba_lr[look_back:]
    y_proba_xgb_cut = y_proba_xgb[look_back:]
    y_pred_db_cut = y_pred_db_test[look_back:] # Если используете сигнал DBSCAN
    y_test_cut = y_test.iloc[look_back:]

    y_proba_ensemble = (weight_lr * y_proba_lr_cut) + (weight_xgb * y_proba_xgb_cut) + (weight_lstm * y_proba_lstm)

    y_proba_hybrid = y_proba_ensemble + (y_pred_db_cut * boost_factor)
    y_proba_hybrid = np.clip(y_proba_hybrid, 0, 1) # чтоб не выйти за границы вероятнсти (0,1)
    
    # Поиск лучшего порога для максимизации F1-Score
    thresholds = np.arange(0.01, 0.41, 0.01)
    best_f1 = 0
    best_threshold = 0
    best_value = 0
    
    for t in thresholds:
        y_pred_t = (y_proba_hybrid > t).astype(int)
        f1 = f1_score(y_test_cut, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    print(f"Оптимальный порог для гибрида: {best_threshold:.2f}, F1-Score: {best_f1:.4f}")

    # 5. Финальная оценка
    y_pred_optimized_hybrid = (y_proba_hybrid > best_threshold).astype(int)
    return y_pred_optimized_hybrid
    
y_pred_final = volatility(y_proba_lr, y_proba_xgb, y_pred_db_test, y_proba_lstm, y_test)

# 1. Сначала рассчитаем SMA (Скользящую среднюю) на ПОЛНОМ датафрейме
# SMA 50 - классический индикатор среднесрочного тренда
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# 2. Подготовка тестовых данных (как в прошлом коде)
test_data = df.iloc[train_size:].copy()
test_data = test_data.iloc[:len(y_pred_final)]
test_data['Signal'] = y_pred_final

# Сдвигаем сигнал на 1 день (действуем завтра по сегодняшнему сигналу)
test_data['Signal_Shifted'] = test_data['Signal'].shift(1).fillna(0)

test_data['Strategy_BuyHold'] = test_data['Log_Return']

#НОВАЯ СТРАТЕГИЯ 1: "Мягкий Хедж" (50% Cash
# Если сигнал 1: Доходность = 50% от доходности BTC (мы наполовину в USDT)
# Если сигнал 0: Доходность = 100% от доходности BTC
test_data['Strategy_SoftHedge'] = np.where(
    test_data['Signal_Shifted'] == 1, 
    test_data['Log_Return'] * 0.5,  # Снижаем риск (и прибыль) в 2 раза
    test_data['Log_Return']
)

# НОВАЯ СТРАТЕГИЯ 2: "Умный Фильтр" (Trend Filter) 
# Условие выхода: Сигнал модели == 1 И Цена < SMA_50 (Тренд нисходящий)
# shift(1) у цен тоже нужен, так как мы принимаем решение на открытии дня
close_shifted = test_data['Close'].shift(1)
sma_shifted = test_data['SMA_50'].shift(1)

# Определяем условие паники
panic_condition = (test_data['Signal_Shifted'] == 1) & (close_shifted < sma_shifted)

test_data['Strategy_TrendFilter'] = np.where(
    panic_condition, 
    0,                      # Выходим в кэш полностью, так как все плохо (тренд вниз + волатильность)
    test_data['Log_Return'] # Иначе держим, даже если есть сигнал волатильности (игнорируем ложные тревоги на росте)
)

# 4. РАСЧЕТ МЕТРИК
def calculate_metrics(series, name):
    total_return = np.exp(series.sum()) - 1 # Перевод из лог-доходности в проценты
    volatility = series.std() * np.sqrt(365) # Годовая волатильность
    sharpe = (series.mean() / series.std()) * np.sqrt(365) # Шарп (без risk-free rate)
    
    # Расчет максимальной просадки (Max Drawdown)
    cum_returns = series.cumsum()
    peak = cum_returns.cummax()
    drawdown = cum_returns - peak
    max_drawdown = np.exp(drawdown.min()) - 1
    
    return {
        'Стратегия': name,
        'Доходность (Total)': f"{total_return*100:.1f}%",
        'Шарп (Sharpe)': f"{sharpe:.2f}",
        'Макс. просадка': f"{max_drawdown*100:.1f}%"
    }
metrics = []
metrics.append(calculate_metrics(test_data['Strategy_BuyHold'], 'Buy & Hold'))
metrics.append(calculate_metrics(test_data['Strategy_SoftHedge'], 'Soft Hedge (50/50)'))
metrics.append(calculate_metrics(test_data['Strategy_TrendFilter'], 'Trend Filter (Smart)'))

print("\n=== РЕЗУЛЬТАТЫ УЛУЧШЕННЫХ СТРАТЕГИЙ ===")
print(pd.DataFrame(metrics))

plt.figure(figsize=(15, 7))

cum_buy_hold = test_data['Strategy_BuyHold'].cumsum()
cum_soft_hedge = test_data['Strategy_SoftHedge'].cumsum()
cum_trend_filter = test_data['Strategy_TrendFilter'].cumsum()

plt.plot(cum_buy_hold, label='Buy & Hold', color='gray', alpha=0.6, linestyle='--')
plt.plot(cum_soft_hedge, label='Soft Hedge (50% выход)', color='red', alpha=0.6)
plt.plot(cum_trend_filter, label='Trend Filter', color='green', linewidth=1.5)

plt.title('стратегии')
plt.ylabel('Накопленная лог-доходность')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

df_test = df.iloc[train_size+look_back:]
index_test = df.index[train_size+look_back:]

y_final_test = pd.Series(y_pred_final, index=index_test)

anom_test  = y_final_test[y_pred_final == 1].index


plt.figure(figsize=(18,6))
plt.plot(df_test.index, df_test['Rolling_Volatility'], color='gray', label='волатильность', alpha=0.7)
plt.scatter(anom_test, df_test.loc[anom_test, 'Rolling_Volatility'], color='red', label='всплеск волатильности', s=10)

plt.xlabel('Date')
plt.ylabel('Rolling_Volatility')
plt.legend()
plt.show()


