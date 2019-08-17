# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
# Домашнее задание
# датасет будем присылать

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
# 1) настроить алгоритм, чтобы суммарная ошибка была меньше 0.780593173587177
# 2) добавить свой алгоритм из scikit-learn
# 3) pdf/html -> в опросник

# все должно работать
# все пункты должны быть выполнены

# оформление
# комментарии
# выводы

# дедлайн: 12:00 МСК 14.08.2019


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
money = pd.read_csv("money.csv")

# merge dataframes: money[new_col] = ...


#%%
#money.head(5) # head -> выводит указанное число строк


#%%
# Определение интервалов
# Глубина в прошлое ( дней * недель)
past = 7 * 4
# Глубина в будущее (дней * недель)
future = 7 * 1


#%%
# len -> length -> длина
# Проверка, что датасет подгрузился в нужном объеме
print("В датасете {} строк".format(len(money)))


#%%
df = list()
values = money["value"]

for i in range(past, len(money) - future):
    part_of_values = values[(i-past):(i+future)]
    df.append(list(part_of_values))


#%%
past_columns = [f"past_{i + 1}" for i in range(past)]
future_columns = [f"future_{i + 1}" for i in range(future)]


#%%
#Заполнение датафрейма данными
df = pd.DataFrame(df, columns=(past_columns + future_columns))


#%%
#df.head(5)


#%%
X = df[past_columns][0:-1] # прошлое + строки за исключением последней
Y = df[future_columns][0:-1] # будущее + строки за исключением последней


#%%
X_test = df[past_columns][-1:]
Y_test = df[future_columns][-1:]


#%%
# Словарь для хранения всех результатов
results = {}


#%%
# Функция добавления к общим результатам, если ошибка меньше 2
def addToResult(name, prediction, norm_error=0):
    if norm_error < 2:
        results[name] = [prediction, norm_error]


#%%
# Выбор лучшего результата
def chooseBestResult():
    min = ['default', 1000]
    for method in results:
        if results[method][1] < min[1]:
            min[0] = method
            min[1] = results[method][1]
    print(f'Лучший результат: {min[0]} с ошибкой {min[1]}')


#%%
# Отображение общих результатов
def showOverallResults():
    print('Графики предполагаемых значений с ошибкой менее 2')
    plt.figure(figsize = [12, 9])
    for method in results:
        lbl = method +':'+ str(round(results[method][1], 4))
        plt.plot(results[method][0], label=lbl) # label <- легенда
    plt.plot(df[future_columns].iloc[-1], label="Факт") 
    plt.legend()


#%%
def showResults(name, df, prediction, Y_test):
    print('Предположение:')
    print(prediction)
    print('Фактические данные:')
    print(Y_test)
    print('Нормализованная ошибка:')
    norm_error = np.linalg.norm(prediction - Y_test)
    print(norm_error)
    plt.plot(prediction, label="Предположение") # label <- легенда
    plt.plot(df[future_columns].iloc[-1], label="Факт") # iloc <- вытаскивает элемент на указанной позиции
    plt.legend() # <- отрисовывает окошко с легендами
    addToResult(name, prediction, norm_error)


#%%
# ------------------
# Линейная регрессия
# ------------------
from sklearn.linear_model import LinearRegression
# Обучение
reg = LinearRegression().fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults('Линейная регрессия',df, prediction, Y_test)


#%%
# ------------------
# Метод ближаших соседей
# ------------------
from sklearn.neighbors import KNeighborsRegressor
# Параметры
n_of_neighbors = 5 # число соседей
# Обучение
reg = KNeighborsRegressor(n_neighbors=n_of_neighbors).fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults(f'Ближайшие соседи {n_of_neighbors}шт.', df, prediction, Y_test)


#%%
# ------------------
# Метод Лассо LARS
# ------------------
from sklearn import linear_model
# Параметры
alpha_param = 0.1 # Альфа-параметр
# Обучение
reg = linear_model.LassoLars(alpha=alpha_param).fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults('LassoLARS', df, prediction, Y_test)


#%%
# ------------------
# Метод Лассо
# ------------------
from sklearn import linear_model
# Параметры
alpha_param = 0.005 # Альфа-параметр
# Обучение
reg = linear_model.Lasso(alpha=alpha_param, normalize=False, random_state= 20).fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults('Lasso', df, prediction, Y_test)


#%%
# ------------------
# Нейронная сеть / Параметры по-умолчанию
# ------------------
from sklearn import neural_network
# Параметры

# Обучение
reg = neural_network.MLPRegressor()
reg = reg.fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults('НС-default', df, prediction, Y_test)


#%%
# ------------------
# Нейронная сеть / Гипотеза 1 - увеличение hls, итераций, алгоритм 
# ------------------
from sklearn import neural_network
# Параметры
hls = 500 # Hidden layer sizes
slvr = 'lbfgs' # Алгоритм -> For small datasets, however, ‘lbfgs’ can converge faster and perform better.
iter = 5000 # Кол-во итераций
# Обучение
reg = neural_network.MLPRegressor(hidden_layer_sizes=hls, solver = slvr, max_iter = iter)
reg = reg.fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults('НС - №1',df, prediction, Y_test)


#%%
# ------------------
# Нейронная сеть / Гипотеза 2 - увеличение hls и итераций по сравнению с Гипотезой 1
# ------------------
from sklearn import neural_network
# Параметры
hls = 750 # Hidden layer sizes
slvr = 'lbfgs' # Алгоритм
iter = 10000 # Кол-во итераций
# Обучение
reg = neural_network.MLPRegressor(hidden_layer_sizes=hls, solver = slvr, max_iter = iter)
reg = reg.fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults('НС - №2', df, prediction, Y_test)


#%%
# ------------------
# Нейронная сеть / Гипотеза 2.2 - изменения параметров
# ------------------
from sklearn import neural_network
# Параметры
hls = 750 # Hidden layer sizes
slvr = 'lbfgs' # Алгоритм
iter = 10000 # Кол-во итераций
# Обучение
reg = neural_network.MLPRegressor(hidden_layer_sizes=hls, solver = slvr, max_iter = iter)
reg = reg.fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults('НС - №2.2', df, prediction, Y_test)


#%%
# ------------------
# Нейронная сеть / Гипотеза 3 - Гипотеза 2, смена алгоритма на adam, изменение параметров
# ------------------
from sklearn import neural_network
# Параметры
hls = 750 # Hidden layer sizes
slvr = 'adam' # Алгоритм
iter = 10000 # Кол-во итераций
# Обучение
reg = neural_network.MLPRegressor(hidden_layer_sizes=hls, solver = slvr, max_iter = iter)
reg = reg.fit(X, Y)
# Предположение
prediction = reg.predict(X_test)[0]
# Вывод результатов
showResults('НС - №3', df, prediction, Y_test)


#%%
# ------------------
# Нейронная сеть / Гипотеза 4 - Гипотеза 2, смена алгоритма на sgd
# ------------------
from sklearn import neural_network
# Параметры
hls = 750 # Hidden layer sizes
slvr = 'sgd' # Алгоритм
iter = 10000 # Кол-во итераций
# Обучение
reg = neural_network.MLPRegressor(hidden_layer_sizes=hls, solver = slvr, max_iter = iter)
reg = reg.fit(X, Y)
# Предположение
#prediction = reg.predict(X_test)[0]
#Вывод результатов
showResults('НС - №4',df, prediction, Y_test)


#%%
# -----------------------------
# ------ УВЕЛИЧЕНИЕ ИНТЕРВАЛА ОБУЧЕНИЯ В 3 РАЗА
# Определение интервалов
# Глубина в прошлое ( дней * недель)
past2 = 7 * 12
# Глубина в будущее (дней * недель)
future = 7 * 1


#%%
df2 = list()
values = money["value"]

for i in range(past2, len(money) - future):
    part_of_values = values[(i-past2):(i+future)]
    df2.append(list(part_of_values))


#%%
past_columns2 = [f"past_{i + 1}" for i in range(past2)]
future_columns = [f"future_{i + 1}" for i in range(future)]


#%%
#Заполнение датафрейма данными
df2 = pd.DataFrame(df2, columns=(past_columns2 + future_columns))


#%%
#df2.head(5)


#%%
X2 = df2[past_columns2][0:-1] # прошлое + строки за исключением последней
Y = df2[future_columns][0:-1] # будущее + строки за исключением последней


#%%
X2_test = df2[past_columns2][-1:]
Y_test = df2[future_columns][-1:]


#%%
# ------------------
# Нейронная сеть / Увеличение числа данных, изменение алгоритма
# ------------------
from sklearn import neural_network
# Параметры
hls = 10000 # Hidden layer sizes
slvr = 'adam' # Алгоритм
iter = 100000 # Кол-во итераций
lri = 0.0001 # learning rate init
# Обучение
reg = neural_network.MLPRegressor(hidden_layer_sizes=hls, solver = slvr, max_iter = iter, learning_rate_init= lri)
reg = reg.fit(X2, Y)
# Предположение
prediction = reg.predict(X2_test)[0]
# Вывод результатов
showResults('НС-увел.данных ADAM',df, prediction, Y_test)


#%%
# ------------------
# Нейронная сеть / Гипотеза 2.3 - Увеличение числа исходных данных 
# ------------------
from sklearn import neural_network
# Параметры
hls = 750 # Hidden layer sizes
slvr = 'lbfgs' # Алгоритм
iter = 10000 # Кол-во итераций
# Обучение
reg = neural_network.MLPRegressor(hidden_layer_sizes=hls, solver = slvr, max_iter = iter)
reg = reg.fit(X2, Y)
# Предположение
prediction = reg.predict(X2_test)[0]
# Вывод результатов
showResults('НС-увел.данных LBFGS',df, prediction, Y_test)


#%%
showOverallResults()
chooseBestResult()


#%%



