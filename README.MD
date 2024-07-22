# Проект по LogisticRegression модели и SVM модели и их сравнению 

### Скачаем все необходимые библиотеки

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```
### Установим таблицу 

```
data = pd.read_csv('processed_train (5).csv')
data
```
![krasivo](https://i.postimg.cc/hvL9kfSg/Group-1-1.png)

### Разобьем выборку на тест и трейн

```
X = data.drop('HasDetections', axis=1)
y = data['HasDetections']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=33)
```
### Обучиv LogReg и SVM с помощью классов `LogisticRegression` и `LinearSVC`. Добавив шаг стандартизации данных в пайплайны. Для разнообразия, возьмем `MinMaxScaler`. Пайплайны запишем в переменные `pipe_lr` и `pipe_svm` соответственно!

```
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

### Your code is here

pipe_lr= Pipeline([('lr_scaler', MinMaxScaler()), 
                    ('lr_estimator', LogisticRegression())])

pipe_svm = Pipeline([('lr_scaler', MinMaxScaler()), 
                    ('svm_estimator', LinearSVC())])
```
### Обучаем модели на трейне

```
pipe_svm.fit(X_train , y_train)

pipe_lr.fit(X_train , y_train)
```


### Теперь построим ROC для обеих моделей, посчитаем AUC . Конечно же, обучение проводим на трейне, а замеры - на тесте.

```
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

fpr, tpr, thresholds = roc_curve(y_test, pipe_lr.predict_proba(X_test)[:, 1])
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
```

### lr модель

![krasivo](https://i.postimg.cc/yYrFw4KD/2024-07-23-015323.png)

### Но чтоб нам построить ROC для SVM модели нам потребуется формула 

```
decision_preds = pipe_svm.decision_function(X_test)

min_pred = min(decision_preds)
max_pred = max(decision_preds)

preds_svm = [-abs(x-min_pred)/min_pred*0.5
             if x <= 0 
             else abs(x/max_pred)*0.5+0.5 
             for x in decision_preds]
```

### Теперь строим Roc кривую длля модели SVM

```
fpr_2, tpr_2, thresholds = roc_curve(y_test, preds_svm)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
```


### SVM модель

![krasivo](https://i.postimg.cc/SxgNZkbP/2024-07-23-015705.png)

### Графики относительно одинаковые но когда мы будем их сравнивать на калибровочной кривой то там удже будет разница

### Посчитаем AUC

```
print("auc_svm = ", auc(fpr_2, tpr_2))
print("auc_lr = ", auc(fpr, tpr))
```

Вывод
```
auc_svm =  0.6877981855211535
auc_lr =  0.6890736795540984
```

### Теперь напишем формулу для lr модели  **Брайера**

```
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


### Your code is here
prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
b_score_lr = brier_score_loss(y_test, prob_lr)
print("Оценка Брайера:",b_score_lr)
true_lr, pred_lr = calibration_curve(y_test, prob_lr, n_bins=10)
```

вывод 
```
Оценка Брайера lr модели : 0.12629320173808245
```

### Выводим Калибровочную кривую lr модели

```
plt.plot(pred_lr,
true_lr,
marker='o',
linewidth=1,
label='Логистическая регрессия')
plt.plot([0, 1],
[0, 1],
linestyle='--',
label='Идеально откалибрована')
plt.title('Калибровочная кривая вероятностей для lr модели')
plt.xlabel('Предсказанная вероятность')
plt.ylabel('Истинная вероятность')
plt.legend(loc='best')
plt.show()
```
![krasivo](https://i.postimg.cc/YSgX82JN/2024-07-23-020849.png)

### Напишем формулу для SVM модели **Брайера**

```
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

### Your code is here
b_score_svm = brier_score_loss(y_test, preds_svm)
print("Оценка Брайера:", b_score_svm)
true_svm, pred_svm = calibration_curve(y_test, preds_svm, n_bins=10)
```

вывод 
```
Оценка Брайера SVM модели : 0.14883229692128133
```


### Выводим Калибровочную кривую SVM модели
```
plt.plot(pred_svm,
true_svm,
marker='o',
linewidth=1,
label='Логистическая регрессия')
plt.plot([0, 1],
[0, 1],
linestyle='--',
label='Идеально откалибрована')
plt.title('Калибровочная кривая вероятностей для SVM модели')
plt.xlabel('Предсказанная вероятность')
plt.ylabel('Истинная вероятность')
plt.legend(loc='best')
plt.show()
```
![krasivo](https://i.postimg.cc/mg6DVBQr/2024-07-23-021208.png)

### Теперь мы видим разницу

### И теперь главный вопрос **Какая калибровочная кривая ближе к диагонали?**

### LR модель ближе к диоганали


