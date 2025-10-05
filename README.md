# Signal Modeling

Библиотека `Python` для математического моделирования и анализа действительных и комплексных сигналов.

> Данный репозиторий, по сути, создан для быстрого импорта в `Google Colab`

## Навигация
1. [Описание](#описание)
2. [Необходимые пакеты](#необходимые-пакеты)
3. [Быстрый импорт в Colab](#быстрый-импорт-в-google-colab)
4. [Функции и возможности](#функции-и-возможности)
5. [Примеры использования](#примеры-использования)
6. [Лицензия и автор](#лицензия-и-автор)


## Описание

Проект содержит инструменты для работы с действительными и комплексными сигналами:
1. Задание функций сигналов
2. Построение графиков сигналов (множественных + суммы сигналов)
3. Вычисление мощности сигнала
4. Построение и сравнение мощностей (RMS)
   
Подробнее: [Функции и возможности](#функции-и-возможности), [Примеры использования](#примеры-использования).


## Необходимые пакеты

Для работы требуются `numpy`, `mathplotlib`:

```bash
pip install numpy matplotlib
```

## Быстрый импорт в Google Colab

Для использования в Google Colab выполните следующие команды:

```python
# Загрузка файлов с GitHub
!wget https://raw.githubusercontent.com/xImDoctor/signal-modeling/main/signals/real_signals.py
!wget https://raw.githubusercontent.com/xImDoctor/signal-modeling/main/signals/complex_signals.py

# Импорт модулей
from real_signals import *
from complex_signals import *
```

Или импортировать действительные и комплексные сигналы сразу одним файлом (*поддерживается на данный момент*):
```python
!wget https://raw.githubusercontent.com/xImDoctor/signal-modeling/main/signal_modeling_both.py

from signal_modeling_both import *
```


Другой способ (клонирование всего репозитория):

```python
!git clone https://github.com/xImDoctor/signal-modeling.git
import sys
sys.path.append('/content/signal-modeling/signals')

from real_signals import *
from complex_signals import *
```


## Функции и возможности

### `signals/real_signals.py` - Действительные сигналы

#### Функции расчёта сигналов:
- **`real_continuous_signal(A, f, phi, t)`** - Расчёт непрерывного действительного сигнала
  - Формула: `x(t) = A·sin(2πft + φ)`

- **`real_discrete_signal(A, f, phi, f_g, k)`** - Расчёт дискретного действительного сигнала
  - Формула: `x[k] = A·sin((2πf/f_g)k + φ)`

#### Функции расчёта мощности:
- **`calculate_rms_real(x)`** - RMS (среднеквадратичное значение) действительного сигнала
  - Теория: `RMS = A/√2` для синусоиды

- **`calculate_instantaneous_power_real(x)`** - Мгновенная мощность (`P(t) = x²(t)`)

#### Функции визуализации:
- **`plot_multiple_real_signals(signals, duration, samples, show_sum)`** - График нескольких непрерывных сигналов
- **`plot_multiple_real_discrete_signals(signals, f_g, k_max, show_sum)`** - График нескольких дискретных сигналов
- **`plot_signal_with_rms_real(A, f, phi, f_g, k_max, t_max)`** - Сигнал с RMS и мгновенной мощностью
- **`compare_signal_powers_real(signals, f_g, k_max)`** - Сравнение мощностей нескольких сигналов

---

### `signals/complex_signals.py` - Комплексные сигналы

#### Функции расчёта сигналов:
- **`complex_continuous_signal(A, f, phi, t)`** - Расчёт непрерывного комплексного сигнала
  - Формула: `z(t) = A·exp(i(2πft + φ))`

- **`complex_discrete_signal(A, f, phi, f_g, k)`** - Расчёт дискретного комплексного сигнала
  - Формула: `z[k] = A·exp(i((2πf/f_g)k + φ))`

#### Функции расчёта мощности:
- **`calculate_rms_complex(z)`** - RMS комплексного сигнала
  - Теория: `RMS = A` для комплексной экспоненты

- **`calculate_instantaneous_power_complex(z)`** - Мгновенная мощность (`P(t) = |z(t)|²`)

#### Функции визуализации:
- **`plot_multiple_complex_signals(signals, f_g, k_max, show_sum, mode)`** - График нескольких комплексных сигналов (модуль и фаза)
  - Режимы: `'discrete'`, `'continuous'`, `'both'`

- **`plot_signal_with_rms_complex(A, f, phi, f_g, k_max, t_max)`** - Комплексный сигнал с RMS и мгновенной мощностью
- **`compare_signal_powers_complex(signals, f_g, k_max)`** - Сравнение мощностей нескольких комплексных сигналов

---

## Примеры использования

Смотрите или запустите `main.py` в корне репозитория.

## Лицензия и автор

`MIT License`<br>
Project by xImDoctor, 2025
