"""
Модуль для работы с комплексными сигналами
==========================================

Содержит функции для:
- Расчёта комплексных непрерывных и дискретных сигналов
- Расчёта RMS и мгновенной мощности
- Визуализации комплексных сигналов (модуль и фаза)

By xImDoctor, 2025
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 1. ФУНКЦИИ РАСЧЁТА КОМПЛЕКСНЫХ СИГНАЛОВ
# ==============================================================================

def complex_continuous_signal(A, f, phi, t):
    """
    Вычисляет непрерывный комплексный сигнал.

    Формула: z(t) = A·exp(i(2πft + φ))

    Параметры:
    ----------
    A : float
        Амплитуда сигнала
    f : float
        Частота (Гц)
    phi : float
        Начальная фаза (радианы)
    t : array_like
        Время (массив или скаляр, секунды)

    Возвращает:
    ----------
    array_like (complex)
        Комплексные значения сигнала в моменты времени t

    Пример:
    -------
    >>> t = np.linspace(0, 1, 1000)
    >>> z = complex_continuous_signal(A=1, f=5, phi=np.pi/4, t=t)
    >>> modulus = np.abs(z)
    >>> phase = np.angle(z)
    """
    return A * np.exp(1j * (2 * np.pi * f * t + phi))


def complex_discrete_signal(A, f, phi, f_g, k):
    """
    Вычисляет дискретный комплексный сигнал.

    Формула: z[k] = A·exp(i((2πf/f_g)k + φ))

    Параметры:
    ----------
    A : float
        Амплитуда сигнала
    f : float
        Частота сигнала (Гц)
    phi : float
        Начальная фаза (радианы)
    f_g : float
        Частота дискретизации (Гц)
    k : array_like
        Номера отсчётов (массив или скаляр)

    Возвращает:
    ----------
    array_like (complex)
        Комплексные значения сигнала в отсчётах k

    Пример:
    -------
    >>> k = np.arange(0, 21)
    >>> z = complex_discrete_signal(A=1, f=2, phi=0, f_g=20, k=k)
    """
    return A * np.exp(1j * ((2 * np.pi * f / f_g) * k + phi))


# ==============================================================================
# 2. ФУНКЦИИ ДЛЯ РАСЧЁТА МОЩНОСТИ (RMS) КОМПЛЕКСНЫХ СИГНАЛОВ
# ==============================================================================

def calculate_rms_complex(z):
    """
    Вычисляет RMS (среднеквадратичное значение) комплексного сигнала.

    Формула: RMS = √(1/N · Σ|z_i|²)
    Теория: RMS = A для комплексной экспоненты с амплитудой A

    Параметры:
    ----------
    z : array_like (complex)
        Массив значений комплексного сигнала

    Возвращает:
    ----------
    float
        Среднеквадратичное значение (RMS)

    Пример:
    -------
    >>> z = np.exp(1j*2*np.pi*5*np.linspace(0, 1, 1000))
    >>> rms = calculate_rms_complex(z)
    >>> print(f"RMS = {rms:.6f}, Теория = 1.000000")
    """
    N = len(z)
    return np.sqrt(np.sum(np.abs(z)**2) / N)


def calculate_instantaneous_power_complex(z):
    """
    Вычисляет мгновенную мощность комплексного сигнала.

    Формула: P(t) = |z(t)|²

    Параметры:
    ----------
    z : array_like (complex)
        Массив значений комплексного сигнала

    Возвращает:
    ----------
    array_like
        Мгновенная мощность в каждый момент времени
    """
    return np.abs(z)**2


# ==============================================================================
# 3. ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ КОМПЛЕКСНЫХ СИГНАЛОВ
# ==============================================================================

def plot_multiple_complex_signals(signals, f_g=20, k_max=20, t_max=None,
                                   show_sum=False, mode='discrete'):
    """
    Строит несколько комплексных сигналов на одном графике (модуль и фаза).

    Параметры:
    ----------
    signals : list of tuples
        Список кортежей [(A1, f1, phi1, label1), (A2, f2, phi2, label2), ...]
    f_g : float, optional
        Частота дискретизации (Гц), по умолчанию 20
    k_max : int, optional
        Максимальный номер отсчёта, по умолчанию 20
    t_max : float, optional
        Максимальное время на оси
    show_sum : bool, optional
        Показывать сумму сигналов
    mode : str, optional
        'discrete', 'continuous' или 'both'

    Пример:
    -------
    >>> signals = [
    ...     (1, 2, 0, 'Сигнал 1'),
    ...     (1.5, 3, np.pi/3, 'Сигнал 2')
    ... ]
    >>> plot_multiple_complex_signals(signals, mode='both', show_sum=True)
    """
    if mode in ['discrete', 'both']:
        k = np.arange(0, k_max + 1)
        t_discrete = k / f_g

    if t_max is None:
        t_max = k_max / f_g if mode in ['discrete', 'both'] else 1

    t_continuous = np.linspace(0, t_max, 1000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    sum_signal_discrete = 0
    sum_signal_continuous = 0

    for i, (A, f, phi, label) in enumerate(signals):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Непрерывный
        if mode in ['continuous', 'both']:
            z_cont = complex_continuous_signal(A, f, phi, t_continuous)
            sum_signal_continuous += z_cont
            alpha = 0.3 if mode == 'both' else 1.0
            linewidth = 1 if mode == 'both' else 2
            ax1.plot(t_continuous, np.abs(z_cont), color=color,
                    linewidth=linewidth, alpha=alpha,
                    label=label if mode == 'continuous' else None)
            ax2.plot(t_continuous, np.angle(z_cont), color=color,
                    linewidth=linewidth, alpha=alpha,
                    label=label if mode == 'continuous' else None)

        # Дискретный
        if mode in ['discrete', 'both']:
            z_disc = complex_discrete_signal(A, f, phi, f_g, k)
            sum_signal_discrete += z_disc
            ax1.scatter(t_discrete, np.abs(z_disc), color=color, marker=marker,
                       s=100, label=label, zorder=5)
            ax2.scatter(t_discrete, np.angle(z_disc), color=color, marker=marker,
                       s=100, label=label, zorder=5)

    # Сумма
    if show_sum and len(signals) > 1:
        if mode == 'continuous':
            ax1.plot(t_continuous, np.abs(sum_signal_continuous), 'k--',
                    linewidth=2.5, label='Сумма')
            ax2.plot(t_continuous, np.angle(sum_signal_continuous), 'k--',
                    linewidth=2.5, label='Сумма')
        elif mode == 'both':
            ax1.plot(t_continuous, np.abs(sum_signal_continuous), 'k-',
                    linewidth=1, alpha=0.3)
            ax2.plot(t_continuous, np.angle(sum_signal_continuous), 'k-',
                    linewidth=1, alpha=0.3)
            ax1.scatter(t_discrete, np.abs(sum_signal_discrete), color='black',
                       marker='D', s=120, label='Сумма', zorder=6,
                       edgecolors='white', linewidths=1)
            ax2.scatter(t_discrete, np.angle(sum_signal_discrete), color='black',
                       marker='D', s=120, label='Сумма', zorder=6,
                       edgecolors='white', linewidths=1)
        else:
            ax1.scatter(t_discrete, np.abs(sum_signal_discrete), color='black',
                       marker='D', s=120, label='Сумма', zorder=6,
                       edgecolors='white', linewidths=1)
            ax2.scatter(t_discrete, np.angle(sum_signal_discrete), color='black',
                       marker='D', s=120, label='Сумма', zorder=6,
                       edgecolors='white', linewidths=1)

    # Настройки
    mode_title = {'discrete': 'Дискретные', 'continuous': 'Непрерывные',
                  'both': 'Непрерывные + дискретные'}[mode]

    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Время t (секунды)', fontsize=12)
    ax1.set_ylabel('|z(t)| - Модуль', fontsize=12)
    ax1.set_title(f'{mode_title} комплексные сигналы - Модуль',
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(-0.05, t_max + 0.05)
    ax1.legend(loc='upper right', fontsize=10)

    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Время t (секунды)', fontsize=12)
    ax2.set_ylabel('arg(z(t)) - Фаза (радианы)', fontsize=12)
    ax2.set_title(f'{mode_title} комплексные сигналы - Фаза',
                  fontsize=13, fontweight='bold')
    ax2.set_xlim(-0.05, t_max + 0.05)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_signal_with_rms_complex(A, f, phi, f_g=20, k_max=40, t_max=2):
    """
    Строит комплексный сигнал с отображением RMS и мгновенной мощности.

    Параметры:
    ----------
    A : float
        Амплитуда
    f : float
        Частота сигнала (Гц)
    phi : float
        Начальная фаза (радианы)
    f_g : float, optional
        Частота дискретизации (Гц)
    k_max : int, optional
        Максимальный номер отсчёта
    t_max : float, optional
        Максимальное время на оси

    Пример:
    -------
    >>> plot_signal_with_rms_complex(A=1, f=2, phi=0, f_g=20, k_max=40, t_max=2)
    """
    # Дискретный
    k = np.arange(0, k_max + 1)
    t_discrete = k / f_g
    z_discrete = complex_discrete_signal(A, f, phi, f_g, k)

    # Непрерывный
    t_continuous = np.linspace(0, t_max, 1000)
    z_continuous = complex_continuous_signal(A, f, phi, t_continuous)

    # RMS
    rms_discrete = calculate_rms_complex(z_discrete)
    rms_continuous = calculate_rms_complex(z_continuous)

    # Мгновенная мощность
    power_discrete = calculate_instantaneous_power_complex(z_discrete)
    power_continuous = calculate_instantaneous_power_complex(z_continuous)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    # График 1: Модуль
    ax1.plot(t_continuous, np.abs(z_continuous), 'b-', linewidth=1, alpha=0.4)
    ax1.scatter(t_discrete, np.abs(z_discrete), color='red', s=60, zorder=5)
    ax1.axhline(y=rms_continuous, color='green', linestyle='--', linewidth=2,
                label=f'RMS (мощность) = {rms_continuous:.3f}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Время t (секунды)', fontsize=12)
    ax1.set_ylabel('|z(t)| - Модуль', fontsize=12)
    ax1.set_title(f'Модуль сигнала: A={A}, f={f} Гц, φ={phi:.2f}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.05, t_max + 0.05)

    # График 2: Фаза
    ax2.plot(t_continuous, np.angle(z_continuous), 'b-', linewidth=1, alpha=0.4)
    ax2.scatter(t_discrete, np.angle(z_discrete), color='red', s=60, zorder=5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Время t (секунды)', fontsize=12)
    ax2.set_ylabel('arg(z(t)) - Фаза (рад)', fontsize=12)
    ax2.set_title('Фаза сигнала', fontsize=13, fontweight='bold')
    ax2.set_xlim(-0.05, t_max + 0.05)
    ax2.set_ylim(-np.pi, np.pi)

    # График 3: Мгновенная мощность
    ax3.plot(t_continuous, power_continuous, 'purple', linewidth=1, alpha=0.4)
    ax3.scatter(t_discrete, power_discrete, color='red', s=60, zorder=5)
    ax3.axhline(y=rms_continuous**2, color='green', linestyle='--', linewidth=2,
                label=f'Средняя мощность = {rms_continuous**2:.3f}')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Время t (секунды)', fontsize=12)
    ax3.set_ylabel('|z(t)|² - Мгновенная мощность', fontsize=12)
    ax3.set_title('Мгновенная мощность сигнала', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_xlim(-0.05, t_max + 0.05)

    plt.tight_layout()
    plt.show()

    # Вывод результатов
    print(f"Параметры сигнала: A={A}, f={f} Гц, φ={phi:.2f}")
    print(f"Мощность (RMS) дискретного сигнала: {rms_discrete:.6f}")
    print(f"Мощность (RMS) непрерывного сигнала: {rms_continuous:.6f}")
    print(f"Теоретическая мощность (равна A): {A:.6f}")


def compare_signal_powers_complex(signals, f_g=20, k_max=40):
    """
    Сравнивает RMS нескольких комплексных сигналов.

    Параметры:
    ----------
    signals : list of tuples
        Список кортежей [(A1, f1, phi1, label1), ...]
    f_g : float, optional
        Частота дискретизации (Гц)
    k_max : int, optional
        Максимальный номер отсчёта

    Пример:
    -------
    >>> signals = [
    ...     (1, 2, 0, 'A=1'),
    ...     (2, 3, np.pi/4, 'A=2')
    ... ]
    >>> compare_signal_powers_complex(signals)
    """
    k = np.arange(0, k_max + 1)
    results = []

    print(f"\nСравнение RMS комплексных сигналов:")
    print("-" * 80)
    print(f"{'Сигнал':<15} {'A':<8} {'f (Гц)':<10} {'φ (рад)':<12} {'RMS':<12}")
    print("-" * 80)

    for A, f, phi, label in signals:
        signal = complex_discrete_signal(A, f, phi, f_g, k)
        rms = calculate_rms_complex(signal)

        results.append((label, A, f, phi, rms))
        print(f"{label:<15} {A:<8.2f} {f:<10.2f} {phi:<12.4f} {rms:<12.6f}")

    print("-" * 80)

    # Визуализация
    labels = [r[0] for r in results]
    amplitudes = [r[1] for r in results]
    rms_values = [r[4] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, amplitudes, width, label='Амплитуда A',
                   color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, rms_values, width, label='RMS (измеренное)',
                   color='green', alpha=0.7)

    ax.set_xlabel('Сигналы', fontsize=12)
    ax.set_ylabel('Значение', fontsize=12)
    ax.set_title('Сравнение амплитуд и RMS (комплексные сигналы)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    return results
