from signals/complex_signals import *
from signals/real_signals import *


if __name__ == "__main__":
    print("="*80)
    print("Библиотека функций для обработки сигналов")
    print("="*80)
    
    # Пример 1: Действительные непрерывные сигналы
    print("\n1. Несколько действительных непрерывных сигналов:")
    signals_real_cont = [
        (1, 2, 0, 'f=2 Гц'),
        (0.7, 5, np.pi/4, 'f=5 Гц')
    ]
    plot_multiple_real_signals(signals_real_cont, duration=1, show_sum=True)
    
    # Пример 2: Действительные дискретные сигналы
    print("\n2. Несколько действительных дискретных сигналов:")
    signals_real_disc = [
        (1, 2, 0, 'Сигнал 1'),
        (2, 4, np.pi/4, 'Сигнал 2')
    ]
    plot_multiple_real_discrete_signals(signals_real_disc, f_g=20, k_max=20, show_sum=True)
    
    # Пример 3: Комплексные сигналы
    print("\n3. Комплексные сигналы (режим 'both'):")
    signals_complex = [
        (1, 2, 0, 'Сигнал 1'),
        (1.5, 3, np.pi/3, 'Сигнал 2')
    ]
    plot_multiple_complex_signals(signals_complex, f_g=20, k_max=40, 
                                   t_max=2, show_sum=True, mode='both')
    
    # Пример 4: RMS действительного сигнала
    print("\n4. RMS действительного сигнала:")
    plot_signal_with_rms_real(A=1, f=2, phi=0, f_g=20, k_max=40, t_max=2)
    
    # Пример 5: RMS комплексного сигнала
    print("\n5. RMS комплексного сигнала:")
    plot_signal_with_rms_complex(A=1, f=2, phi=0, f_g=20, k_max=40, t_max=2)
    
    # Пример 6: Сравнение мощностей
    print("\n6. Сравнение мощностей нескольких сигналов:")
    signals_compare = [
        (1, 2, 0, 'A=1'),
        (2, 3, np.pi/4, 'A=2'),
        (3, 1, np.pi/2, 'A=3')
    ]
    compare_signal_powers(signals_compare, signal_type='real')
    compare_signal_powers(signals_compare, signal_type='complex')