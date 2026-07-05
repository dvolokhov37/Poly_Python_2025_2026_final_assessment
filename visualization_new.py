import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from pathlib import Path

from oil_reservoir_new import OilReservoir, OilWell


def _save_or_show(save_path=None):
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pressure_2d(reservoir: OilReservoir, save_path=None):
    """2D карта пластового давления с улучшенной интерполяцией"""
    pressure = np.array(reservoir.pressure[len(reservoir.time_points) - 1])

    # Создаем более детальную сетку для интерполяции
    x = np.linspace(0, reservoir.field_size, pressure.shape[0])
    y = np.linspace(0, reservoir.field_size, pressure.shape[1])
    X, Y = np.meshgrid(x, y)

    # Создаем более плотную сетку для гладкой визуализации
    x_smooth = np.linspace(0, reservoir.field_size, 1000)
    y_smooth = np.linspace(0, reservoir.field_size, 1000)
    X_smooth, Y_smooth = np.meshgrid(x_smooth, y_smooth)

    # Интерполируем данные на более плотную сетку
    interp_spline = RectBivariateSpline(x, y, pressure)
    pressure_smooth = interp_spline(x_smooth, y_smooth)

    plt.figure(figsize=(12, 8))

    # Используем улучшенные параметры для imshow
    plt.imshow(pressure_smooth.T, origin='lower',
               extent=[0, reservoir.field_size, 0, reservoir.field_size],
               cmap='viridis', aspect='equal',
               interpolation='gaussian')

    # Отмечаем скважины
    for well in reservoir.wells:
        marker = 'v' if well.well_type == 'injection' else '^'
        color = 'red' if well.well_type == 'injection' else 'white'
        plt.plot(well.x, well.y, marker=marker, color=color,
                 markersize=10, label=f'Скважина {int(well.x)}:{int(well.y)}')

    plt.colorbar(label='Давление, Па')
    plt.xlabel('X, км')
    plt.ylabel('Y, км')
    plt.title('Карта распределения пластового давления')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_or_show(save_path)


def plot_pressure_3d_views(reservoir: OilReservoir, save_path=None):
    """3D карта пластового давления с разными углами обзора"""
    pressure = np.array(reservoir.pressure[len(reservoir.time_points) - 1])

    x = np.linspace(0, reservoir.field_size, pressure.shape[0])
    y = np.linspace(0, reservoir.field_size, pressure.shape[1])
    xgrid, ygrid = np.meshgrid(x, y)

    fig = plt.figure(figsize=(20, 8))

    # Первый вид (сверху-сбоку)
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(xgrid, ygrid, pressure.T, cmap='viridis',
                             linewidth=0, antialiased=True)

    # Отображение скважин
    for well in reservoir.wells:
        z = pressure[int(well.x / reservoir.grid_step)][int(well.y / reservoir.grid_step)]
        color = 'red' if well.well_type == 'injection' else 'yellow'
        marker = 'v' if well.well_type == 'injection' else '^'
        ax1.scatter([well.x], [well.y], [z], color=color, s=100,
                    marker=marker, label=f'Скважина {int(well.x)}:{int(well.y)}')

    ax1.view_init(elev=30, azim=45)
    plt.colorbar(surf1, ax=ax1, label='Давление, Па')
    ax1.set_xlabel('X, км')
    ax1.set_ylabel('Y, км')
    ax1.set_zlabel('Давление, Па')
    ax1.set_title('Карта распределения пластового давления (вид 1)')
    ax1.legend()

    # Второй вид (сверху)
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(xgrid, ygrid, pressure.T, cmap='viridis',
                             linewidth=0, antialiased=True)

    # Отображение скважин
    for well in reservoir.wells:
        z = pressure[int(well.x / reservoir.grid_step)][int(well.y / reservoir.grid_step)]
        color = 'red' if well.well_type == 'injection' else 'yellow'
        marker = 'v' if well.well_type == 'injection' else '^'
        ax2.scatter([well.x], [well.y], [z], color=color, s=100,
                    marker=marker, label=f'Скважина {well.x}:{well.y}')

    ax2.view_init(elev=20, azim=120)
    plt.colorbar(surf2, ax=ax2, label='Давление, Па')
    ax2.set_xlabel('X, км')
    ax2.set_ylabel('Y, км')
    ax2.set_zlabel('Давление, Па')
    ax2.set_title('Карта распределения пластового давления (вид 2)')
    ax2.legend()

    plt.tight_layout()
    _save_or_show(save_path)


def plot_flow_rates(wells: list[OilWell], reservoir: OilReservoir, save_path=None, figsize=(12, 8)):
    """График дебитов и давлений скважин"""
    n = len(wells)
    rows = (n + 1) // 2
    cols = 2

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.asarray(axs).ravel()

    for i, well in enumerate(wells):
        if well.well_type == 'production':
            axs[i].plot(reservoir.time_points, well.flow_rate_history, 'b-', linewidth=2, label=f'Скважина {int(well.x)}:{int(well.y)}')
            axs[i].set_title(f'Дебит скважины {int(well.x)}:{int(well.y)}')
            axs[i].set_ylabel('Дебит, м³/сут')
        else:
            axs[i].plot(reservoir.time_points, well.pressure_history, 'r-', linewidth=2, label=f'Скважина {int(well.x)}:{int(well.y)}')
            axs[i].set_title(f'Давление скважины {int(well.x)}:{int(well.y)}')
            axs[i].set_ylabel('Давление, Па')

        axs[i].set_xlabel('Время, сут')
        axs[i].grid(True)
        axs[i].legend()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    _save_or_show(save_path)
