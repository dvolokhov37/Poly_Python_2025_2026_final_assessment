import enum
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class WellTypeEnum(str, enum.Enum):
    injection = "injection"
    production = "production"


class OilWell:
    def __init__(self, x: float, y: float, initial_pressure: float, well_type: WellTypeEnum | str):
        self.x = x
        self.y = y
        self.initial_pressure = initial_pressure
        self.pressure_history = []
        self.well_type = WellTypeEnum(well_type)
        self.flow_rate_history = []

    def calculate_flow_rate(self, reservoir: "OilReservoir"):
        """Расчёт дебита для скважины"""
        self.flow_rate_history = []
        for k in range(len(reservoir.time_points)):
            x_idx = int(self.x)
            y_idx = int(self.y)
            p_x = (
                reservoir.pressure[k][x_idx + 1][y_idx]
                - reservoir.pressure[k][x_idx][y_idx]
            ) / reservoir.grid_step
            p_y = (
                reservoir.pressure[k][x_idx][y_idx + 1]
                - reservoir.pressure[k][x_idx][y_idx]
            ) / reservoir.grid_step
            self.flow_rate_history.append(p_y + p_x)

    def calculate_pressure(self, reservoir: "OilReservoir"):
        """Возвращает давление в скважине с учетом базового давления"""
        x_idx = int(self.x)
        y_idx = int(self.y)
        self.pressure_history = [
            reservoir.pressure[k][x_idx][y_idx]
            for k in range(len(reservoir.time_points))
        ]

    def calculate_productivity_index(self):
        """Расчет коэффициента продуктивности скважины"""
        if not self.pressure_history:
            return 0
        avg_pressure = np.mean(self.pressure_history)
        avg_flow_rate = np.mean(self.flow_rate_history)
        return abs(avg_flow_rate / avg_pressure) if avg_pressure != 0 else 0



class OilReservoir:
    def __init__(self, time_step=0.001, grid_step=0.5, field_size=10, base_pressure=600.0):
        self.time_step = time_step
        self.grid_step = grid_step
        self.field_size = field_size
        self.base_pressure = base_pressure  # Новое: базовое давление в пласте
        self.time_points = np.linspace(0, 1, int(1 / time_step))
        self.x = self.y = np.linspace(0, field_size, int(field_size / grid_step))
        self.time_steps = len(self.time_points)
        self.grid_size = len(self.x)
        self.pressure = np.full((self.time_steps, self.grid_size, self.grid_size), self.base_pressure)
        self.wells = []

    def add_wells(self, wells: list[OilWell]) -> None:
        self.wells = wells
        for well in wells:
            x_idx = int(well.x / self.grid_step)
            y_idx = int(well.y / self.grid_step)
            self.pressure[0][x_idx][y_idx] = well.initial_pressure

    def simulate_pressure(self):
        logging.info("Starting solver...")
        for k in range(self.time_steps - 1):
            # Основной цикл по внутренним узлам сетки
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    conductivity = self.calculate_conductivity(i, j)
                    self.pressure[k + 1, i, j] = (
                            self.pressure[k, i, j]
                            + self.time_step
                            * conductivity
                            / self.grid_step ** 2
                            * (
                                    self.pressure[k, i + 1, j]
                                    + self.pressure[k, i - 1, j]
                                    + self.pressure[k, i, j + 1]
                                    + self.pressure[k, i, j - 1]
                                    - 4 * self.pressure[k, i, j]
                            )
                    )

            # Учет давления в скважинах
            for well in self.wells:
                x_idx = int(well.x / self.grid_step)
                y_idx = int(well.y / self.grid_step)
                self.pressure[k + 1, x_idx, y_idx] = well.initial_pressure  # Фиксация давления

            logging.info(f"Step {k + 1}/{self.time_steps - 1} completed.")

    def calculate_conductivity(self, i: int, j: int) -> float:
        return 0.1 if i > j else 0.5

    def save_results(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.pressure, f)
        logging.info(f"Results saved to {filename}")

    def load_pressure_data(self, filename):
        with open(filename, "rb") as f:
            self.pressure = pickle.load(f)

    def plot_productivity_indices(self, save_path=None):
        """График коэффициентов продуктивности скважин"""
        fig, ax = plt.subplots(figsize=(10, 6))
        wells_pi = [well.calculate_productivity_index() for well in self.wells]
        well_names = [f'Скважина {int(well.x)}:{int(well.y)}\n{well.well_type}' for well in self.wells]
        colors = ['red' if well.well_type == 'injection' else 'blue' for well in self.wells]

        ax.bar(well_names, wells_pi, color=colors)
        ax.set_xlabel('Скважины')
        ax.set_ylabel('Коэффициент продуктивности, м³/(сут·Па)')
        ax.set_title('Коэффициенты продуктивности скважин')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
