import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from pathlib import Path


class WellBoreStorageSimulator:

    def __init__(self, params=None):

        if params is None:
            params = {}
        self._set_parameters(params)
        self._convert_to_SI()
        self._setup_grid()
        self._compute_coefficients()

    def _set_parameters(self, params):
        # Исходные параметры (в промысловых единицах)
        self.k_mD = params.get('k_mD', 10.0)                 # проницаемость, мД
        self.p_e_atm = params.get('p_e_atm', 250.0)          # начальное пластовое давление, атм
        self.r_w = params.get('r_w', 0.1)                    # радиус скважины, м
        self.r_e = params.get('r_e', 250.0)                  # радиус контура питания, м
        self.h = params.get('h', 10.0)                       # толщина пласта, м
        self.mu_cP = params.get('mu_cP', 1.5)                # вязкость, сП
        self.phi = params.get('phi', 0.2)                    # пористость, д.ед.
        self.c_atm = params.get('c_atm', 3e-4)               # общая сжимаемость, 1/атм
        self.Q_const_m3day = params.get('Q_const_m3day', 100.0)  # дебит на поверхности, м³/сут
        self.C_storage_m3_atm = params.get('C_storage_m3_atm', 1.0)  # коэффициент влияния ствола, м³/атм
        self.N = params.get('N', 300)                        # число узлов по радиусу
        self.Nt = params.get('Nt', 15000)                    # число шагов по времени
        self.t_total_days = params.get('t_total_days', 100)  # общее время моделирования, сут

        self.t_total = self.t_total_days * 86400             # общее время, с

    def _convert_to_SI(self):
        # Переводные коэффициенты
        self.atm_to_Pa = 101325.0
        self.mD_to_m2 = 9.869233e-16
        self.cP_to_Pas = 0.001

        # Параметры в СИ
        self.p_e = self.p_e_atm * self.atm_to_Pa
        self.k = self.k_mD * self.mD_to_m2
        self.mu = self.mu_cP * self.cP_to_Pas
        self.c_total = self.c_atm / self.atm_to_Pa
        self.kappa = self.k / (self.phi * self.mu * self.c_total)   # пьезопроводность

        self.Q_const = self.Q_const_m3day / 86400.0                 # м³/с
        self.C_storage = self.C_storage_m3_atm / self.atm_to_Pa     # м³/Па
        self.kprod = 2 * math.pi * self.k * self.h / self.mu        # коэффициент продуктивности

    def _setup_grid(self):
        # Логарифмическая сетка по радиусу и равномерная по времени
        self.dh = np.log(self.r_e / self.r_w) / (self.N-1)
        self.dt = self.t_total / (self.Nt - 1)
        self.tau = self.dt * self.kappa / self.r_w**2

        self.T_sec = np.linspace(0, self.t_total, self.Nt)
        self.T_day = self.T_sec / 86400.0
        self.r = np.exp(np.linspace(np.log(self.r_w), np.log(self.r_e), self.N))

    def _compute_coefficients(self):
        dh = self.dh
        tau = self.tau
        N = self.N
        A = np.array([np.exp(-2 * i * dh) / (4 * dh) * (np.exp(2 * dh) - np.exp(-2 * dh))
                      for i in range(N - 1)])
        a = A * tau / dh**2
        b = a.copy()
        c = 2 * a + 1
        self.a = a
        self.b = b
        self.c = c

    def thomas(self, a, c, b, d):
        n = len(c)
        alpha = np.zeros(n)
        betta = np.zeros(n)

        # Прямая прогонка
        alpha[0] = b[0]/c[0]
        betta[0] = d[0]/c[0]
        for i in range(1, n):
            denom = c[i] - a[i] * alpha[i - 1]
            alpha[i] = b[i] / denom
            betta[i] = (d[i] - a[i] * betta[i - 1]) / denom

        # Обратная прогонка
        x = np.zeros(n)
        x[-1] = betta[-1]

        for i in range(n - 2, -1, -1):
            x[i] = betta[i] - alpha[i] * x[i + 1]

        return x

    def _solve_plast(self):
        P = np.full(self.N, self.p_e, dtype=float)
        Q_sandface = np.zeros(self.Nt, dtype=float)
        P_hist = np.zeros((self.Nt, self.N), dtype=float)

        P_hist[0] = P.copy()
        dh = self.dh
        dt = self.dt

        omega = (2 * np.pi * self.k * self.h) / (self.mu * self.dh * self.r_w)
        storage_term = self.C_storage / dt

        for t in range(1, self.Nt):
            a_diag = np.zeros(self.N)
            b_diag = np.zeros(self.N)
            c_diag = np.zeros(self.N)
            d_vec = np.zeros(self.N)

            # Внутренние узлы
            for i in range(1, self.N - 1):
                ai = self.a[i]
                a_diag[i] = -ai
                b_diag[i] = -ai
                c_diag[i] = 2 * ai + 1
                d_vec[i] = P[i]

            # Правая граница (постоянное давление)
            a_diag[self.N - 1] = 0.0
            c_diag[self.N - 1] = 1.0
            b_diag[self.N - 1] = 0.0
            d_vec[self.N - 1] = self.p_e

            # Левая граница (ствол скважины)
            a_diag[0] = 0.0
            c_diag[0] = storage_term + omega
            b_diag[0] = -omega
            d_vec[0] = -self.Q_const + storage_term * P[0]

            P_new = self.thomas(a_diag, c_diag, b_diag, d_vec)
            P_hist[t] = P_new.copy()

            # Производная для дебита из пласта
            deriv_z = (-1.5 * P_new[0] + 2.0 * P_new[1] - 0.5 * P_new[2]) / dh
            Q_sandface[t] = (2 * np.pi * self.k * self.h / (self.mu * self.r_w)) * deriv_z

            P = P_new.copy()

        return Q_sandface, P_hist

    def solve(self):
        self.Q, self.P_hist = self._solve_plast()
        self.Q_m3day = self.Q * 86400.0
        return self.Q, self.P_hist

    def plot_results(self, save_path=None):
        """
        Построение четырёх графиков.
        Если save_path указан, график сохраняется в файл, иначе отображается интерактивно.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Профиль давления на 10-е сутки
        ax1 = axes[0, 0]
        idx10 = np.argmin(np.abs(self.T_day - 10))
        ax1.plot(self.r, self.P_hist[idx10] / self.atm_to_Pa, 'b-', linewidth=2,
                 label='Пласт (постоянное давление на границе)')
        ax1.axvline(x=self.r_e, color='b', linestyle='--', alpha=0.7, label='Граница пласта')
        ax1.axhline(y=self.p_e_atm, color='k', linestyle=':', alpha=0.5, label='Начальное давление')
        ax1.set_xlabel('Радиус, м')
        ax1.set_ylabel('Давление, атм')
        ax1.set_title('Воронка депрессий при t=10 сут')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=9)

        # 2. Забойное давление
        ax2 = axes[0, 1]
        ax2.semilogx(self.T_day, self.P_hist[:, 0] / self.atm_to_Pa, 'k-', linewidth=2)
        ax2.set_xlabel('Время, сут')
        ax2.set_ylabel('Забойное давление, атм')
        ax2.set_title('Динамика забойного давления')
        ax2.grid(True, which='both', alpha=0.3)

        # 3. Дебит из пласта
        ax3 = axes[1, 0]
        ax3.semilogx(self.T_day, self.Q_m3day, 'b-', linewidth=2, label='Дебит из пласта')
        ax3.axhline(y=self.Q_const_m3day, color='k', linestyle=':', linewidth=1.5,
                    label=f'Дебит на поверхности ({self.Q_const_m3day} м³/сут)')
        ax3.set_xlabel('Время, сут')
        ax3.set_ylabel('Дебит, м³/сут')
        ax3.set_title('Динамика дебита из пласта')
        ax3.grid(True, which='both', alpha=0.3)
        ax3.legend(loc='lower right')

        # 4. Логарифмическая производная депрессии
        ax4 = axes[1, 1]
        delta_p = self.p_e_atm - self.P_hist[:, 0] / self.atm_to_Pa
        dp_dt = np.gradient(delta_p, self.T_sec)
        log_deriv = self.T_sec * dp_dt
        positive_time = self.T_day > 0
        ax4.loglog(self.T_day[positive_time], np.abs(log_deriv[positive_time]), 'b-', linewidth=2)
        ax4.set_xlabel('Время, сут')
        ax4.set_ylabel('dΔP/d(log t), атм')
        ax4.set_title('Логарифмическая производная депрессии')
        ax4.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_data(self, filename='well_data_c.npz'):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 r=self.r,
                 P_hist=self.P_hist,
                 Q_m3day=self.Q_m3day,
                 T_day=self.T_day,
                 atm_to_Pa=self.atm_to_Pa,
                 p_e_atm=self.p_e_atm,
                 r_e=self.r_e,
                 Q_const_m3day=self.Q_const_m3day,
                 T_sec=self.T_sec)
        print(f"Данные сохранены в {path}")


if __name__ == "__main__":
    params = {
        'k_mD': 10.0,
        'p_e_atm': 250.0,
        'r_w': 0.1,
        'r_e': 250.0,
        'h': 10.0,
        'mu_cP': 1.5,
        'phi': 0.2,
        'c_atm': 3e-4,
        'Q_const_m3day': 100.0,
        'C_storage_m3_atm': 1.0,
        'N': 1000,
        'Nt': 10000,
        't_total_days': 100
    }

    sim = WellBoreStorageSimulator(params)
    sim.solve()
    sim.plot_results()
    sim.plot_results("test.png")
    sim.save_data()
