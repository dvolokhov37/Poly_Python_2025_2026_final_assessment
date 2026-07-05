from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from well_chisl import WellBoreStorageSimulator
from oil_reservoir_new import OilReservoir, OilWell, WellTypeEnum
from visualization_new import plot_flow_rates, plot_pressure_2d, plot_pressure_3d_views
from well_map_builder import generate_well_map

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

_current_reservoir: OilReservoir | None = None
_current_wells: list[OilWell] = []
_current_wellbore_sim: WellBoreStorageSimulator | None = None


class ToolError(Exception):
    pass


def _ok(data: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "data": data, "error": None}


def _err(exc: Exception | str) -> dict[str, Any]:
    return {"status": "error", "data": None, "error": str(exc)}


def _number(params: dict[str, Any], name: str, default: float) -> float:
    value = params.get(name, default)
    try:
        value = float(str(value).replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ToolError(f"Parameter {name} must be a number, got {value!r}.") from exc
    return value


def _integer(params: dict[str, Any], name: str, default: int, minimum: int = 2, maximum: int | None = None) -> int:
    value = int(round(_number(params, name, default)))
    if value < minimum:
        raise ToolError(f"Parameter {name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ToolError(f"Parameter {name} must be <= {maximum}.")
    return value

def _output_path(value: str | None, default_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if value is None:
        filename = default_name
    else:
        filename = Path(value).name
    base_path = OUTPUT_DIR / filename

    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = OUTPUT_DIR / new_name
        if not new_path.exists():
            return new_path
        counter += 1

def _tool_result_to_text(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)


def init_and_simulate(params: dict[str, Any] | None = None) -> dict[str, Any]:
    global _current_reservoir, _current_wells
    params = dict(params or {})
    try:
        time_step = _number(params, "time_step", 0.001)
        grid_step = _number(params, "grid_step", 0.5)
        field_size = _number(params, "field_size", 10.0)
        base_pressure = _number(params, "base_pressure", 5000.0)
        wells_data = params.get("wells") or [
            {"x": 3, "y": 3, "initial_pressure": 9000, "type": "injection"},
            {"x": 7, "y": 8, "initial_pressure": 9000, "type": "injection"},
            {"x": 4, "y": 3, "initial_pressure": 1000, "type": "production"},
            {"x": 8, "y": 8, "initial_pressure": 1000, "type": "production"},
        ]

        wells: list[OilWell] = []
        for item in wells_data:
            well_type = item.get("type", item.get("well_type", "production"))
            if str(well_type).lower() in {"inj", "injector", "нагнетательная", "нагнетатель"}:
                well_type = "injection"
            elif str(well_type).lower() in {"prod", "producer", "добывающая", "добыча"}:
                well_type = "production"
            well = OilWell(
                x=_number(item, "x", 0.0),
                y=_number(item, "y", 0.0),
                initial_pressure=_number(item, "initial_pressure", _number(item, "pressure", base_pressure)),
                well_type=WellTypeEnum(str(well_type).lower()),
            )
            wells.append(well)

        reservoir = OilReservoir(time_step, grid_step, field_size, base_pressure)
        reservoir.add_wells(wells)
        reservoir.simulate_pressure()
        for well in wells:
            well.calculate_flow_rate(reservoir)
            well.calculate_pressure(reservoir)

        _current_reservoir = reservoir
        _current_wells = wells
        last_pressure = reservoir.pressure[-1]
        return _ok(
            {
                "message": "2D reservoir simulation completed.",
                "grid_size": reservoir.grid_size,
                "time_steps": reservoir.time_steps,
                "min_pressure": float(np.nanmin(last_pressure)),
                "max_pressure": float(np.nanmax(last_pressure)),
                "well_count": len(wells),
            }
        )
    except Exception as exc:
        return _err(exc)


def get_productivity_indices(params: dict[str, Any] | None = None) -> dict[str, Any]:
    if _current_reservoir is None:
        return _err("Call init_and_simulate first.")
    try:
        data: dict[str, Any] = {}
        for idx, well in enumerate(_current_wells, start=1):
            data[f"well_{idx}_{well.x:g}_{well.y:g}_{well.well_type.value}"] = float(well.calculate_productivity_index())
        return _ok(data)
    except Exception as exc:
        return _err(exc)


def plot_pressure_map_2d(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    if _current_reservoir is None:
        return _err("Call init_and_simulate first.")
    try:
        path = _output_path(params.get("output_file"), "pressure_2d.png")
        plot_pressure_2d(_current_reservoir, save_path=str(path))
        return _ok({"image_path": str(path)})
    except Exception as exc:
        return _err(exc)


def plot_pressure_map_3d(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    if _current_reservoir is None:
        return _err("Call init_and_simulate first.")
    try:
        path = _output_path(params.get("output_file"), "pressure_3d.png")
        plot_pressure_3d_views(_current_reservoir, save_path=str(path))
        return _ok({"image_path": str(path)})
    except Exception as exc:
        return _err(exc)


def plot_well_flow_rates(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    if _current_reservoir is None:
        return _err("Call init_and_simulate first.")
    try:
        path = _output_path(params.get("output_file"), "flow_rates.png")
        plot_flow_rates(_current_wells, _current_reservoir, save_path=str(path))
        return _ok({"image_path": str(path)})
    except Exception as exc:
        return _err(exc)


def plot_productivity_indices(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    if _current_reservoir is None:
        return _err("Call init_and_simulate first.")
    try:
        path = _output_path(params.get("output_file"), "productivity_indices.png")
        _current_reservoir.plot_productivity_indices(save_path=str(path))
        return _ok({"image_path": str(path)})
    except Exception as exc:
        return _err(exc)


def run_wellbore_simulation(params: dict[str, Any] | None = None) -> dict[str, Any]:
    global _current_wellbore_sim
    params = dict(params or {})
    try:
        sim_params = {
            "k_mD": _number(params, "k_mD", 10.0),
            "p_e_atm": _number(params, "p_e_atm", 250.0),
            "r_w": _number(params, "r_w", 0.1),
            "r_e": _number(params, "r_e", 250.0),
            "h": _number(params, "h", 10.0),
            "mu_cP": _number(params, "mu_cP", 1.5),
            "phi": _number(params, "phi", 0.2),
            "c_atm": _number(params, "c_atm", 3e-4),
            "Q_const_m3day": _number(params, "Q_const_m3day", 100.0),
            "C_storage_m3_atm": _number(params, "C_storage_m3_atm", 1.0),
            "N": _integer(params, "N", 300, minimum=5, maximum=3000),
            "Nt": _integer(params, "Nt", 15000, minimum=5, maximum=200000),
            "t_total_days": _number(params, "t_total_days", 100.0),
        }
        sim = WellBoreStorageSimulator(sim_params)
        sim.solve()
        _current_wellbore_sim = sim
        idx10 = int(np.argmin(np.abs(sim.T_day - min(10.0, sim.t_total_days))))
        return _ok(
            {
                "message": "Wellbore storage simulation completed.",
                "bottomhole_pressure_at_10_days_atm": float(sim.P_hist[idx10, 0] / sim.atm_to_Pa),
                "sandface_rate_at_10_days_m3day": float(sim.Q_m3day[idx10]),
                "productivity_m3day_Pa": float(sim.kprod / sim.atm_to_Pa * 86400.0),
                "hydraulic_diffusivity_m2_s": float(sim.kappa),
                "N": sim.N,
                "Nt": sim.Nt,
                "t_total_days": sim.t_total_days,
            }
        )
    except Exception as exc:
        return _err(exc)


def plot_wellbore_results(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    if _current_wellbore_sim is None:
        return _err("Call run_wellbore_simulation first.")
    try:
        path = _output_path(params.get("output_file"), "wellbore_plots.png")
        _current_wellbore_sim.plot_results(save_path=str(path))
        return _ok({"image_path": str(path)})
    except Exception as exc:
        return _err(exc)


TOOL_FUNCTIONS = {
    "init_and_simulate": init_and_simulate,
    "get_productivity_indices": get_productivity_indices,
    "plot_pressure_map_2d": plot_pressure_map_2d,
    "plot_pressure_map_3d": plot_pressure_map_3d,
    "plot_well_flow_rates": plot_well_flow_rates,
    "plot_productivity_indices": plot_productivity_indices,
    "run_wellbore_simulation": run_wellbore_simulation,
    "plot_wellbore_results": plot_wellbore_results,
    "build_well_map": generate_well_map,
}


TOOLS = {
    "init_and_simulate": {
        "func": init_and_simulate,
        "description": "Запустить 2D симуляцию давления в пласте. Все параметры необязательны: time_step, grid_step, field_size, base_pressure, wells (если не указан, используется стандартный набор скважин).",
    },
    "get_productivity_indices": {
        "func": get_productivity_indices,
        "description": "Вернуть коэффициенты продуктивности скважин после init_and_simulate. Параметры не требуются.",
    },
    "plot_pressure_map_2d": {
        "func": plot_pressure_map_2d,
        "description": "Сохранить 2D карту давления. Параметр output_file необязателен (по умолчанию pressure_2d.png).",
    },
    "plot_pressure_map_3d": {
        "func": plot_pressure_map_3d,
        "description": "Сохранить два 3D вида карты давления. Параметр output_file необязателен (по умолчанию pressure_3d.png).",
    },
    "plot_well_flow_rates": {
        "func": plot_well_flow_rates,
        "description": "Сохранить графики забойного давления и дебита для всех скважин. Параметр output_file необязателен (по умолчанию flow_rates.png).",
    },
    "plot_productivity_indices": {
        "func": plot_productivity_indices,
        "description": "Сохранить столбчатую диаграмму коэффициентов продуктивности. Параметр output_file необязателен (по умолчанию productivity_indices.png).",
    },
    "run_wellbore_simulation": {
        "func": run_wellbore_simulation,
        "description": "Запустить радиальную симуляцию с учётом ствола скважины. Все параметры необязательны; если параметр не указан, используется значение по умолчанию. Параметры: k_mD, p_e_atm, r_w, r_e, h, mu_cP, phi, c_atm, Q_const_m3day, C_storage_m3_atm, N, Nt, t_total_days.",
    },
    "plot_wellbore_results": {
        "func": plot_wellbore_results,
        "description": "Сохранить четыре графика для радиальной симуляции. Параметр output_file необязателен (по умолчанию wellbore_plots.png).",
    },
    "build_well_map": {
        "func": generate_well_map,
        "description": "Построить интерактивную карту скважин (траектории, точки среза, зоны дренирования) с сохранением в HTML. Параметр output_html необязателен (по умолчанию well_map.html).",
    },
}


OPENAI_STYLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "init_and_simulate",
            "description": TOOLS["init_and_simulate"]["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    "time_step": {"type": "number"},
                    "grid_step": {"type": "number"},
                    "field_size": {"type": "number"},
                    "base_pressure": {"type": "number"},
                    "total_time": {"type": "number"},
                    "wells": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "initial_pressure": {"type": "number"},
                                "type": {"type": "string", "enum": ["injection", "production"]},
                            },
                            "required": ["x", "y", "initial_pressure", "type"],
                        },
                    },
                },
            },
        },
    },
    {"type": "function", "function": {"name": "get_productivity_indices", "description": TOOLS["get_productivity_indices"]["description"], "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "plot_pressure_map_2d", "description": TOOLS["plot_pressure_map_2d"]["description"], "parameters": {"type": "object", "properties": {"output_file": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "plot_pressure_map_3d", "description": TOOLS["plot_pressure_map_3d"]["description"], "parameters": {"type": "object", "properties": {"output_file": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "plot_well_flow_rates", "description": TOOLS["plot_well_flow_rates"]["description"], "parameters": {"type": "object", "properties": {"output_file": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "plot_productivity_indices", "description": TOOLS["plot_productivity_indices"]["description"], "parameters": {"type": "object", "properties": {"output_file": {"type": "string"}}}}},
    {
        "type": "function",
        "function": {
            "name": "run_wellbore_simulation",
            "description": TOOLS["run_wellbore_simulation"]["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    "k_mD": {"type": "number"},
                    "p_e_atm": {"type": "number"},
                    "r_w": {"type": "number"},
                    "r_e": {"type": "number"},
                    "h": {"type": "number"},
                    "mu_cP": {"type": "number"},
                    "phi": {"type": "number"},
                    "c_atm": {"type": "number"},
                    "Q_const_m3day": {"type": "number"},
                    "C_storage_m3_atm": {"type": "number"},
                    "N": {"type": "integer"},
                    "Nt": {"type": "integer"},
                    "t_total_days": {"type": "number"},
                },
            },
        },
    },
    {"type": "function", "function": {"name": "plot_wellbore_results", "description": TOOLS["plot_wellbore_results"]["description"], "parameters": {"type": "object", "properties": {"output_file": {"type": "string"}}}}},
    {
        "type": "function",
        "function": {
            "name": "build_well_map",
            "description": TOOLS["build_well_map"]["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    "output_html": {"type": "string", "description": "Имя выходного HTML-файла (сохраняется в папке output, по умолчанию well_map.html)."}
                },
                "additionalProperties": False
            }
        }
    }
]


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return _tool_result_to_text(_err(f"Unknown tool: {name}"))
    return _tool_result_to_text(func(arguments or {}))
