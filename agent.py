from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

import requests

from tools import OPENAI_STYLE_TOOLS, TOOL_FUNCTIONS, execute_tool


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:8b"
MAX_ROUNDS = 10


SYSTEM_PROMPT = """
Ты агент для нефтегазовых расчетов. У тебя есть инструменты:
1. 2D моделирование пласта: init_and_simulate, plot_pressure_map_2d, plot_pressure_map_3d,
   plot_well_flow_rates, get_productivity_indices, plot_productivity_indices.
2. Радиальная расчетка со стволом скважины: run_wellbore_simulation, plot_wellbore_results.
3. Построение интерактивной карты скважин (траектории, точки среза, зоны дренирования): build_well_map.

**Важное правило для всех инструментов:**
- Все входные параметры являются необязательными. Если пользователь не указал какой-либо параметр, используй значение по умолчанию (оно уже задано в инструменте).
- Передавай в вызов инструмента только те параметры, которые явно упомянуты в запросе пользователя. Не добавляй параметры со значениями по умолчанию, если они не были запрошены.
- Не запрашивай у пользователя недостающие параметры – они уже имеют значения по умолчанию.

Примеры правильных вызовов:
- Запрос: «радиус контура 260 м и давление 245 атм» — вызови run_wellbore_simulation с {"r_e": 260, "p_e_atm": 245}.
- Запрос: «шаг сетки 0.2» — вызови init_and_simulate с {"grid_step": 0.2}.
- Запрос: «сохрани карту как my_map.png» — вызови plot_pressure_map_2d с {"output_file": "my_map.png"}.

После выполнения инструментов дай краткий Final-ответ: что посчитано, какие важные числа получены, и где лежат сохраненные файлы.
Не придумывай результаты – используй только Observation от инструментов.
"""


EXAMPLES: dict[str, dict[str, Any]] = {
    "1": {
        "title": "2D расчет пласта",
        "mode": "tool",
        "tool": "init_and_simulate",
        "args": {
            "time_step": 0.001,
            "grid_step": 0.5,
            "field_size": 10,
            "base_pressure": 5000,
            "wells": [
                {"x": 3, "y": 3, "initial_pressure": 9000, "type": "injection"},
                {"x": 7, "y": 8, "initial_pressure": 9000, "type": "injection"},
                {"x": 4, "y": 3, "initial_pressure": 1000, "type": "production"},
                {"x": 8, "y": 8, "initial_pressure": 1000, "type": "production"},
            ],
        },
    },
    "2": {
        "title": "Коэффициенты продуктивности",
        "mode": "tool",
        "tool": "get_productivity_indices",
        "args": {},
        "requires": "Перед этим выполни пример 1.",
    },
    "3": {
        "title": "2D карта давления",
        "mode": "tool",
        "tool": "plot_pressure_map_2d",
        "args": {"output_file": "example_pressure_2d.png"},
        "requires": "Перед этим выполни пример 1.",
    },
    "4": {
        "title": "3D карта давления",
        "mode": "tool",
        "tool": "plot_pressure_map_3d",
        "args": {"output_file": "example_pressure_3d.png"},
        "requires": "Перед этим выполни пример 1.",
    },
    "5": {
        "title": "Графики скважин",
        "mode": "tool",
        "tool": "plot_well_flow_rates",
        "args": {"output_file": "example_flow_rates.png"},
        "requires": "Перед этим выполни пример 1.",
    },
    "6": {
        "title": "График коэффициентов продуктивности",
        "mode": "tool",
        "tool": "plot_productivity_indices",
        "args": {"output_file": "example_productivity_indices.png"},
        "requires": "Перед этим выполни пример 1.",
    },
    "7": {
        "title": "Радиальная расчетка со стволом",
        "mode": "tool",
        "tool": "run_wellbore_simulation",
        "args": {
            "k_mD": 10,
            "p_e_atm": 250,
            "r_w": 0.1,
            "r_e": 250,
            "h": 10,
            "mu_cP": 1.5,
            "phi": 0.2,
            "c_atm": 3e-4,
            "Q_const_m3day": 100,
            "C_storage_m3_atm": 1,
            "N": 300,
            "Nt": 15000,
            "t_total_days": 100,
        },
    },
    "8": {
        "title": "Графики радиальной расчетки",
        "mode": "tool",
        "tool": "plot_wellbore_results",
        "args": {"output_file": "example_wellbore_plots.png"},
        "requires": "Перед этим выполни пример 7.",
    },
    "9": {
        "title": "Полный пример по 2D пласту",
        "mode": "agent",
        "query": (
            "Запусти 2D симуляцию пласта с time_step=0.001, grid_step=0.5, "
            "field_size=10, base_pressure=5000. Скважины: нагнетательные "
            "(3,3) давление 9000 и (7,8) давление 9000; добывающие "
            "(4,3) давление 1000 и (8,8) давление 1000. После расчета "
            "построй 2D карту давления, график дебитов и давлений, рассчитай "
            "коэффициенты продуктивности и построй график коэффициентов продуктивности."
        ),
    },
    "10": {
        "title": "Полный пример по радиальной расчетке",
        "mode": "agent",
        "query": (
            "Запусти радиальную расчетку со стволом скважины: k_mD=10, "
            "p_e_atm=250, r_w=0.1, r_e=250, h=10, mu_cP=1.5, phi=0.2, "
            "c_atm=3e-4, Q_const_m3day=100, C_storage_m3_atm=1, "
            "N=300, Nt=15000, t_total_days=100. После расчета построй графики."
        ),
    },
    "11": {
        "title": "Построение карты скважин (траектории, точки среза, зоны дренирования)",
        "mode": "tool",
        "tool": "build_well_map",
        "args": {},
        "description": "Генерирует интерактивную HTML-карту по умолчанию (сохраняется в output/well_map.html)."
    },
}

class AgentError(Exception):
    pass

class OllamaClient:
    def __init__(self, model: str = MODEL_NAME, url: str = OLLAMA_URL, timeout: int = 300):
        self.model = model
        self.url = url
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class Agent:
    def __init__(self, client: OllamaClient, max_rounds: int = MAX_ROUNDS):
        self.client = client
        self.max_rounds = max_rounds

    def run(self, user_query: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        for round_idx in range(1, self.max_rounds + 1):
            assistant_message = self.client.chat(messages, OPENAI_STYLE_TOOLS)
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                content = (assistant_message.get("content") or "").strip()
                return content or "Готово, но модель не вернула текст ответа."

            for call in tool_calls:
                function = call.get("function", {})
                tool_name = function.get("name")
                arguments = _parse_tool_arguments(function.get("arguments"))
                print(f"[round {round_idx}] tool: {tool_name} {json.dumps(arguments, ensure_ascii=False)}")
                observation = execute_tool(tool_name, arguments)
                print(f"[round {round_idx}] observation: {observation[:500]}")
                messages.append(
                    {
                        "role": "tool",
                        "content": observation,
                        "tool_name": tool_name,
                    }
                )

        return "Превышено число шагов агента. Последние инструменты выполнены, но финальный ответ не получен."


def run_smoke_test() -> None:
    print("Running tool smoke test...")
    steps = [
        (
            "init_and_simulate",
            {
                "time_step": 0.01,
                "grid_step": 0.5,
                "field_size": 10,
                "base_pressure": 5000,
                "wells": [
                    {"x": 3, "y": 3, "initial_pressure": 9000, "type": "injection"},
                    {"x": 7, "y": 8, "initial_pressure": 9000, "type": "injection"},
                    {"x": 4, "y": 3, "initial_pressure": 1000, "type": "production"},
                    {"x": 8, "y": 8, "initial_pressure": 1000, "type": "production"},
                ],
            },
        ),
        ("get_productivity_indices", {}),
        ("plot_pressure_map_2d", {"output_file": "test_pressure_2d.png"}),
        ("plot_pressure_map_3d", {"output_file": "test_pressure_3d.png"}),
        ("plot_well_flow_rates", {"output_file": "test_flow_rates.png"}),
        ("plot_productivity_indices", {"output_file": "test_productivity_indices.png"}),
        ("run_wellbore_simulation", {"N": 30, "Nt": 120, "t_total_days": 2}),
        ("plot_wellbore_results", {"output_file": "test_wellbore.png"}),
    ]
    for name, args in steps:
        result = TOOL_FUNCTIONS[name](args)
        print(f"{name}: {json.dumps(result, ensure_ascii=False, allow_nan=False)}")
        if result["status"] != "ok":
            raise AgentError(f"Smoke test failed at {name}: {result['error']}")
    print("Smoke test passed.")


def print_examples() -> None:
    print("\nПримеры, которые можно вызвать цифрами:")
    for number, example in EXAMPLES.items():
        suffix = f" ({example['requires']})" if example.get("requires") else ""
        print(f"  {number}. {example['title']}{suffix}")
    print("  help. Показать этот список еще раз")
    print("  exit. Выход")


def short_result_summary(example: dict[str, Any], result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return f"Пример '{example['title']}' не выполнен. Ошибка: {result.get('error')}"

    data = result.get("data") or {}
    tool = example.get("tool")
    if tool == "init_and_simulate":
        return (
            "Выполнен 2D расчет пласта; "
            f"сетка {data.get('grid_size')}x{data.get('grid_size')}, "
            f"шагов {data.get('time_steps')}, "
            f"давление от {data.get('min_pressure')} до {data.get('max_pressure')}."
        )
    if tool == "get_productivity_indices":
        return f"Рассчитаны коэффициенты продуктивности для {len(data)} скважин."
    if tool in {"plot_pressure_map_2d", "plot_pressure_map_3d", "plot_well_flow_rates", "plot_productivity_indices", "plot_wellbore_results"}:
        return f"Построен и сохранен график: {data.get('image_path')}"
    if tool == "run_wellbore_simulation":
        return (
            "Выполнена радиальная расчетка со стволом; "
            f"забойное давление около {data.get('bottomhole_pressure_at_10_days_atm'):.3g} атм, "
            f"дебит из пласта около {data.get('sandface_rate_at_10_days_m3day'):.3g} м3/сут."
        )
    return f"Выполнен пример '{example['title']}'."


def run_numbered_example(agent: Agent, number: str) -> None:
    example = EXAMPLES[number]
    print(f"\nПример {number}: {example['title']}")
    if example["mode"] == "tool":
        result = TOOL_FUNCTIONS[example["tool"]](example["args"])
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        print(short_result_summary(example, result))
        return
    print("\nЗапрос:\n" + example["query"])
    print("\nОтвет:\n" + agent.run(example["query"]))
    print(f"Выполнен пример '{example['title']}'.")


def interactive_loop(agent: Agent) -> None:
    print("Нефтегазовый агент. Введите запрос, номер примера или exit для выхода.")
    print_examples()
    while True:
        try:
            query = input("\nЗапрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"exit", "quit", "q"}:
            return
        if query.lower() in {"help", "examples", "пример", "примеры"}:
            print_examples()
            continue
        if query in EXAMPLES:
            try:
                run_numbered_example(agent, query)
            except Exception as exc:
                print(f"Ошибка примера: {exc}")
            continue
        if not query:
            continue
        try:
            print("\nОтвет:\n" + agent.run(query))
            print("Запрос обработан агентом.")
        except Exception as exc:
            print(f"Ошибка агента: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Oil and gas calculation agent for Ollama.")
    parser.add_argument("query", nargs="*", help="User query. If omitted, interactive mode is started.")
    parser.add_argument("--test", action="store_true", help="Run local tool smoke test without Ollama.")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Ollama model name, default: {MODEL_NAME}.")
    parser.add_argument("--url", default=OLLAMA_URL, help=f"Ollama chat URL, default: {OLLAMA_URL}.")
    args = parser.parse_args()

    try:
        if args.test:
            run_smoke_test()
            return 0

        agent = Agent(OllamaClient(model=args.model, url=args.url))
        if args.query:
            print(agent.run(" ".join(args.query)))
        else:
            interactive_loop(agent)
        return 0
    except requests.RequestException as exc:
        print(
            "Не удалось обратиться к Ollama. Проверь, что запущен `ollama serve` "
            f"и загружена модель `{args.model}`. Детали: {exc}"
        )
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
