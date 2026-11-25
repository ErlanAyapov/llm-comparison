"""Точка входа эксперимента (протокол: docs/experiment_protocol.md).

  python main.py build                 # собрать замороженные задачи  -> tasks/
  python main.py run [аргументы]       # прогон моделей               -> results/<run_id>/
      напр.: python main.py run --exp e2 --limit 10 --run-id pilot

Оценка ответов (evaluate.py) — отдельный этап после прогонов.
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "run"):
        print(__doc__)
        sys.exit(1)
    cmd, sys.argv = sys.argv[1], [sys.argv[0]] + sys.argv[2:]
    if cmd == "build":
        import build_tasks
        build_tasks.main()
    else:
        import run_experiment
        run_experiment.main()
