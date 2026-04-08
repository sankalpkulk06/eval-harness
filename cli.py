import argparse
import sys

from core.report import ReportBuilder
from core.runner import EvalRunner


def cmd_run(args):
    """Run evaluation on a single config."""
    runner = EvalRunner(args.config)
    results = runner.run(args.dataset)
    name = args.config.replace("/", "_").replace(".yaml", "")
    builder = ReportBuilder(results, config_name=name)
    json_path = builder.to_json()
    html_path = builder.to_html()
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")


def cmd_compare(args):
    """Compare multiple configs on the same dataset."""
    all_summaries = []
    for cfg_path in args.configs:
        runner = EvalRunner(cfg_path)
        results = runner.run(args.dataset)
        name = cfg_path.replace("/", "_").replace(".yaml", "")
        builder = ReportBuilder(results, config_name=name)
        json_path = builder.to_json()
        html_path = builder.to_html()
        print(f"[{name}] JSON: {json_path}  HTML: {html_path}")
        all_summaries.append({"config": cfg_path, **builder._aggregate()})

    print("\n--- Comparison Summary ---")
    for s in all_summaries:
        scores_str = " ".join(
            f"{k}={v:.3f}" for k, v in s["avg_scores"].items()
        )
        print(
            f"{s['config']}: latency={s['avg_latency_ms']:.1f}ms {scores_str}"
        )


def main():
    parser = argparse.ArgumentParser(prog="rag-eval-harness")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run evaluation on a single config")
    run_p.add_argument("--config", required=True, help="Path to YAML config")
    run_p.add_argument(
        "--dataset", required=True, help="Path to JSONL dataset"
    )

    cmp_p = sub.add_parser(
        "compare", help="Compare multiple configs on same dataset"
    )
    cmp_p.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Paths to YAML configs",
    )
    cmp_p.add_argument(
        "--dataset", required=True, help="Path to JSONL dataset"
    )

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
