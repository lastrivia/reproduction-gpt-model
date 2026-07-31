import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


def _moving_window_means(values: Sequence[float], window: int) -> tuple[list[float], list[float], list[float]]:
    if window < 1:
        raise ValueError("window must be at least 1")

    full_means = []
    front_means = []
    back_means = []

    for i in range(len(values)):
        current_window = values[max(0, i - window + 1):i + 1]
        sorted_window = sorted(current_window)
        edge_count = max(1, math.ceil(len(current_window) * 0.25))

        full_means.append(sum(current_window) / len(current_window))
        front_means.append(sum(sorted_window[:edge_count]) / edge_count)
        back_means.append(sum(sorted_window[-edge_count:]) / edge_count)

    return full_means, front_means, back_means


def plot_training_curve(*, step: Sequence[int], log_ppl: Sequence[float], window: int, save: str | Path):
    if len(step) != len(log_ppl):
        raise ValueError("step and log_ppl must have the same length")
    if not step:
        raise ValueError("step and log_ppl must not be empty")

    mean_log_ppl, front_log_ppl, back_log_ppl = _moving_window_means(log_ppl, window)
    mean_ppl = [math.exp(value) for value in mean_log_ppl]
    front_ppl = [math.exp(value) for value in front_log_ppl]
    back_ppl = [math.exp(value) for value in back_log_ppl]

    plt.figure(figsize=(8, 6))
    plt.plot(step, mean_ppl, label=f"MA{window}", linewidth=1, color="C0")
    plt.fill_between(
        step,
        front_ppl,
        back_ppl,
        label=f"MA{window} low/high 25%",
        alpha=0.3,
        color="C0",
        linewidth=0,
    )

    ax = plt.gca()
    ax.set_yscale("log")

    second_half = mean_ppl[len(mean_ppl) // 2:]
    y_lim_ref = sum(second_half) / len(second_half) * 5.0
    y_lim_presets = [125, 250, 490, 1250, 2500, 4900, 12500]
    y_lim = ([x for x in y_lim_presets if x > y_lim_ref] + [25000])[0]
    ax.set_ylim(top=y_lim)

    ax.grid(True, which="both", axis="both")

    ax.yaxis.set_major_locator(
        ticker.LogLocator(
            base=10,
            subs=range(1, 6) if y_lim > 250 else range(1, 10),
        )
    )
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.xlabel("Step")
    plt.ylabel("Perplexity (PPL)")
    plt.title("Training Curve")
    plt.legend()
    plt.savefig(save)
    plt.close("all")


def _read_log(path: Path) -> tuple[list[int], list[float]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return [int(row["step"]) for row in rows], [float(row["loss"]) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_csv", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-w", "--window", required=True, type=int)
    args = parser.parse_args()

    if len(args.log_csv) != 1:
        parser.error("exactly one log.csv input is required")

    step, log_ppl = _read_log(Path(args.log_csv[0]))
    plot_training_curve(step=step, log_ppl=log_ppl, window=args.window, save=args.output)


if __name__ == "__main__":
    main()
