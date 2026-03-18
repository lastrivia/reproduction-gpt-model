import random

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


def plot_training_curve(iteration: list, ppl: list, show: bool = False, save: str = None):
    plt.figure(figsize=(8, 6))
    plt.plot(iteration, ppl)

    ax = plt.gca()
    ax.set_yscale("log")

    y_lim_ref = ppl[len(ppl) // 20] # ignore first 5%; assumed approximately monotonic
    y_lim_presets = [125, 250, 490, 1250]
    y_lim = ([x for x in y_lim_presets if x > y_lim_ref] + [2500])[0]
    ax.set_ylim(top=y_lim)

    ax.grid(True, which='both', axis='both')

    ax.yaxis.set_major_locator(
        ticker.LogLocator(
            base=10,
            subs=range(1,6) if y_lim > 250 else range(1,10)
        )
    )
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.xlabel("Iteration")
    plt.ylabel("Perplexity (PPL)")
    plt.title("Training Curve")
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()


if __name__ == "__main__":
    iteration = list(range(100, 90001, 100))
    random.seed(42)
    ppl = [1.15e7 / (x ** 1.38) * random.gauss(1.0, 0.05) + 20 for x in iteration]
    plot_training_curve(iteration, ppl, show=True, save="test.png")