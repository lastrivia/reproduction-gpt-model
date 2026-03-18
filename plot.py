import random

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


def plot_training_curve(iteration: list, ppl: list, show: bool = False, save: str = None):
    plt.figure(figsize=(8, 6))
    plt.plot(iteration, ppl)

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_ylim(top=2500)
    ax.grid(True, which='both', axis='both')

    ax.yaxis.set_major_locator(
        ticker.LogLocator(base=10, subs=range(1,6))
    )
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.xlabel("Iteration")
    plt.ylabel("Perplexity (PPL)")
    plt.title("Training Curve")
    # plt.grid()
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)


if __name__ == "__main__":
    iteration = list(range(100, 10001, 100))
    random.seed(42)
    ppl = [1.15e7 / (x ** 1.38) * random.gauss(1.0, 0.05) for x in iteration]
    plot_training_curve(iteration, ppl, show=True)