"""Drawing rollouts: in the terminal, and as pictures of the truth next to what the model wrote."""
import numpy as np

def show(rows):
    """A rollout (n, W) of 0 and 1 as text, one row to a line."""
    return "\n".join("".join("█" if c else "·" for c in row) for row in np.asarray(rows))

def save_rollouts(items, path, prompt_rows):
    """A picture of each (rule, truth, written) in items, each (n, W). Three panels: the true
    rollout, the model's, and where the two differ (red). A line marks where the prompt ended
    and the model took over."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(items), 3, figsize=(7.5, 2.4 * len(items)), squeeze=False)
    for (rule, truth, written, label), row in zip(items, axes):
        wrong = (np.asarray(truth) != np.asarray(written))[prompt_rows:]
        panels = [("truth", truth, "binary"), ("model", written, "binary"),
                  (f"wrong: {wrong.mean():.0%}", np.asarray(truth) != np.asarray(written), "Reds")]
        for ax, (title, grid, cmap) in zip(row, panels):
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.axhline(prompt_rows - 0.5, color="tab:blue", lw=1)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(title, fontsize=8)
        row[0].set_ylabel(f"rule {rule}\n{label}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
