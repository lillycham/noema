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

def strip(frames, gap=1):
    """Frames (n, H, W) side by side in one picture (H, n * (W + gap) - gap), with masked gaps."""
    frames = np.asarray(frames, dtype=float)
    n, H, W = frames.shape
    out = np.full((H, n * (W + gap) - gap), np.nan)
    for t in range(n):
        out[:, t * (W + gap):t * (W + gap) + W] = frames[t]
    return np.ma.masked_invalid(out)

def save_worlds(items, path, prompt_frames):
    """A picture of each (rule, truth, written, label) in items, each (n, H, W): three strips of
    frames, the true ones, the model's, and where the two differ (red). A blue bar marks the frames
    of the prompt, which the model was given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, H, W = np.asarray(items[0][1]).shape
    fig, axes = plt.subplots(3 * len(items), 1, figsize=(0.55 * n * (W + 1), 0.62 * H * 3 * len(items)), squeeze=False)
    for k, (rule, truth, written, label) in enumerate(items):
        wrong = np.asarray(truth) != np.asarray(written)
        panels = [("truth", truth, "binary"), ("model", written, "binary"),
                  (f"wrong\n{wrong[prompt_frames:].mean():.0%}", wrong, "Reds")]
        for ax, (title, grids, cmap) in zip(axes[3 * k:3 * k + 3, 0], panels):
            cm = matplotlib.colormaps[cmap].with_extremes(bad="#c8c8c8")        # the gaps between frames, not a 0
            ax.imshow(strip(grids), cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            ax.plot([-0.5, prompt_frames * (W + 1) - 1.5], [-0.8, -0.8], color="tab:blue", lw=2, clip_on=False)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_ylabel(title, fontsize=7, rotation=0, ha="right", va="center")
        axes[3 * k, 0].set_title(f"rule {rule}, {label}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
