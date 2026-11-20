"""Offline report figures from validated evaluation tables, without model I/O."""

from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path


GROUPS = ("A", "B", "C")
COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73", "Naive": "#777777"}


@contextmanager
def _figure(size):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # An explicit Agg canvas avoids switching the host application's backend
    # or registering a GUI figure in pyplot's global figure manager.
    figure = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(figure)
    try:
        yield figure
    finally:
        from matplotlib import pyplot as plt

        plt.close(figure)
        figure.clear()


def _save(figure, path):
    # Exclusive creation also protects against another process creating a file
    # after the initial output validation.
    with path.open("xb") as handle:
        figure.savefig(handle, format="png", dpi=160, bbox_inches="tight")


def _axes_style(axes):
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color="#D9DDE2", linewidth=0.7)
    axes.set_axisbelow(True)


def _prices(bundle, path, period):
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, DateFormatter

    seed = bundle["plot_seed"]
    reference = bundle["naive_predictions"]
    single_target = len(reference) == 1
    with _figure((12, 5.5)) as figure:
        axes = figure.subplots()
        actual_marker = {"marker": "s", "markersize": 6} if single_target else {}
        axes.plot(reference["target_date"], reference["actual_close"], color="#171717", linewidth=2,
                  label="Actual", **actual_marker)
        for position, group in enumerate(GROUPS):
            frame = bundle["predictions"][group, seed]
            marker = {"marker": ("o", "^", "v")[position], "markersize": 5 + 2 * position} if single_target else {}
            axes.plot(frame["target_date"], frame["predicted_close"], color=COLORS[group],
                      linewidth=1.4, label=f"{group} (seed {seed})", **marker)
        naive_marker = {"marker": "D", "markersize": 11, "markerfacecolor": "none"} if single_target else {}
        axes.plot(reference["target_date"], reference["predicted_close"], color=COLORS["Naive"],
                  linewidth=1.3, linestyle="--", label="Naive (previous close)", **naive_marker)
        figure.suptitle(f"{bundle['ticker']} | Adjusted close forecasts", fontsize=14)
        axes.set_title(f"Test: {period} | Fixed LSTM seed {seed}", fontsize=10, loc="left", pad=12)
        axes.set_ylabel(f"Adjusted close ({bundle['price_unit']})")
        axes.set_xlabel("Target trading date")
        locator = AutoDateLocator(minticks=4, maxticks=9)
        axes.xaxis.set_major_locator(locator)
        axes.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        if single_target:
            day = reference["target_date"].iloc[0]
            axes.set_xlim(day - timedelta(days=1), day + timedelta(days=1))
            axes.set_xticks([day])
            axes.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        axes.ticklabel_format(axis="y", style="plain", useOffset=False)
        axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)
        _axes_style(axes)
        _save(figure, path)


def _error_bars(axes, x, mean, std, color):
    axes.bar(x, mean, width=0.6, color=color, alpha=0.9)
    if std is not None:
        axes.errorbar(x, mean, yerr=std, fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)


def _errors(bundle, path, period, seed_label):
    rows = {row["group"]: row for row in bundle["summary"]}
    groups = (*GROUPS, "Naive")
    with _figure((10.5, 4.8)) as figure:
        axes = figure.subplots(1, 2)
        figure.suptitle(f"{bundle['ticker']} | Test price errors", fontsize=14)
        for axis, metric in zip(axes, ("mae", "rmse")):
            for position, group in enumerate(groups):
                row = rows[group]
                _error_bars(axis, position, row[f"{metric}_mean"], row[f"{metric}_std"], COLORS[group])
            axis.set_xticks(range(len(groups)), groups)
            axis.set_ylabel(f"{metric.upper()} ({bundle['price_unit']}, adjusted close)")
            axis.set_title(metric.upper(), fontsize=11)
            _axes_style(axis)
        figure.supxlabel(f"Test: {period}\n{seed_label} | Mean +/- sample SD; no SD for Naive or a single seed", fontsize=9)
        _save(figure, path)


def _direction(bundle, path, period, seed_label):
    rows = {row["group"]: row for row in bundle["summary"]}
    with _figure((8.5, 4.8)) as figure:
        axes = figure.subplots()
        upper = 100.0
        lower = 0.0
        for position, group in enumerate(GROUPS):
            row = rows[group]
            mean = 100 * row["directional_accuracy_mean"]
            std = row["directional_accuracy_std"]
            std = None if std is None else 100 * std
            _error_bars(axes, position, mean, std, COLORS[group])
            upper = max(upper, mean + (std or 0))
            lower = min(lower, mean - (std or 0))
        axes.text(3, 0.04, "N/A", transform=axes.get_xaxis_transform(), ha="center", color=COLORS["Naive"], fontsize=12)
        axes.set_xticks(range(4), (*GROUPS, "Naive"))
        axes.set_xlim(-0.6, 3.6)
        axes.set_ylim(lower - (3 if lower < 0 else 0), upper + 5)
        axes.set_ylabel("Three-way direction accuracy (%)")
        figure.suptitle(f"{bundle['ticker']} | Next-session direction", fontsize=14)
        axes.set_title(f"Test: {period}\n{seed_label} | Mean +/- sample SD", fontsize=10, loc="left", pad=12)
        figure.supxlabel("Up / down / flat relative to previous close | Naive direction: N/A", fontsize=9)
        _axes_style(axes)
        _save(figure, path)


def _loss(bundle, path, seed_label):
    from matplotlib.ticker import FixedLocator

    seed_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    with _figure((14, 4.6)) as figure:
        axes = figure.subplots(1, 3)
        figure.suptitle(f"{bundle['ticker']} | Training and validation loss\n{seed_label}", fontsize=13)
        for axis, group in zip(axes, GROUPS):
            last_epoch = 1
            for position, seed in enumerate(bundle["seeds"]):
                history = bundle["histories"][group, seed]
                color = seed_colors[position % len(seed_colors)]
                single_epoch = len(history) == 1
                train_marker = {"marker": "o", "markersize": 5} if single_epoch else {}
                validation_marker = {"marker": "D", "markersize": 7, "markerfacecolor": "none"} if single_epoch else {}
                axis.plot(history["epoch"], history["loss"], color=color, linestyle="-",
                          label=f"seed {seed}: train", **train_marker)
                axis.plot(history["epoch"], history["val_loss"], color=color, linestyle="--",
                          label=f"seed {seed}: validation", **validation_marker)
                last_epoch = max(last_epoch, int(history["epoch"].max()))
            axis.set_title(f"Model {group}", fontsize=11)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("MSE (scaled Close)")
            ticks = sorted({1 + round(i * (last_epoch - 1) / max(1, min(5, last_epoch - 1)))
                            for i in range(min(5, last_epoch - 1) + 1)})
            axis.xaxis.set_major_locator(FixedLocator(ticks))
            axis.set_xlim((1, last_epoch) if last_epoch > 1 else (0.75, 1.25))
            _axes_style(axis)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
        _save(figure, path)


def render_figures(bundle, output_dir):
    """Save four PNGs from a validated report bundle and return their paths.

    The bundle supplies ticker, price_unit, seeds, plot_seed=42, predictions
    keyed by (A/B/C, seed), naive_predictions, summary and histories. Frames
    use datetime target_date, original-unit close columns, and epoch/loss/
    val_loss history columns. The renderer performs no model or source I/O.
    It does not modify inputs, rcParams or an existing pyplot figure manager.
    Existing output PNGs are never overwritten.
    """
    from matplotlib import rc_context

    if bundle["plot_seed"] != 42 or 42 not in bundle["seeds"]:
        raise ValueError("Price figures require the fixed plot seed 42")
    dates = bundle["naive_predictions"]["target_date"]
    if dates.empty:
        raise ValueError("Figure data must contain test target dates")
    for group in GROUPS:
        if not bundle["predictions"][group, 42]["target_date"].equals(dates):
            raise ValueError("Price figures require identical test target dates for all groups")
    period = f"{dates.iloc[0]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}"
    seed_label = "All seeds: " + ", ".join(str(seed) for seed in bundle["seeds"])
    output_dir = Path(output_dir)
    paths = {key: output_dir / f"{key}.png" for key in ("prices", "errors", "direction", "loss")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Report figure files already exist; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    with rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.unicode_minus": False}):
        _prices(bundle, paths["prices"], period)
        _errors(bundle, paths["errors"], period, seed_label)
        _direction(bundle, paths["direction"], period, seed_label)
        _loss(bundle, paths["loss"], seed_label)
    return paths
