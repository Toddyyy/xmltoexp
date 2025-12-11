import matplotlib.pyplot as plt

from score_explainer import FullPerformanceToken, ScoreExplainer
from tokenizer.base_performance_tokenizer import PerformanceMetadata
from tokenizer.base_score_tokenizer import ScoreMetadata
from tokenizer.paired_tokenizer import PairedTokenizer


def test_score_explainer_tempo():
    paired_tokenizer = PairedTokenizer(
        xml_score_path="./example_data/nasap/Chopin/Scherzos/20/xml_score.musicxml",
        midi_path="./example_data/nasap/Chopin/Scherzos/20/Wong03.mid",
        match_path="./example_data/nasap/Chopin/Scherzos/20/Wong03.match",
    )

    (
        score_metadata,
        midi_metadata,
        score_tokens,
        midi_tokens,
        score_expressions,
        midi_controls,
    ) = paired_tokenizer.tokenize()

    se = ScoreExplainer(
        score_tokens=score_tokens,
        score_expressions=score_expressions,
        performance_tokens=midi_tokens,
        performance_controls=midi_controls,
    )

    full_tokens = se.get_full_tokens()
    from rich import print

    print("1st Full Performance Tokens:")
    print(full_tokens[0])

    plot_explained_tokens(
        full_tokens=full_tokens,
        score_metadata=score_metadata,
        midi_metadata=midi_metadata,
    )


def plot_explained_tokens(
    full_tokens: list[FullPerformanceToken],
    score_metadata: ScoreMetadata,
    midi_metadata: PerformanceMetadata,
):
    xs = [n.score_note_token.position for n in full_tokens]
    tempi = [n.local_tempo for n in full_tokens]
    onset_deviations = [n.onset_deviation_in_beats for n in full_tokens]
    duration_deviations = [n.duration_deviation_in_beats for n in full_tokens]
    sustain_levels = [n.sustain_level for n in full_tokens]
    vel = [n.performance_note_token.velocity for n in full_tokens]

    _fig, axes = plt.subplots(5, 1, figsize=(9, 8))
    axes[0].plot(xs, tempi, "x-", markersize=3)
    axes[0].set_title("Tempi (BPM)")

    axes[1].plot(
        xs,
        onset_deviations,
        "o-",
        markersize=3,
        alpha=0.8,
        color="green",
    )
    axes[1].set_ylim(-1, 1)
    axes[1].set_title("Onset Deviations (Beats)")

    axes[2].plot(
        xs,
        duration_deviations,
        "^-r",
        markersize=3,
        alpha=0.8,
        color="orange",
    )
    axes[2].set_ylim(-1, 1)
    axes[2].set_title("Duration Deviations (Beats)")

    axes[3].plot(xs, vel, "o-", markersize=3, alpha=0.8, color="blue")
    axes[3].set_ylim(0, 128)
    axes[3].set_title("Velocities")

    axes[4].plot(xs, sustain_levels, "o-", markersize=3, alpha=0.8, color="red")
    axes[4].set_ylim(0, 128)
    axes[4].set_title("Sustain Levels")

    plt.suptitle(
        f"{score_metadata.title} \n Composer: {score_metadata.composer} Performer: {midi_metadata.performer}",
        fontsize=15,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_score_explainer_tempo()
