import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_legacy_source_split import analyze, cluster_bootstrap


def test_cluster_bootstrap_preserves_plant_weighted_point_estimate() -> None:
    rows = [
        {"conversation_id": "c07", "raw_EB": 1.0},
        {"conversation_id": "c07", "raw_EB": 3.0},
        {"conversation_id": "c08", "raw_EB": 8.0},
    ]
    result = cluster_bootstrap(rows, n_boot=100, seed=0)
    assert result["mean_raw_EB_nats_per_token"] == pytest.approx(4.0)
    assert result["n_plants"] == 3
    assert result["n_conversations"] == 2


def test_analyze_rejects_incomplete_conversation_set(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.json"
    path.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "conversation_id": "c07",
                        "raw_EB": 0.1,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="conversation set mismatch"):
        analyze({"fixture": path}, n_boot=10, seed=0)
