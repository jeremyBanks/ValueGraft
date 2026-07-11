import json

import pytest

from coherent_state_store import (
    ArtifactError,
    checkpoint_path,
    promote_checkpoint,
    read_checkpoint,
    save_render,
)


def test_render_checkpoint_is_atomic_and_resume_stable(tmp_path):
    fp = {"model": "m", "revision": "r"}
    conv = {"id": "c10", "messages": [{"role": "system", "content": "x"}]}
    path = checkpoint_path(tmp_path, 1, "c10")
    first = save_render(path, fingerprint=fp, order_position=1,
                        conversation=conv, reply_records=[])
    assert read_checkpoint(path, fp) == first
    assert save_render(path, fingerprint=fp, order_position=1,
                       conversation=conv, reply_records=[]) == first
    assert not list(tmp_path.glob(".*.tmp"))


def test_resume_rejects_changed_render_or_fingerprint(tmp_path):
    fp = {"model": "m"}
    path = checkpoint_path(tmp_path, 1, "c10")
    conv = {"id": "c10", "messages": []}
    save_render(path, fingerprint=fp, order_position=1,
                conversation=conv, reply_records=[])
    with pytest.raises(ArtifactError, match="fingerprint"):
        read_checkpoint(path, {"model": "different"})
    with pytest.raises(ArtifactError, match="changed render"):
        save_render(path, fingerprint=fp, order_position=1,
                    conversation={"id": "c10", "messages": [1]},
                    reply_records=[])


def test_stage_promotion_is_additive_and_no_regression(tmp_path):
    fp = {"model": "m"}
    path = checkpoint_path(tmp_path, 1, "c10")
    rendered = save_render(
        path, fingerprint=fp, order_position=1,
        conversation={"id": "c10", "messages": []}, reply_records=[])
    captured = promote_checkpoint(path, rendered, {"summary": {"ids": [1]}},
                                  "captured")
    assert read_checkpoint(path, fp, "captured")["summary"]["ids"] == [1]
    with pytest.raises(ArtifactError, match="regression"):
        promote_checkpoint(path, captured, {}, "rendered")
    with pytest.raises(ArtifactError, match="overwrite"):
        promote_checkpoint(path, captured, {"summary": {"ids": [2]}},
                           "scored")
