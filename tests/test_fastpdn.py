"""FastPDN's runtime: the same entities as the Hugging Face pipeline the arbiter was fitted with.

These tests need neither onnxruntime nor the model. The pipeline's own steps are checked on made-up
tokens and scores; reading a whole text, with a tokenizer and a session that stand in for the
real ones. That the runtime returns what the pipeline returns on real texts was checked on 3,140
of the benchmarks' texts, long ones among them, with no difference.
"""

import json

import pytest

from polanon import fastpdn as export_command
from polanon.guards import _fastpdn as fp
from polanon.guards._fastpdn import Entity, FastPDN, entities_of, missing_files, overlapping, pieces

LABELS = ["O", "B-nam_liv_person", "I-nam_liv_person", "B-nam_loc_gpe_city", "I-nam_loc_gpe_city"]


def one_hot(label, score=0.9):
    row = [0.01] * len(LABELS)
    row[LABELS.index(label)] = score
    return row


def test_special_tokens_are_left_out_and_a_piece_after_no_space_goes_on_its_word():
    text = "Jan Kowalski"
    found = pieces(
        text,
        ids=[0, 10, 11, 12, 2],
        offsets=[(0, 0), (0, 3), (4, 8), (8, 12), (0, 0)],
        special=[1, 0, 0, 0, 1],
        scores=[None, "a", "b", "c", None],
        token_of=lambda _id: "",
        unknown_id=3,
        subword_prefix=None,
    )
    assert [(p.start, p.end, p.is_subword) for p in found] == [
        (0, 3, False),
        (4, 8, False),
        (8, 12, True),
    ]


def test_an_unknown_token_starts_a_word_and_a_prefix_marks_the_pieces_where_there_is_one():
    text = "ab"
    unknown = pieces(
        text,
        [3, 3],
        [(0, 1), (1, 2)],
        [0, 0],
        ["x", "y"],
        token_of=str,
        unknown_id=3,
        subword_prefix=None,
    )
    assert [p.is_subword for p in unknown] == [False, False]
    marked = pieces(
        "Kowalski",
        [1, 2],
        [(0, 4), (4, 8)],
        [0, 0],
        ["x", "y"],
        token_of={1: "Kowa", 2: "##lski"}.get,
        unknown_id=3,
        subword_prefix="##",
    )
    assert [p.is_subword for p in marked] == [False, True]


def piece(start, end, label, subword=False, score=0.9):
    return fp._Piece(start, end, one_hot(label, score), subword)


def test_a_word_takes_its_first_piece_s_label_and_neighbours_of_one_category_join():
    found = [
        piece(0, 3, "B-nam_liv_person"),
        piece(4, 8, "I-nam_liv_person"),
        piece(8, 12, "B-nam_loc_gpe_city", subword=True),  # a piece inside "Kowalski"
        piece(13, 14, "O"),
        piece(15, 21, "B-nam_loc_gpe_city", score=0.7),
    ]
    assert entities_of(found, LABELS) == [
        Entity(0, 12, "nam_liv_person", 0.9),
        Entity(15, 21, "nam_loc_gpe_city", 0.7),
    ]


def test_two_beginnings_of_one_category_stay_two_entities():
    found = [piece(0, 3, "B-nam_liv_person"), piece(4, 7, "B-nam_liv_person")]
    assert [(e.start, e.end) for e in entities_of(found, LABELS)] == [(0, 3), (4, 7)]


def test_of_what_two_chunks_both_found_the_longer_or_the_surer_stays():
    found = [
        Entity(0, 5, "nam_liv_person", 0.9),
        Entity(2, 9, "nam_liv_person", 0.5),
        Entity(20, 25, "nam_loc_gpe_city", 0.6),
        Entity(20, 25, "nam_loc_gpe_city", 0.8),
        Entity(30, 32, "nam_liv_person", 0.9),
    ]
    assert overlapping(found) == [found[1], found[3], found[4]]
    assert overlapping([]) == []


def test_a_directory_without_the_export_says_which_files_are_missing(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert missing_files(tmp_path) == ["model.onnx", "tokenizer.json"]
    with pytest.raises(
        RuntimeError, match=r"missing model\.onnx, tokenizer\.json.*python -m polanon\.fastpdn"
    ):
        FastPDN(tmp_path)


class Encoding:
    def __init__(self, ids, offsets, special, overflowing=()):
        self.ids = ids
        self.offsets = offsets
        self.special_tokens_mask = special
        self.attention_mask = [1] * len(ids)
        self.overflowing = list(overflowing)


class Session:
    """Returns, for each chunk, the logits its token ids stand for."""

    def __init__(self, logits_of):
        self.logits_of = logits_of
        self.fed = []

    def run(self, names, feed):
        np = pytest.importorskip("numpy")
        ids = [int(i) for i in feed["input_ids"][0]]
        self.fed.append(ids)
        return [np.array([[self.logits_of(i) for i in ids]], dtype=np.float32)]


def reading(encoding, logits_of):
    model = object.__new__(FastPDN)
    model.labels = LABELS
    model.unknown_id = 3
    model.subword_prefix = None
    model._tokenizer = type(
        "Tokenizer",
        (),
        {"encode_batch": lambda self, texts: [encoding], "id_to_token": lambda self, i: ""},
    )()
    model._session = Session(logits_of)
    import threading

    model._lock = threading.Lock()
    return model


def logits(label):
    return [5.0 if name == label else 0.0 for name in LABELS]


def test_a_text_is_read_chunk_by_chunk_and_what_two_chunks_found_is_kept_once():
    pytest.importorskip("numpy")
    text = "Jan Kowalski z Gdyni"
    by_id = {10: "B-nam_liv_person", 11: "I-nam_liv_person", 12: "O", 13: "B-nam_loc_gpe_city"}
    first = Encoding(
        [0, 10, 11, 12, 2], [(0, 0), (0, 3), (4, 12), (13, 14), (0, 0)], [1, 0, 0, 0, 1]
    )
    second = Encoding(
        [0, 11, 12, 13, 2], [(0, 0), (4, 12), (13, 14), (15, 20), (0, 0)], [1, 0, 0, 0, 1]
    )
    first.overflowing = [second]
    model = reading(first, lambda i: logits(by_id.get(i, "O")))
    found = model.entities(text)
    assert [(e.start, e.end, e.group) for e in found] == [
        (0, 12, "nam_liv_person"),
        (15, 20, "nam_loc_gpe_city"),
    ]
    assert all(0.9 < e.score <= 1.0 for e in found)
    assert len(model._session.fed) == 2
    assert model.entities("   ") == []


def test_the_export_command_reports_what_failed_and_where_it_wrote(monkeypatch, tmp_path, capsys):
    def refused(directory, revision):
        raise RuntimeError("the export needs torch")

    monkeypatch.setattr(export_command, "export", refused)
    assert export_command.main(["--dir", str(tmp_path)]) == 1
    assert "the export needs torch" in capsys.readouterr().err
    monkeypatch.setattr(export_command, "export", lambda directory, revision: directory)
    assert export_command.main(["--dir", str(tmp_path)]) == 0
    assert str(tmp_path) in capsys.readouterr().out


def test_the_export_writes_the_attribution_fastpdn_s_licence_asks_for():
    assert "CC-BY-4.0" in export_command.NOTICE
    assert "CLARIN-PL" in export_command.NOTICE
    assert "{revision}" in export_command.NOTICE
    assert json.dumps(export_command.REVISION)
