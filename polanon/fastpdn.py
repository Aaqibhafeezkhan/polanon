"""Export FastPDN, CLARIN-PL's Polish named-entity model, for the arbiter: once, on your machine.

    python -m polanon.fastpdn                  # to ~/.cache/polanon/fastpdn
    python -m polanon.fastpdn --dir /models/fastpdn

The arbiter (``default_guards(arbiter=True)``) runs FastPDN with onnxruntime alone. This command
makes what it reads: it downloads the model from Hugging Face at the revision the arbiter was
fitted with, writes the network as ONNX, the tokenizer and the labels, checks that the ONNX model
gives what the original gives on a few sentences, and writes FastPDN's licence notice next to it.
It needs torch, transformers and onnx for this one run, and nothing of them afterwards:

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers onnx

FastPDN (https://huggingface.co/clarin-pl/FastPDN) is by CLARIN-PL and licensed CC-BY-4.0
(https://creativecommons.org/licenses/by/4.0/). polanon does not ship it; the export is a copy on
your machine, converted to another format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from polanon.guards._fastpdn import (
    CONFIG_FILE,
    DEFAULT_DIRECTORY,
    FORMAT,
    MODEL_FILE,
    TOKENIZER_FILE,
)

#: The model, and the revision of it the arbiter's weights were fitted with.
SOURCE = "clarin-pl/FastPDN"
REVISION = "0fcce46a197decff51f11a327a6509b9a5e4c95f"
#: How far the chunks of a long text overlap, in tokens, as in the pipeline the arbiter
#: was fitted with.
STRIDE = 128

NOTICE = """FastPDN, exported for polanon's arbiter.

Source: https://huggingface.co/clarin-pl/FastPDN (revision {revision})
Authors: CLARIN-PL (Grupa Wieszcze CLARIN-PL)
Licence: Creative Commons Attribution 4.0 International (CC-BY-4.0),
https://creativecommons.org/licenses/by/4.0/

Changes: the network was converted from PyTorch to ONNX by `python -m polanon.fastpdn`; its
weights and its tokenizer are unchanged.
"""

#: Sentences the exported model must read as the original does.
CHECKS = (
    "Nazywam się Jan Kowalski i mieszkam we Wrocławiu przy ul. Długiej 12.",
    "Faktura dla Studio Kwadrat, kontakt: Ewelina Sadowska, Gdańsk.",
)


def export(directory: Path = DEFAULT_DIRECTORY, revision: str = REVISION) -> Path:
    """Download FastPDN and write what the arbiter reads to `directory`.

    Raises:
        RuntimeError: when torch, transformers or onnx is missing, or the ONNX model reads one
            of the check sentences otherwise than the original.
    """
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as exc:  # pragma: no cover - only without the tools
        raise RuntimeError(
            "the export needs torch, transformers and onnx: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu; "
            "pip install transformers onnx"
        ) from exc
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, revision=revision)
    model = AutoModelForTokenClassification.from_pretrained(SOURCE, revision=revision).eval()

    def start(self: Any) -> None:
        torch.nn.Module.__init__(self)
        self.model = model

    def forward(self: Any, input_ids: Any, attention_mask: Any) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    # The network with its logits as the one output. Built with type() rather than a class
    # statement: where torch is not installed, as in CI, mypy cannot see what it would subclass.
    logits_only = type("Logits", (torch.nn.Module,), {"__init__": start, "forward": forward})

    # What the original reads in the check sentences, before the export: tracing the model for
    # ONNX leaves it reading some of them otherwise.
    original = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        stride=STRIDE,
    )
    wanted = [
        [(int(e["start"]), int(e["end"]), str(e["entity_group"])) for e in original(s)]
        for s in CHECKS
    ]

    sample = tokenizer(CHECKS[0], return_tensors="pt")
    axes = {name: {0: "batch", 1: "tokens"} for name in ("input_ids", "attention_mask", "logits")}
    torch.onnx.export(
        logits_only(),
        (sample["input_ids"], sample["attention_mask"]),
        str(directory / MODEL_FILE),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )
    tokenizer.backend_tokenizer.save(str(directory / TOKENIZER_FILE))
    prefix = getattr(tokenizer.backend_tokenizer.model, "continuing_subword_prefix", None)
    config = {
        "format": FORMAT,
        "source": SOURCE,
        "revision": revision,
        "labels": [model.config.id2label[index] for index in range(len(model.config.id2label))],
        "max_length": int(tokenizer.model_max_length),
        "stride": STRIDE,
        "unknown_id": int(tokenizer.unk_token_id),
        "pad_id": int(tokenizer.pad_token_id),
        "pad_token": str(tokenizer.pad_token),
        "subword_prefix": prefix or None,
    }
    (directory / CONFIG_FILE).write_text(
        json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (directory / "NOTICE.txt").write_text(NOTICE.format(revision=revision), encoding="utf-8")

    from polanon.guards._fastpdn import FastPDN

    exported = FastPDN(directory)
    for sentence, want in zip(CHECKS, wanted, strict=True):
        got = [(e.start, e.end, e.group) for e in exported.entities(sentence)]
        if want != got:
            raise RuntimeError(
                f"the exported model reads a check sentence otherwise: {got} instead of {want}"
            )
    return directory


def main(argv: list[str] | None = None) -> int:
    """Export FastPDN from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dir", type=Path, default=DEFAULT_DIRECTORY, help="where to write the model"
    )
    parser.add_argument("--revision", default=REVISION, help="the model's revision on Hugging Face")
    args = parser.parse_args(argv)
    try:
        where = export(args.dir, args.revision)
    except RuntimeError as exc:
        print(f"polanon.fastpdn: {exc}", file=sys.stderr)
        return 1
    print(f"FastPDN exported to {where}; default_guards(arbiter=True) reads it from there.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
