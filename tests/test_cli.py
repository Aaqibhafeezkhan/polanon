"""The `python -m polanon` entry point, which is the library's shop window."""

import io
import json

import pytest

from polanon import __version__
from polanon.__main__ import main

TEXT = "Jan Kowalski, PESEL 50810112354, mail jan@example.com"


def test_arguments_are_masked_on_stdout_and_the_mapping_goes_to_stderr(capsys):
    assert main([TEXT]) == 0
    captured = capsys.readouterr()
    assert "<PESEL_1>" in captured.out
    assert "50810112354" not in captured.out
    assert json.loads(captured.err)["<PESEL_1>"] == "50810112354"


def test_several_arguments_are_joined_into_one_text(capsys):
    assert main(["PESEL", "50810112354"]) == 0
    assert "<PESEL_1>" in capsys.readouterr().out


def test_text_is_read_from_standard_input_when_no_argument_is_given(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(TEXT))
    assert main([]) == 0
    assert "<EMAIL_1>" in capsys.readouterr().out


def test_empty_input_explains_itself_and_fails(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("   \n"))
    assert main([]) == 2
    assert "nothing to mask" in capsys.readouterr().err


def test_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"polanon {__version__}"
    assert captured.err == ""


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--ner" in capsys.readouterr().out


class PipedInput:
    """Standard input the way a pipe delivers it: bytes, whatever the console's code page."""

    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)


def test_piped_utf8_is_read_as_utf8_whatever_the_console(capsys, monkeypatch):
    """Windows decoded pipes with the ANSI code page, and garbled names went out unmasked."""
    monkeypatch.setattr("sys.stdin", PipedInput("Umowę podpisał Łukasz Wiśniewski.".encode()))
    assert main([]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Umowę podpisał <OSOBA_1>."
    assert json.loads(captured.err) == {"<OSOBA_1>": "Łukasz Wiśniewski"}


def test_a_cp1250_file_from_older_software_is_read_too(capsys, monkeypatch):
    text = "Umowę podpisał Łukasz Wiśniewski."
    monkeypatch.setattr("sys.stdin", PipedInput(text.encode("cp1250")))
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == "Umowę podpisał <OSOBA_1>."


def test_ner_without_spacy_says_what_to_install(capsys, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert main(["--ner", "Jan Kowalski"]) == 3
    assert "polanon[ner]" in capsys.readouterr().err


def test_the_command_runs_with_docstrings_stripped():
    """python -OO removes docstrings, and the description used to be read from one."""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-OO", "-m", "polanon", "PESEL 50810112354"],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert b"<PESEL_1>" in completed.stdout


def test_output_is_utf8_whatever_the_console_encoding():
    """With a cp1250 console the mapping came out in cp1250 and was no longer valid UTF-8 JSON."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONIOENCODING": "cp1250"}
    env.pop("PYTHONUTF8", None)
    completed = subprocess.run(
        [sys.executable, "-m", "polanon", "Umowę podpisał Łukasz Wiśniewski."],
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0
    assert completed.stdout.decode("utf-8").strip() == "Umowę podpisał <OSOBA_1>."
    assert json.loads(completed.stderr.decode("utf-8")) == {"<OSOBA_1>": "Łukasz Wiśniewski"}


def test_a_utf8_file_is_masked_with_file_option(capsys, tmp_path):
    path = tmp_path / "umowa.txt"
    path.write_bytes("Umowę podpisał Łukasz Wiśniewski.".encode())
    assert main(["--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Umowę podpisał <OSOBA_1>."
    assert json.loads(captured.err) == {"<OSOBA_1>": "Łukasz Wiśniewski"}


def test_a_cp1250_file_is_masked_with_file_option(capsys, tmp_path):
    path = tmp_path / "umowa.txt"
    path.write_bytes("Umowę podpisał Łukasz Wiśniewski.".encode("cp1250"))
    assert main(["--file", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "Umowę podpisał <OSOBA_1>."


def test_a_missing_file_is_reported_and_fails(capsys, tmp_path):
    assert main(["--file", str(tmp_path / "missing.txt")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_file_option_refuses_text_arguments(capsys, tmp_path):
    path = tmp_path / "umowa.txt"
    path.write_text("Jan Kowalski", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["--file", str(path), "Jan Kowalski"])
    assert exit_info.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


SIGNED = "Umowę podpisał Łukasz Wiśniewski, PESEL 50810112354."
SIGNED_MASKED = "Umowę podpisał <OSOBA_1>, PESEL <PESEL_1>."


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(SIGNED.encode("utf-16"), id="utf-16 with a byte order mark"),
        pytest.param(SIGNED.encode("utf-16-le"), id="utf-16-le without one"),
        pytest.param(SIGNED.encode("utf-16-be"), id="utf-16-be without one"),
        pytest.param(SIGNED.encode("utf-32"), id="utf-32 with a byte order mark"),
        pytest.param(SIGNED.encode("utf-8-sig"), id="utf-8 with a byte order mark"),
    ],
)
def test_unicode_files_are_decoded_by_their_bytes(data, capsys, tmp_path):
    """Windows PowerShell 5.1 writes UTF-16 with `>`, and its NULs used to hide every value."""
    path = tmp_path / "umowa.txt"
    path.write_bytes(data)
    assert main(["--file", str(path)]) == 0
    assert capsys.readouterr().out.strip() == SIGNED_MASKED


def test_piped_utf16_is_decoded_too(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", PipedInput(SIGNED.encode("utf-16")))
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == SIGNED_MASKED


def test_one_cp1250_line_does_not_garble_a_utf8_file(capsys, tmp_path):
    """Decoding the whole file as cp1250 turned its UTF-8 names into mojibake that went out."""
    path = tmp_path / "umowa.txt"
    path.write_bytes((SIGNED + "\n").encode() + "Kwota: 100 zł.\n".encode("cp1250"))
    assert main(["--file", str(path)]) == 0
    assert capsys.readouterr().out.rstrip("\n").splitlines() == [SIGNED_MASKED, "Kwota: 100 zł."]


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        pytest.param(b"\x00\x01\x02 binary \x00\xff\x00", "NUL", id="binary"),
        pytest.param(b"Umowa \x81 podpisana", "neither UTF-8 nor cp1250", id="undefined in cp1250"),
        pytest.param(b"\xff\xfeU\x00m\x00o", "UTF-16", id="truncated utf-16"),
    ],
)
def test_input_that_cannot_be_decoded_for_certain_is_refused(data, reason, capsys, tmp_path):
    path = tmp_path / "umowa.txt"
    path.write_bytes(data)
    assert main(["--file", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert reason in captured.err


def test_crlf_line_endings_come_out_as_they_went_in(tmp_path):
    """Windows text mode turned CRLF into CR CR LF, double-spacing the masked document."""
    import subprocess
    import sys

    (tmp_path / "umowa.txt").write_bytes(b"PESEL 50810112354\r\nKoniec\r\n")
    completed = subprocess.run(
        [sys.executable, "-m", "polanon", "--file", "umowa.txt"],
        capture_output=True,
        check=False,
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == b"PESEL <PESEL_1>\r\nKoniec\r\n\n"


def test_mapping_option_writes_the_mapping_to_a_file(capsys, tmp_path):
    mapping = tmp_path / "mapping.json"
    assert main(["--mapping", str(mapping), SIGNED]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == SIGNED_MASKED
    assert captured.err == ""
    assert json.loads(mapping.read_text(encoding="utf-8")) == {
        "<OSOBA_1>": "Łukasz Wiśniewski",
        "<PESEL_1>": "50810112354",
    }


def test_an_unwritable_mapping_fails_before_any_masked_text(capsys, tmp_path):
    assert main(["--mapping", str(tmp_path), SIGNED]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot write" in captured.err


def test_a_directory_given_as_the_file_is_reported_with_a_reason(capsys, tmp_path):
    assert main(["--file", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "cannot read" in err
    assert "None" not in err


def test_no_standard_input_at_all_explains_itself(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", None)
    assert main([]) == 2
    assert "nothing to mask" in capsys.readouterr().err
