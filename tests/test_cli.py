import pytest
from sim2gate import __version__
from sim2gate.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and __version__ in capsys.readouterr().out


def test_no_command_prints_help():
    assert main([]) == 0
