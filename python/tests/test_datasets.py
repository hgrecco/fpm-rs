from __future__ import annotations

import json

import pytest

import fpm_rs as fpm
from fpm_rs.datasets.cli import build_parser, main


def test_registry_lists_an_empty_local_document_and_exposes_configuration(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"registry_version": 1, "datasets": []}), encoding="utf-8"
    )
    cache = tmp_path / "cache"
    registry = fpm.DatasetRegistry(registry_url=str(registry_path), cache_dir=cache)

    assert registry.registry_url == str(registry_path)
    assert registry.cache_dir == cache
    assert registry.list() == []
    assert registry.download_all() == []
    assert registry.clean_all() == 0


def test_registry_errors_remain_typed_dataset_errors(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"registry_version": 1, "datasets": []}), encoding="utf-8"
    )
    registry = fpm.DatasetRegistry(
        registry_url=str(registry_path), cache_dir=tmp_path / "cache"
    )

    with pytest.raises(fpm.DatasetError, match="no entry 'missing'"):
        registry.open("missing")
    with pytest.raises(fpm.DatasetError, match="no entry 'missing'"):
        fpm.open_dataset(
            "missing", registry_url=str(registry_path), cache_dir=tmp_path / "cache"
        )


def test_python_cli_supports_every_command_and_lists_cache_state(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"registry_version": 1, "datasets": []}), encoding="utf-8"
    )
    common = [
        "--registry-url",
        str(registry_path),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]

    assert main([*common, "list"]) == 0
    assert capsys.readouterr().out == "ID\tVERSION\tCACHED\tSIZE_BYTES\tTITLE\n"
    assert main([*common, "download", "--all"]) == 0
    assert main([*common, "clean", "fixture"]) == 0
    assert "removed=false" in capsys.readouterr().out
    assert main([*common, "clean", "--all"]) == 0
    assert "removed=0" in capsys.readouterr().out

    for arguments in [
        ["list"],
        ["download", "fixture"],
        ["download", "--all"],
        ["open", "fixture"],
        ["clean", "fixture"],
        ["clean", "--all"],
    ]:
        build_parser().parse_args(arguments)
