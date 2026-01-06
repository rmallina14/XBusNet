import argparse
import json
import inspect
import importlib
import os
import sys
from collections import Counter
from shutil import copy
from subprocess import run, PIPE
from os.path import (
    join,
    dirname,
    realpath,
    expanduser,
    isfile,
    isdir,
    basename,
)

import torch
import yaml


# ----------------------------------------------------------------------
# Lightweight logger
# ----------------------------------------------------------------------


class Logger(object):
    """
    Minimal logger shim.

    Any attribute access (info, warning, hint, ...) just forwards to print().
    This keeps the rest of the code simple while giving a log-like interface.
    """

    def __getattr__(self, name):
        return print


log = Logger()


# ----------------------------------------------------------------------
# Config loading helpers
# ----------------------------------------------------------------------


class AttributeDict(dict):
    """
    dict → object-like config.

    - Supports attribute access: cfg.key
    - Tracks how often each key is accessed (via a Counter).
      This allows checking for unused keys after wiring configs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__["counter"] = Counter()

    # --- access & mutation ---

    def __getitem__(self, key):
        self.__dict__["counter"][key] += 1
        return super().__getitem__(key)

    def __getattr__(self, key):
        # Called when normal attribute lookup fails
        self.__dict__["counter"][key] += 1
        return super().get(key)

    def __setattr__(self, key, value):
        # store all config-style fields in the dict
        return super().__setitem__(key, value)

    def __delattr__(self, key):
        return super().__delitem__(key)

    # --- diagnostics ---

    def unused_keys(self, exceptions=()):
        """
        Returns a list of keys that were never accessed,
        excluding any in `exceptions`.
        """
        return [
            k for k in super().keys()
            if self.__dict__["counter"][k] == 0 and k not in exceptions
        ]

    def assume_no_unused_keys(self, exceptions=()):
        """
        Convenience check to call near the end of wiring configs.
        """
        unused = self.unused_keys(exceptions=exceptions)
        if unused:
            log.warning("Unused config keys:", unused)


def _load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def training_config_from_cli_args():
    """
    Parse --config <yaml> from the command line and return
    a merged training configuration as an AttributeDict.

    YAML structure expected:
      configuration: {...base defaults...}
      individual_configurations:
        - {...override for exp 0...}
        - {...override for exp 1...}
      (we only use the first individual_configurations entry here)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args, _ = parser.parse_known_args()

    yaml_cfg = _load_yaml_config(args.config)

    base_cfg = yaml_cfg.get("configuration", {})
    indiv_list = yaml_cfg.get("individual_configurations", [])

    if isinstance(indiv_list, list) and len(indiv_list) > 0:
        merged = {**base_cfg, **indiv_list[0]}
    else:
        merged = dict(base_cfg)

    return AttributeDict(merged)


def score_config_from_cli_args():
    """
    Scoring / test-time config helper.

    NOTE: This is kept for compatibility with the original code
    but is typically unused in the current X-BusNet training setup.
    """
    experiment_name = "busi.yaml"
    experiment_id = 0

    yaml_cfg = _load_yaml_config(f"experiments/{experiment_name}")

    common = yaml_cfg.get("test_configuration_common", {})
    test_cfg = yaml_cfg.get("test_configuration", {})

    # test_configuration may be a list or a dict
    if isinstance(test_cfg, list) and len(test_cfg) > 0:
        test_cfg = test_cfg[0]

    merged = {**common, **test_cfg}

    indiv_cfgs = yaml_cfg.get("individual_configurations", [])
    if (
        isinstance(indiv_cfgs, list)
        and len(indiv_cfgs) > experiment_id
        and "test_configuration" in indiv_cfgs[experiment_id]
    ):
        merged.update(indiv_cfgs[experiment_id]["test_configuration"])

    train_checkpoint_id = indiv_cfgs[experiment_id]["name"]
    return AttributeDict(merged), train_checkpoint_id


# ----------------------------------------------------------------------
# Generic dynamic utilities
# ----------------------------------------------------------------------


def get_attribute(name):
    """
    Resolve a fully-qualified object name like 'module.submodule.ClassName'
    into the corresponding Python object.
    """
    if name is None:
        raise ValueError("The provided attribute is None")

    parts = name.split(".")
    if len(parts) < 2:
        raise ValueError(f"Expected 'module.object', got: {name}")

    module_name = ".".join(parts[:-1])
    obj_name = parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def filter_args(input_args, default_args):
    """
    Split a config-like mapping into:

      (updated_args, used_args, unused_args)

    where:
      - updated_args: dict with every default parameter present,
                      overridden by input_args if provided
      - used_args:    subset of input_args that matched default_args keys
      - unused_args:  leftover input_args that didn't map to default_args
    """
    # normalize input_args into a simple dict-like
    if isinstance(input_args, AttributeDict):
        src = dict(input_args)
    elif hasattr(input_args, "items"):
        src = dict(input_args.items())
    else:
        src = dict(input_args)

    updated = {k: src[k] if k in src else v for k, v in default_args.items()}
    used = {k: v for k, v in src.items() if k in default_args}
    unused = {k: v for k, v in src.items() if k not in default_args}

    return AttributeDict(updated), AttributeDict(used), AttributeDict(unused)


# ----------------------------------------------------------------------
# Dataset repository helper (kept for compatibility)
# ----------------------------------------------------------------------


def extract_archive(filename, target_folder=None, noarchive_ok=False):
    """
    Extract tar/zip archive to a target folder using system tools.
    If noarchive_ok=True and the file extension is unknown, silently return.
    """
    if filename.endswith((".tgz", ".tar")):
        cmd = ["tar", "-xf", filename]
        if target_folder:
            cmd += ["-C", target_folder]
    elif filename.endswith(".tar.gz"):
        cmd = ["tar", "-xzf", filename]
        if target_folder:
            cmd += ["-C", target_folder]
    elif filename.endswith(".zip"):
        cmd = ["unzip", filename]
        if target_folder:
            cmd += ["-d", target_folder]
    else:
        if noarchive_ok:
            return
        raise ValueError(f"Unsupported file format: {filename}")

    log.hint(" ".join(cmd))
    result = run(cmd, stdout=PIPE, stderr=PIPE)
    if result.returncode != 0:
        print(result.stdout, result.stderr)


def get_from_repository(
    local_name,
    repo_files,
    integrity_check=None,
    repo_dir="~/dataset_repository",
    local_dir="./third_party/thyroid/",
):
    """
    Copy and extract dataset archives from a central repository into a local data folder.

    - local_name is unused but kept for API compatibility.
    - repo_files can be:
        ["archive.tgz", ("remote/path.zip", "local/path.zip"), ...]
    """
    local_dir = realpath(join(expanduser(local_dir), "data"))
    os.makedirs(local_dir, exist_ok=True)

    dataset_exists = isdir(local_dir)

    # Optional custom integrity check
    if integrity_check is not None:
        try:
            ok = integrity_check(local_dir)
        except Exception:
            ok = False

        if ok:
            log.hint("Passed custom integrity check")
        else:
            log.hint("Custom integrity check failed")

        dataset_exists = dataset_exists and ok

    if dataset_exists:
        return

    repo_dir = realpath(expanduser(repo_dir))

    for entry in repo_files:
        if isinstance(entry, str):
            origin_rel, target_rel = entry, entry
        else:
            origin_rel, target_rel = entry

        archive_origin = join(repo_dir, origin_rel)

        # where the archive will be placed locally before extraction
        archive_target = join(local_dir, dirname(target_rel), basename(origin_rel))
        extract_target = join(local_dir, dirname(target_rel))

        os.makedirs(dirname(archive_target), exist_ok=True)

        if isfile(archive_target):
            if os.path.getsize(archive_target) != os.path.getsize(archive_origin):
                log.hint(
                    f"file exists but filesize differs: "
                    f"target {os.path.getsize(archive_target)} vs. origin {os.path.getsize(archive_origin)}"
                )
                copy(archive_origin, archive_target)
        else:
            log.hint(f"copy: {archive_origin} → {archive_target}")
            copy(archive_origin, archive_target)

        extract_archive(archive_target, extract_target, noarchive_ok=True)

        if isfile(archive_target):
            os.remove(archive_target)


# ----------------------------------------------------------------------
# Model loading and training logger
# ----------------------------------------------------------------------


def load_model(
    checkpoint_id,
    weights_file=None,
    strict=True,
    model_args="from_config",
    with_config=False,
):
    """
    Reconstruct a model from a training run in logs_busi/<checkpoint_id>.

    Expects:
      logs_busi/<checkpoint_id>/config.json
      logs_busi/<checkpoint_id>/weights.pth  (or a custom weights_file)

    Args:
        checkpoint_id:  name of the subfolder under logs_busi/
        weights_file:   optional override for the weights filename
        strict:         passed to model.load_state_dict
        model_args:     'from_config' or a dict of constructor kwargs
        with_config:    if True, returns (model, cfg) instead of just model
    """
    base_dir = "logs_busi"
    cfg_path = join(base_dir, checkpoint_id, "config.json")

    if not isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.json at: {cfg_path}")

    config = json.load(open(cfg_path, "r"))

    if model_args != "from_config" and not isinstance(model_args, dict):
        raise ValueError('model_args must be "from_config" or a dict of kwargs')

    # Resolve model class
    model_cls = get_attribute(config["model"])

    # Derive constructor args
    if model_args == "from_config":
        _, ctor_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
    else:
        ctor_args = model_args

    model = model_cls(**ctor_args)

    # Decide which weights file to use
    if weights_file is None:
        weights_path = realpath(join(base_dir, checkpoint_id, "weights.pth"))
    else:
        weights_path = realpath(join(base_dir, checkpoint_id, weights_file))

    if not isfile(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

    weights = torch.load(weights_path, map_location="cpu")

    # sanity check for NaNs
    for name, w in weights.items():
        if torch.any(torch.isnan(w)):
            raise ValueError(f"Weights contain NaNs in parameter: {name}")

    model.load_state_dict(weights, strict=strict)

    return (model, config) if with_config else model


class TrainingLogger(object):
    """
    Simple training logger:

    - creates logs_busi/<run_name>/
    - stores config.json
    - provides .iter(...) print logging every N iterations
    - saves model weights via .save_weights()
    """

    def __init__(self, model, log_dir, config=None, *args):
        super().__init__()
        self.model = model
        # keep directory structure compatible with previous runs
        self.base_path = join("logs_busi", log_dir) if log_dir else None

        os.makedirs("logs_busi", exist_ok=True)
        if self.base_path is not None:
            os.makedirs(self.base_path, exist_ok=True)

        if config is not None and self.base_path is not None:
            # AttributeDict inherits from dict → json-serializable
            cfg_path = join(self.base_path, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(config, f)

    def iter(self, i, **kwargs):
        """
        Called every iteration from the training loop.

        By default, prints a progress line every 100 iterations
        if a 'loss' keyword is provided.
        """
        if i % 100 == 0 and "loss" in kwargs:
            loss_val = float(kwargs["loss"])
            print(f"iteration {i:6d} | loss = {loss_val:.4f}")

    def save_weights(self, only_trainable=False, weight_file="weights.pth"):
        """
        Save model.state_dict() into logs_busi/<run_name>/<weight_file>.
        """
        if self.model is None:
            raise AttributeError("You need to provide a model to save weights.")

        if self.base_path is None:
            raise RuntimeError("TrainingLogger.base_path is not set.")

        weights_path = join(self.base_path, weight_file)
        state = self.model.state_dict()

        if only_trainable:
            state = {n: w for n, w in state.items() if w.requires_grad}

        torch.save(state, weights_path)
        log.info(f"Saved weights to {weights_path}")

    # Context manager interface
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing special to clean up
        pass
