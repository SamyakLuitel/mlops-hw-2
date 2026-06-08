"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

# Allow `python scripts/promote.py ...` (script dir, not repo root, is on the
# path by default) to import the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"

# Key metrics surfaced by `show`, in display order.
SHOW_METRICS = ("accuracy_overall", "verdict_rate_leaked", "total_cost_usd")


def _client() -> MlflowClient:
    """An MlflowClient pointed at the same tracking server the eval writes to."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def _config_id(mv) -> str:
    """The config_id tag of a ModelVersion (empty string if absent)."""
    return mv.tags.get("config_id", "")


def _resolve_version(client: MlflowClient, name: str, config_id: str):
    """Find the registered version whose config_id tag == config_id.

    Returns the matching ModelVersion, or None (after printing the error) if
    there are zero matches. On multiple matches, prints a warning and returns
    the one with the highest MLflow integer version number.
    """
    matches = client.search_model_versions(
        f"name = '{name}' AND tags.config_id = '{config_id}'"
    )
    if not matches:
        print(f"error: no version found with config_id={config_id}")
        return None
    matches.sort(key=lambda mv: int(mv.version))
    if len(matches) > 1:
        versions = [int(mv.version) for mv in matches]
        latest = versions[-1]
        print(
            f"warning: multiple versions match config_id={config_id} "
            f"(MLflow versions {versions}); using latest ({latest})"
        )
    return matches[-1]


def _current_config_id(client: MlflowClient, name: str, alias: str) -> str:
    """config_id the alias currently points at, or '' if the alias is unset."""
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except RestException:
        return ""
    return _config_id(mv)


def _append_log(alias: str, frm: str, to: str, op: str) -> None:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": alias,
        "from": frm,
        "to": to,
        "op": op,
    }
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps(event) + "\n")


def _last_log_entry(alias: str) -> dict | None:
    """Most recent log entry for `alias`, or None if the log/entry is absent."""
    if not LOG_FILE.exists():
        return None
    last = None
    with LOG_FILE.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("alias") == alias:
                last = entry
    return last


def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    client = _client()
    mv = _resolve_version(client, args.name, args.config_id)
    if mv is None:
        sys.exit(1)

    current = _current_config_id(client, args.name, args.alias)
    client.set_registered_model_alias(args.name, args.alias, mv.version)
    _append_log(args.alias, current, args.config_id, "set")

    shown_from = current if current else "(unset)"
    print(f"{args.alias}: {shown_from} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    client = _client()
    try:
        mv = client.get_model_version_by_alias(args.name, args.alias)
    except RestException:
        print(f"error: alias '{args.alias}' is not set")
        sys.exit(1)

    print(f"{args.name} @ {args.alias}")
    print(f"  config_id: {_config_id(mv)}")
    for key, value in sorted(mv.tags.items()):
        if key == "config_id":
            continue
        print(f"  {key}: {value}")

    metrics = client.get_run(str(mv.run_id)).data.metrics
    for key in SHOW_METRICS:
        if key not in metrics:
            continue
        if key.endswith("_usd"):
            print(f"  {key}: ${metrics[key]:.2f}")
        else:
            print(f"  {key}: {metrics[key]}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""
    client = _client()
    try:
        model = client.get_registered_model(args.name)
    except RestException:
        print("no aliases set")
        return

    aliases = model.aliases or {}
    if not aliases:
        print("no aliases set")
        return

    width = max(len(a) for a in aliases)
    for alias in sorted(aliases):
        mv = client.get_model_version_by_alias(args.name, alias)
        print(f"{alias.ljust(width)} -> {_config_id(mv)}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    client = _client()
    try:
        current_mv = client.get_model_version_by_alias(args.name, args.alias)
    except RestException:
        print("nothing to roll back")
        return
    current = _config_id(current_mv)

    entry = _last_log_entry(args.alias)
    if entry is None:
        print(f"no promotion history for alias {args.alias}")
        return
    if entry.get("op") == "rollback":
        print(
            f"error: {args.alias} was just rolled back; "
            "no further history to walk back to"
        )
        return
    target = entry.get("from", "")
    if not target:
        print(f"{args.alias} has no previous target (first promotion ever)")
        return

    mv = _resolve_version(client, args.name, target)
    if mv is None:
        sys.exit(1)

    client.set_registered_model_alias(args.name, args.alias, mv.version)
    _append_log(args.alias, current, target, "rollback")
    print(f"{args.alias}: {current} → {target} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
