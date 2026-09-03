"""Repository-wide, offline integrity audit for committed Adventure Bench data."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from .aggregate import ReleaseError, aggregate_release
from .collect import validate_run_id
from .render_site import render_html
from .rescore import EvidenceError, rescore_run


SITE_PAGE = Path("benchmarks/adventurebench/index.html")


class AuditError(ValueError):
    """A repository artifact is missing, malformed, unexpected, or stale."""


@dataclass(frozen=True)
class AuditReport:
    """Identifiers checked by one fully read-only repository audit."""

    run_ids: tuple[str, ...]
    release_ids: tuple[str, ...]
    site_release_id: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "releases": len(self.release_ids),
            "runs": len(self.run_ids),
            "site_release_id": self.site_release_id,
        }


def _checked_root(path: Path, label: str) -> Path | None:
    if path.is_symlink():
        raise AuditError(f"refusing symlinked {label} directory")
    if not path.exists():
        return None
    if not path.is_dir():
        raise AuditError(f"{label} must be a directory")
    return path.resolve()


def _children(root: Path, label: str) -> list[Path]:
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as err:
        raise AuditError(f"cannot read {label} directory") from err
    for child in children:
        if child.is_symlink():
            raise AuditError(f"refusing symlinked {label} artifact: {child}")
    return children


def _safe_id(value: str, label: str) -> None:
    try:
        validate_run_id(value)
    except ValueError as err:
        raise AuditError(f"{label} has an unsafe identifier: {value!r}") from err


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{label} must be a regular file: {path}")


def _directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise AuditError(f"{label} must be a directory: {path}")


def _discover_runs(runs_dir: Path | None) -> tuple[str, ...]:
    if runs_dir is None:
        return ()
    run_ids: list[str] = []
    for child in _children(runs_dir, "runs"):
        if not child.is_dir():
            raise AuditError(f"unexpected runs artifact: {child}")
        _safe_id(child.name, "run directory")
        _regular(child / "manifest.json", "run manifest")
        run_ids.append(child.name)
    return tuple(run_ids)


def _check_run_tree(run_dir: Path) -> None:
    """Reject files that rescore intentionally does not need to consume."""
    names = {child.name for child in _children(run_dir, "run")}
    if names != {"manifest.json", "responses"}:
        raise AuditError(f"run has unexpected artifacts: {run_dir}")
    _regular(run_dir / "manifest.json", "run manifest")
    responses = run_dir / "responses"
    _directory(responses, "run responses")
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        models = manifest["models"]
        expected = {Path(row["responses_file"]).name for row in models.values()}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, AttributeError, TypeError) as err:
        raise AuditError(f"cannot inspect audited run manifest: {run_dir / 'manifest.json'}") from err
    response_children = _children(responses, "run response")
    if any(not child.is_file() for child in response_children) or {child.name for child in response_children} != expected:
        raise AuditError(f"run has unexpected response artifacts: {run_dir}")


def _check_results_layout(results_dir: Path | None, run_ids: tuple[str, ...]) -> None:
    if results_dir is None:
        if run_ids:
            raise AuditError("derived results directory is missing")
        return
    expected = set(run_ids)
    found: set[str] = set()
    for child in _children(results_dir, "results"):
        if child.name == "legacy":
            _directory(child, "legacy results")
            continue
        if child.name == "README.md":
            _regular(child, "results README")
            continue
        if not child.is_dir():
            raise AuditError(f"unexpected results artifact: {child}")
        _safe_id(child.name, "derived results directory")
        found.add(child.name)
    missing, orphaned = expected - found, found - expected
    if missing:
        raise AuditError("missing derived results directories: " + ", ".join(sorted(missing)))
    if orphaned:
        raise AuditError("orphan derived results directories: " + ", ".join(sorted(orphaned)))


def _check_result_tree(results_dir: Path, run_id: str, result: dict[str, object]) -> None:
    output_dir = results_dir / run_id
    _directory(output_dir, "derived results")
    try:
        models = result["models"]
        expected = {"summary.json", "tag-matrix.json", *(f"{slug}.json" for slug in models)}
    except (KeyError, TypeError) as err:
        raise AuditError(f"cannot inspect replayed derived results for {run_id}") from err
    children = _children(output_dir, "derived results")
    if any(not child.is_file() for child in children) or {child.name for child in children} != expected:
        raise AuditError(f"derived results have unexpected artifacts: {output_dir}")


def _read_release(release_path: Path, release_id: str) -> tuple[str, ...]:
    _regular(release_path, "release artifact")
    try:
        value = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise AuditError(f"cannot read release artifact: {release_path}") from err
    if not isinstance(value, dict) or value.get("release_id") != release_id:
        raise AuditError(f"release artifact ID does not match its directory: {release_path}")
    run_ids = value.get("run_ids")
    if not isinstance(run_ids, list) or not run_ids or not all(isinstance(run_id, str) for run_id in run_ids):
        raise AuditError(f"release artifact has malformed run IDs: {release_path}")
    return tuple(run_ids)


def _discover_releases(releases_dir: Path | None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if releases_dir is None:
        return ()
    releases: list[tuple[str, tuple[str, ...]]] = []
    for child in _children(releases_dir, "releases"):
        if not child.is_dir():
            raise AuditError(f"unexpected releases artifact: {child}")
        _safe_id(child.name, "release directory")
        names = {entry.name for entry in _children(child, "release")}
        if names != {"release.json"}:
            raise AuditError(f"release has unexpected artifacts: {child}")
        releases.append((child.name, _read_release(child / "release.json", child.name)))
    return tuple(releases)


def _site_page(site_dir: Path | None) -> tuple[Path | None, str | None]:
    if site_dir is None:
        return None, None
    path = site_dir / SITE_PAGE
    resolved = path.resolve()
    if resolved != site_dir and site_dir not in resolved.parents:
        raise AuditError("published benchmark page resolves outside the site directory")
    if not path.exists():
        return path, None
    _regular(path, "published benchmark page")
    try:
        return path, path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise AuditError(f"cannot read published benchmark page: {path}") from err


def audit_repository(*, runs_dir: Path, results_dir: Path, releases_dir: Path,
                     dataset_path: Path, site_dir: Path) -> AuditReport:
    """Audit every local publication artifact without writing or calling a provider."""
    checked_runs = _checked_root(runs_dir, "runs")
    checked_results = _checked_root(results_dir, "results")
    checked_releases = _checked_root(releases_dir, "releases")
    checked_site = _checked_root(site_dir, "site")
    run_ids = _discover_runs(checked_runs)
    _check_results_layout(checked_results, run_ids)
    replayed: dict[str, dict[str, object]] = {}
    for run_id in run_ids:
        assert checked_runs is not None and checked_results is not None
        try:
            result = rescore_run(
                runs_dir=checked_runs,
                results_dir=checked_results,
                run_id=run_id,
                dataset_path=dataset_path,
                check=True,
            )
        except (EvidenceError, ValueError) as err:
            raise AuditError(f"run {run_id} failed offline rescore: {err}") from err
        _check_run_tree(checked_runs / run_id)
        _check_result_tree(checked_results, run_id, result)
        replayed[run_id] = result

    releases = _discover_releases(checked_releases)
    release_artifacts: dict[str, dict[str, object]] = {}
    for release_id, release_run_ids in releases:
        unknown_runs = set(release_run_ids) - set(run_ids)
        if unknown_runs:
            raise AuditError(f"release {release_id} references undiscovered run(s): {', '.join(sorted(unknown_runs))}")
        assert checked_runs is not None and checked_results is not None and checked_releases is not None
        try:
            release_artifacts[release_id] = aggregate_release(
                runs_dir=checked_runs,
                results_dir=checked_results,
                releases_dir=checked_releases,
                release_id=release_id,
                run_ids=release_run_ids,
                dataset_path=dataset_path,
                check=True,
            )
        except (ReleaseError, ValueError) as err:
            raise AuditError(f"release {release_id} failed canonical aggregate: {err}") from err

    page_path, page = _site_page(checked_site)
    site_release_id = None
    if page is not None:
        matching = [release_id for release_id, artifact in release_artifacts.items() if render_html(artifact) == page]
        if len(matching) != 1:
            label = str(page_path) if page_path is not None else str(SITE_PAGE)
            raise AuditError(f"published benchmark page does not match exactly one audited release: {label}")
        site_release_id = matching[0]
    return AuditReport(run_ids=run_ids, release_ids=tuple(release_artifacts), site_release_id=site_release_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read-only offline audit of all committed Adventure Bench artifacts.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--releases-dir", default="releases")
    parser.add_argument("--data", default="adventure_bench/data/cases.jsonl")
    parser.add_argument("--site-dir", default="site")
    args = parser.parse_args(argv)
    try:
        report = audit_repository(
            runs_dir=Path(args.runs_dir),
            results_dir=Path(args.results_dir),
            releases_dir=Path(args.releases_dir),
            dataset_path=Path(args.data),
            site_dir=Path(args.site_dir),
        )
    except (AuditError, ValueError) as err:
        parser.error(str(err))
    print(json.dumps(report.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
