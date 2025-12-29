#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import re
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable, Literal

WorkspaceRepoName = str
Severity = Literal["pass", "warn", "fail"]


CANONICAL_REPOS: tuple[str, ...] = (
    "SearchSpaceCollapse",
    "qigkernels",
    "qig-core",
    "qig-consciousness",
    "qig-verification",
    "qig-dreams",
    "qig-con2",
    "qig-archive",
)

SCAN_ONLY_REPOS: frozenset[str] = frozenset({"qig-verification", "qig-con2", "qig-archive"})

ARCHIVAL_REPOS: frozenset[str] = frozenset({"qig-con2", "qig-archive"})

ENGINE_REPOS: frozenset[str] = frozenset({"qigkernels", "qig-core", "qig-consciousness"})

CANONICAL_FROZEN_FACTS_PATH = "qig-verification/docs/current/FROZEN_FACTS.md"

# NOTE: dependencies must flow "downstream" only. Any edge not in this allowlist is reported.
# (Downstream repo) -> set of allowed (upstream) deps.

# Rough package-name -> repo-name mapping for static import scans.
MODULE_TO_REPO: dict[str, str] = {
    "qigkernels": "qigkernels",
    "qig_core": "qig-core",
    "qig_consciousness": "qig-consciousness",
    "qig_verification": "qig-verification",
    "qig_dreams": "qig-dreams",
}

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
}

TEXT_EXTS = {
    ".py",
    ".pyi",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
}


SECRET_HINT_RE = re.compile(
    r"(BEGIN\s+(?:OPENSSH|RSA|EC)?\s*PRIVATE\s+KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})",
    re.IGNORECASE,
)


@dataclasses.dataclass(frozen=True)
class Finding:
    severity: Severity
    rule: str
    message: str
    paths: tuple[str, ...] = ()
    recommendation: str = ""
    owner_repo: str = ""


@dataclasses.dataclass
class CommandResult:
    cmd: str
    cwd: str
    exit_code: int
    duration_s: float
    stdout_tail: str


@dataclasses.dataclass
class RepoReport:
    name: WorkspaceRepoName
    path: Path
    unexpected_repo: bool = False
    findings: list[Finding] = dataclasses.field(default_factory=list)
    commands: list[CommandResult] = dataclasses.field(default_factory=list)
    entry_points: list[str] = dataclasses.field(default_factory=list)
    toolchain: dict[str, str] = dataclasses.field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def status(self) -> Severity:
        if any(f.severity == "fail" for f in self.findings):
            return "fail"
        if any(f.severity == "warn" for f in self.findings):
            return "warn"
        return "pass"


def iter_repo_dirs(root: Path) -> list[RepoReport]:
    reports: list[RepoReport] = []

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"reports", ".windsurf", ".venv", ".pytest_cache", "__pycache__"}:
            continue

        looks_like_qig_repo = child.name in CANONICAL_REPOS or child.name.startswith("qig-")
        if not looks_like_qig_repo:
            continue

        unexpected_repo = child.name not in CANONICAL_REPOS
        reports.append(RepoReport(name=child.name, path=child, unexpected_repo=unexpected_repo))

    return reports


def iter_text_files(repo_path: Path) -> Iterable[Path]:
    for p in repo_path.rglob("*"):
        if p.is_dir():
            # Prune by directory name
            if p.name in SKIP_DIR_NAMES:
                # rglob doesn't support pruning directly; we skip children via name check in loop
                continue
            continue

        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue

        if p.suffix.lower() not in TEXT_EXTS:
            continue

        yield p


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except OSError:
            return None
    except OSError:
        return None


IMPORT_PY_RE = re.compile(r"^\s*(?:from\s+([a-zA-Z_][\w\.]*?)\s+import|import\s+([a-zA-Z_][\w\.]*))\b")
IMPORT_TS_RE = re.compile(r"\bfrom\s+['\"]([^'\"]+)['\"]")
REQUIRE_TS_RE = re.compile(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)")


def is_archival_repo(repo_name: str) -> bool:
    if repo_name in ARCHIVAL_REPOS:
        return True
    return "archive" in repo_name.lower()


def is_scan_only_repo(repo_name: str) -> bool:
    if repo_name in SCAN_ONLY_REPOS:
        return True
    return is_archival_repo(repo_name)


def scan_secrets(repo: RepoReport, *, root: Path) -> None:
    hits: list[str] = []
    for ext in (".pem", ".key", ".p12", ".pfx"):
        for p in repo.path.rglob(f"*{ext}"):
            if any(part in SKIP_DIR_NAMES for part in p.parts):
                continue
            content = read_text(p)
            if not content:
                continue
            if "PRIVATE KEY" in content or SECRET_HINT_RE.search(content):
                hits.append(p.relative_to(root).as_posix())
    for f in iter_text_files(repo.path):
        content = read_text(f)
        if not content:
            continue
        if SECRET_HINT_RE.search(content):
            hits.append(f.relative_to(root).as_posix())

    if hits:
        repo.add(
            Finding(
                severity="fail",
                rule="security.secrets",
                message="Potential secret material detected (high-confidence patterns)",
                paths=tuple(sorted(set(hits))[:80]),
                recommendation="Remove secrets from the repo history and rotate credentials if applicable.",
                owner_repo=repo.name,
            )
        )


def extract_repo_deps(file_path: Path, content: str) -> set[str]:
    deps: set[str] = set()

    if file_path.suffix == ".py" or file_path.suffix == ".pyi":
        for line in content.splitlines():
            m = IMPORT_PY_RE.match(line)
            if not m:
                continue
            mod = (m.group(1) or m.group(2) or "").split(".")[0]
            repo = MODULE_TO_REPO.get(mod)
            if repo:
                deps.add(repo)

    elif file_path.suffix in {".ts", ".tsx", ".js", ".jsx"}:
        for m in IMPORT_TS_RE.finditer(content):
            spec = m.group(1)
            top = spec.split("/")[0]
            repo = MODULE_TO_REPO.get(top)
            if repo:
                deps.add(repo)
        for m in REQUIRE_TS_RE.finditer(content):
            spec = m.group(1)
            top = spec.split("/")[0]
            repo = MODULE_TO_REPO.get(top)
            if repo:
                deps.add(repo)

    return deps


AuditConfig = dict[str, Any]


def load_audit_config(config_path: Path) -> AuditConfig:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load audit config") from exc

    data = read_text(config_path)
    if data is None:
        raise RuntimeError(f"Failed to read config: {config_path}")
    cfg = yaml.safe_load(data)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid config (expected mapping): {config_path}")
    return cfg


def _get_allowed_deps(config: AuditConfig) -> dict[str, set[str]]:
    raw = (config.get("governance") or {}).get("allowed_deps")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, set[str]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = set(v)
        elif isinstance(k, str) and v == []:
            out[k] = set()
    return out


def _resolve_config_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Prefer explicit paths relative to CWD; then repo root; then alongside this script.
    if p.exists():
        return p
    repo_root = Path(__file__).resolve().parents[1]
    alt_repo = repo_root / p
    if alt_repo.exists():
        return alt_repo
    alt_repo_configs = repo_root / "configs" / p.name
    if alt_repo_configs.exists():
        return alt_repo_configs
    alt_tools = Path(__file__).resolve().parent / p
    return alt_tools


def run_cmd(cmd: list[str], cwd: Path, timeout_s: int) -> CommandResult:
    start = time.time()
    output_tail: collections.deque[str] = collections.deque(maxlen=60)
    exit_code = -1

    try:
        with subprocess.Popen(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PAGER": "cat"},
        ) as proc:
            if proc.stdout is not None:
                for line in proc.stdout:
                    output_tail.append(line.rstrip("\n"))
            exit_code = int(proc.wait(timeout=timeout_s))
    except subprocess.TimeoutExpired:
        output_tail.append("... (process timed out)")
    except (OSError, ValueError) as exc:
        output_tail.append(f"Failed to run command: {exc}")

    dur = time.time() - start

    return CommandResult(
        cmd=" ".join(cmd),
        cwd=str(cwd),
        exit_code=int(exit_code),
        duration_s=dur,
        stdout_tail="\n".join(output_tail),
    )


def _try_version(cmd: list[str], cwd: Path, timeout_s: int) -> str | None:
    try:
        res = run_cmd(cmd, cwd, timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return None

    if res.exit_code != 0:
        return None

    # Usually versions are a single line; take the first non-empty line.
    for line in res.stdout_tail.splitlines():
        if line.strip():
            return line.strip()
    return None


def detect_toolchain_versions(repo_path: Path, timeout_s: int) -> dict[str, str]:
    versions: dict[str, str] = {}
    py = _try_version(["python", "--version"], repo_path, timeout_s)
    if py:
        versions["python"] = py
    uv = _try_version(["uv", "--version"], repo_path, timeout_s)
    if uv:
        versions["uv"] = uv
    ruff = _try_version(["python", "-m", "ruff", "--version"], repo_path, timeout_s)
    if ruff:
        versions["ruff"] = ruff
    pytest = _try_version(["python", "-m", "pytest", "--version"], repo_path, timeout_s)
    if pytest:
        versions["pytest"] = pytest
    return versions


def venv_has_modules(venv_python: Path, repo_path: Path, timeout_s: int, modules: list[str]) -> bool:
    check = "\n".join(["import importlib", *[f"importlib.import_module('{m}')" for m in modules]])
    try:
        res = run_cmd([str(venv_python), "-c", check], repo_path, timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.exit_code == 0


def detect_entry_points(repo_path: Path) -> list[str]:
    candidates: list[str] = []

    patterns = [
        "**/__main__.py",
        "chat_interfaces/**/*.py",
        "scripts/**/*",
        "cli/**/*",
        "experiments/**/*",
        "src/index.ts",
        "src/main.ts",
        "src/app.ts",
        "server/src/index.ts",
        "server/src/main.ts",
        "app.ts",
        "index.ts",
    ]

    for pat in patterns:
        for p in repo_path.glob(pat):
            if p.is_dir():
                continue
            if any(part in SKIP_DIR_NAMES for part in p.parts):
                continue
            rel = p.relative_to(repo_path).as_posix()
            candidates.append(rel)

    # Dedup and keep short list
    uniq = sorted(set(candidates))
    return uniq[:40]


def check_lockfiles(repo: RepoReport) -> None:
    repo_path = repo.path

    pyproject = repo_path / "pyproject.toml"
    package_json = repo_path / "package.json"

    if pyproject.exists():
        uv_lock = repo_path / "uv.lock"
        if not uv_lock.exists():
            repo.add(
                Finding(
                    severity="warn",
                    rule="tooling.python.uv_lock_present",
                    message="Python repo missing uv.lock",
                    paths=(str(uv_lock),),
                    recommendation="Commit uv.lock for reproducibility.",
                    owner_repo=repo.name,
                )
            )

    if package_json.exists():
        locks = [
            repo_path / "yarn.lock",
            repo_path / "pnpm-lock.yaml",
            repo_path / "package-lock.json",
        ]
        present = [p.name for p in locks if p.exists()]
        if len(present) == 0:
            repo.add(
                Finding(
                    severity="warn",
                    rule="tooling.node.lockfile_single",
                    message="Node repo missing lockfile",
                    paths=(),
                    recommendation="Choose exactly one package manager and commit its lockfile.",
                    owner_repo=repo.name,
                )
            )
        elif len(present) > 1:
            repo.add(
                Finding(
                    severity="fail",
                    rule="tooling.node.lockfile_single",
                    message=f"Node repo has multiple lockfiles: {', '.join(present)}",
                    paths=tuple(str(repo_path / p) for p in present),
                    recommendation="Remove extra lockfiles; enforce one package manager per repo.",
                    owner_repo=repo.name,
                )
            )


def check_docs_basics(repo: RepoReport) -> None:
    repo_path = repo.path

    readme = None
    for name in ("README.md", "Readme.md", "README.MD"):
        if (repo_path / name).exists():
            readme = repo_path / name
            break

    if readme is None:
        repo.add(
            Finding(
                severity="fail" if repo.name in ENGINE_REPOS else "warn",
                rule="docs.readme",
                message="Missing README.md",
                recommendation="Add a README explaining purpose, boundaries, and entry points.",
                owner_repo=repo.name,
            )
        )

    docs_index_candidates = (
        "docs/00-index.md",
        "docs/INDEX.md",
        "docs/README.md",
    )
    has_index = any((repo_path / p).exists() for p in docs_index_candidates)
    if not has_index:
        severity: Severity = "warn"
        repo.add(
            Finding(
                severity=severity,
                rule="docs.index",
                message="Missing docs index (docs/00-index.md or equivalent)",
                recommendation="Add a docs index listing canonical docs, entry points, and validated vs exploratory.",
                owner_repo=repo.name,
            )
        )


def _node_has_package_json(repo_path: Path) -> bool:
    return (repo_path / "package.json").exists()


def _iter_ts_dirs(repo_path: Path) -> Iterable[Path]:
    for p in repo_path.rglob("*"):
        if not p.is_dir():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        yield p


def _read_index_exports(index_path: Path) -> list[str]:
    content = read_text(index_path)
    if not content:
        return []
    exports: list[str] = []
    for line in content.splitlines():
        s = line.strip()
        if not (s.startswith("export") and "from" in s):
            continue
        m = re.search(r"from\s+['\"]([^'\"]+)['\"]", s)
        if not m:
            continue
        exports.append(m.group(1))
    return exports


def scan_node_barrels(repo: RepoReport) -> None:
    if not _node_has_package_json(repo.path):
        return

    # Missing barrel heuristic: directories with many TS files but no index.ts
    missing: list[str] = []
    for d in _iter_ts_dirs(repo.path):
        ts_files = [p for p in d.iterdir() if p.is_file() and p.suffix in {".ts", ".tsx"}]
        if len(ts_files) < 10:
            continue
        if (d / "index.ts").exists() or (d / "index.tsx").exists():
            continue
        rel = d.relative_to(repo.path).as_posix()
        missing.append(rel)

    if missing:
        repo.add(
            Finding(
                severity="warn",
                rule="structure.barrels",
                message="Potential missing barrel exports (index.ts) in high-file-count TS directories",
                paths=tuple(missing[:50]),
                recommendation="Consider adding index.ts barrels only at stable boundaries (avoid barrel-of-barrels).",
                owner_repo=repo.name,
            )
        )

    # Barrel-of-barrels / circular barrel heuristic: index.ts exporting from sibling index.ts.
    index_files = [p for p in repo.path.rglob("index.ts") if p.is_file() and not any(part in SKIP_DIR_NAMES for part in p.parts)]
    adj: dict[Path, set[Path]] = {}
    for idx in index_files:
        targets: set[Path] = set()
        for spec in _read_index_exports(idx):
            if not spec.startswith("."):
                continue
            resolved = (idx.parent / spec).resolve()
            # Support re-exports pointing at folders or files.
            candidates = [
                resolved,
                resolved.with_suffix(".ts"),
                resolved / "index.ts",
            ]
            for c in candidates:
                if c.exists() and c.name == "index.ts":
                    targets.add(c)
                    break
        if targets:
            adj[idx] = targets

    visiting: set[Path] = set()
    visited: set[Path] = set()
    cycles: set[Path] = set()

    def dfs(n: Path) -> None:
        if n in visited:
            return
        if n in visiting:
            cycles.add(n)
            return
        visiting.add(n)
        for m in adj.get(n, set()):
            dfs(m)
        visiting.remove(n)
        visited.add(n)

    for n in adj.keys():
        dfs(n)

    if cycles:
        rels = [p.relative_to(repo.path).as_posix() for p in sorted(cycles)]
        repo.add(
            Finding(
                severity="warn",
                rule="structure.barrels",
                message="Potential circular barrel exports detected (index.ts re-exporting other index.ts)",
                paths=tuple(rels[:50]),
                recommendation="Avoid barrel-of-barrels patterns; prefer direct imports or flatten a single stable barrel.",
                owner_repo=repo.name,
            )
        )


def scan_frozen_facts(root: Path, repo_reports: dict[str, RepoReport]) -> list[Finding]:
    findings: list[Finding] = []

    def _is_archival_rel(rel: str) -> bool:
        return "/archive/" in rel or rel.startswith("archive/")

    def _is_pointer_frozen_facts(content: str | None) -> bool:
        if not content:
            return False
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if len(lines) > 120:
            return False
        text = "\n".join(lines).lower()
        if "do not edit" in text and "canonical" in text:
            return True
        return False

    # 1) File presence: any FROZEN_FACTS.md outside canonical path
    for p in root.rglob("FROZEN_FACTS.md"):
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith("qig-archive/"):
            continue
        if _is_archival_rel(rel):
            continue
        if rel != CANONICAL_FROZEN_FACTS_PATH and not _is_pointer_frozen_facts(read_text(p)):
            findings.append(
                Finding(
                    severity="fail",
                    rule="governance.docs.frozen_facts.canonical_location",
                    message="Non-canonical FROZEN_FACTS.md found",
                    paths=(rel,),
                    recommendation=f"Keep only the canonical file at {CANONICAL_FROZEN_FACTS_PATH}; others must be pointer/redirect only.",
                    owner_repo="qig-verification",
                )
            )

    # 2) References: any reference to FROZEN_FACTS that does not mention canonical path
    ref_paths: list[str] = []
    for repo in repo_reports.values():
        for f in iter_text_files(repo.path):
            content = read_text(f)
            if not content:
                continue
            if "FROZEN_FACTS" not in content:
                continue
            rel = f.relative_to(root).as_posix()
            if _is_archival_rel(rel):
                continue
            ref_paths.append(rel)
            if CANONICAL_FROZEN_FACTS_PATH not in content:
                findings.append(
                    Finding(
                        severity="warn",
                        rule="governance.docs.frozen_facts.reference",
                        message="FROZEN_FACTS referenced without canonical path string present",
                        paths=(rel,),
                        recommendation=f"Update references to point to {CANONICAL_FROZEN_FACTS_PATH}.",
                        owner_repo="qig-verification",
                    )
                )

    if not ref_paths:
        findings.append(
            Finding(
                severity="warn",
                rule="governance.docs.frozen_facts.reference",
                message="No references to FROZEN_FACTS found (could be OK, but verify documentation pointers)",
                recommendation=f"Ensure canonical facts live at {CANONICAL_FROZEN_FACTS_PATH} and are referenced where needed.",
                owner_repo="qig-verification",
            )
        )

    return findings


def scan_prohibitions(root: Path, repo_reports: dict[str, RepoReport], config: AuditConfig) -> list[Finding]:
    findings: list[Finding] = []

    raw = config.get("prohibitions")
    if not isinstance(raw, list):
        return findings

    rules: list[dict[str, Any]] = [r for r in raw if isinstance(r, dict)]
    if not rules:
        return findings

    def is_archival_path(path: str) -> bool:
        return "/archive/" in path or path.startswith("archive/")

    for rule_cfg in rules:
        rule = rule_cfg.get("rule")
        owner = rule_cfg.get("owner_repo")
        patterns = rule_cfg.get("patterns")
        include_prefixes = rule_cfg.get("include_prefixes")
        exclude_prefixes = rule_cfg.get("exclude_prefixes")
        include_exts = rule_cfg.get("include_exts")
        sev_active = rule_cfg.get("severity_active")
        sev_archive = rule_cfg.get("severity_archive")

        if not isinstance(rule, str) or not rule:
            continue
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            continue

        prefixes = (
            [p for p in include_prefixes if isinstance(p, str)]
            if isinstance(include_prefixes, list)
            else []
        )
        excludes = (
            [p for p in exclude_prefixes if isinstance(p, str)]
            if isinstance(exclude_prefixes, list)
            else []
        )
        exts = (
            [e for e in include_exts if isinstance(e, str)]
            if isinstance(include_exts, list)
            else []
        )
        if not prefixes:
            continue
        if not exts:
            exts = [".py", ".pyi", ".ts", ".tsx", ".js", ".jsx"]
        exts_lower = {e.lower() for e in exts}

        severity_active: Severity = sev_active if sev_active in {"pass", "warn", "fail"} else "fail"
        severity_archive: Severity = sev_archive if sev_archive in {"pass", "warn", "fail"} else "warn"

        compiled: list[re.Pattern[str]] = []
        for pat in patterns:
            try:
                compiled.append(re.compile(pat))
            except re.error:
                continue
        if not compiled:
            continue

        hits_active: set[str] = set()
        hits_archive: set[str] = set()

        for repo in repo_reports.values():
            for f in iter_text_files(repo.path):
                if f.suffix.lower() not in exts_lower:
                    continue

                rel = f.relative_to(root).as_posix()
                if not any(rel.startswith(pref) for pref in prefixes):
                    continue
                if any(rel.startswith(pref) for pref in excludes):
                    continue

                content = read_text(f)
                if not content:
                    continue

                if not any(p.search(content) for p in compiled):
                    continue

                archival = is_archival_repo(repo.name) or is_archival_path(rel)
                if archival:
                    hits_archive.add(rel)
                else:
                    hits_active.add(rel)

        if hits_active:
            findings.append(
                Finding(
                    severity=severity_active,
                    rule=rule,
                    message="Forbidden architecture pattern detected in active code",
                    paths=tuple(sorted(hits_active))[:80],
                    recommendation="Remove forbidden symbols/patterns from active code paths.",
                    owner_repo=owner if isinstance(owner, str) else "qig-consciousness",
                )
            )

        if hits_archive:
            findings.append(
                Finding(
                    severity=severity_archive,
                    rule=rule,
                    message="Forbidden architecture pattern detected under archive path",
                    paths=tuple(sorted(hits_archive))[:80],
                    recommendation=(
                        "Keep forbidden patterns out of active code; archive hits are reported as warnings."
                    ),
                    owner_repo=owner if isinstance(owner, str) else "qig-consciousness",
                )
            )

    return findings


FROZEN_CONST_HINT_RE = re.compile(
    r"\b(FROZEN_FACTS|frozen facts|frozen constants|PLANCK|BOLTZMANN|SPEED_OF_LIGHT|GRAVITATIONAL_CONSTANT)\b",
    re.IGNORECASE,
)


def scan_frozen_constants_outside_verification(root: Path, repo_reports: dict[str, RepoReport]) -> list[Finding]:
    findings: list[Finding] = []

    for repo in repo_reports.values():
        if repo.name == "qig-verification":
            continue

        hits: list[str] = []
        for f in iter_text_files(repo.path):
            if f.suffix.lower() not in {".py", ".ts", ".tsx", ".js", ".jsx"}:
                continue
            content = read_text(f)
            if not content:
                continue
            if FROZEN_CONST_HINT_RE.search(content):
                hits.append(f.relative_to(root).as_posix())

        if hits:
            findings.append(
                Finding(
                    severity="warn",
                    rule="purity.frozen_constants.outside_verification",
                    message="Possible frozen-constant definition/reference outside qig-verification",
                    paths=tuple(sorted(set(hits))[:50]),
                    recommendation="Validate that frozen facts/constants are sourced from qig-verification, not redefined elsewhere.",
                    owner_repo="qig-verification",
                )
            )

    return findings


CIRCULARITY_HINT_RE = re.compile(
    r"\b(compute_D|\bD\s*=|kappa|κ|from_kappa|derived\s+from\s+kappa|justify\s+kappa)\b",
    re.IGNORECASE,
)


def scan_potential_circularity(root: Path, repo_reports: dict[str, RepoReport]) -> list[Finding]:
    findings: list[Finding] = []

    for repo in repo_reports.values():
        hits: list[str] = []
        for f in iter_text_files(repo.path):
            if f.suffix.lower() not in {".py", ".md", ".ts", ".tsx", ".js", ".jsx"}:
                continue
            content = read_text(f)
            if not content:
                continue
            if CIRCULARITY_HINT_RE.search(content):
                hits.append(f.relative_to(root).as_posix())

        if hits:
            findings.append(
                Finding(
                    severity="warn",
                    rule="science.circularity_risk.D_kappa",
                    message=f"Potential circularity risk signals found in {repo.name}",
                    paths=tuple(sorted(set(hits))[:50]),
                    recommendation="If experiments claim D supports κ, require anti-circular harness (cross-fit/null/orthogonal/ablation) and cite where stored.",
                    owner_repo=repo.name,
                )
            )

    return findings


def scan_import_direction(
    root: Path, repo_reports: dict[str, RepoReport], config: AuditConfig
) -> list[Finding]:
    findings: list[Finding] = []

    allowed_deps = _get_allowed_deps(config)

    edges: dict[tuple[str, str], list[str]] = {}

    for repo in repo_reports.values():
        for f in iter_text_files(repo.path):
            content = read_text(f)
            if not content:
                continue
            deps = extract_repo_deps(f, content)
            if not deps:
                continue
            for dep in deps:
                edges.setdefault((repo.name, dep), []).append(f.relative_to(root).as_posix())

    for (src, dep), files in sorted(edges.items()):
        if src == dep:
            continue
        allowed = allowed_deps.get(src, set())
        if dep not in allowed:
            archival = is_archival_repo(src)
            findings.append(
                Finding(
                    severity="warn" if archival else "fail",
                    rule="governance.import_direction",
                    message=f"Disallowed dependency: {src} imports {dep}",
                    paths=tuple(sorted(set(files))[:80]),
                    recommendation="Remove reverse dependency or move code to the correct layer.",
                    owner_repo=src,
                )
            )

    return findings


def scan_dry_collisions(root: Path, repo_reports: dict[str, RepoReport], config: AuditConfig) -> list[Finding]:
    findings: list[Finding] = []

    dry_cfg = config.get("dry")
    if not isinstance(dry_cfg, dict):
        return findings

    primitives = dry_cfg.get("primitives")
    coexistence = dry_cfg.get("coexistence")
    if not isinstance(primitives, list):
        primitives = []
    if not isinstance(coexistence, list):
        coexistence = []

    patterns: dict[str, re.Pattern[str]] = {}
    primitive_rules: list[dict[str, Any]] = []
    for item in primitives:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        pat = item.get("pattern")
        if not isinstance(name, str) or not isinstance(pat, str):
            continue
        patterns[name] = re.compile(pat, re.MULTILINE)
        primitive_rules.append(item)

    coexist_rules: list[dict[str, Any]] = []
    for item in coexistence:
        if not isinstance(item, dict):
            continue
        pats = item.get("patterns")
        if not isinstance(pats, dict):
            continue
        for k, v in pats.items():
            if isinstance(k, str) and isinstance(v, str):
                patterns[k] = re.compile(v, re.MULTILINE)
        coexist_rules.append(item)

    if not patterns:
        return findings

    occurrences: dict[str, list[str]] = {k: [] for k in patterns}

    for repo in repo_reports.values():
        for f in iter_text_files(repo.path):
            if f.suffix.lower() != ".py":
                continue
            content = read_text(f)
            if not content:
                continue
            rel = f.relative_to(root).as_posix()
            for name, pat in patterns.items():
                if pat.search(content):
                    occurrences[name].append(rel)

    for rule_cfg in primitive_rules:
        name = rule_cfg.get("name")
        rule = rule_cfg.get("rule")
        if not isinstance(name, str) or not isinstance(rule, str):
            continue
        occ = sorted(set(occurrences.get(name, [])))
        if not occ:
            continue

        allowed_suffixes = rule_cfg.get("allowed_suffixes")
        allowed_prefixes = rule_cfg.get("allowed_prefixes")
        suffixes = [s for s in allowed_suffixes if isinstance(s, str)] if isinstance(allowed_suffixes, list) else []
        prefixes = [s for s in allowed_prefixes if isinstance(s, str)] if isinstance(allowed_prefixes, list) else []

        def is_allowed(path: str) -> bool:
            if any(path.endswith(suf) for suf in suffixes):
                return True
            if any(path.startswith(pref) for pref in prefixes):
                return True
            return False

        extra = [p for p in occ if not is_allowed(p)]
        if not extra:
            continue

        sev = rule_cfg.get("severity")
        severity: Severity = sev if sev in {"pass", "warn", "fail"} else "fail"
        msg = rule_cfg.get("message")
        rec = rule_cfg.get("recommendation")
        owner = rule_cfg.get("owner_repo")
        findings.append(
            Finding(
                severity=severity,
                rule=rule,
                message=msg if isinstance(msg, str) else f"Duplicate definition(s) for {name} found",
                paths=tuple(extra),
                recommendation=rec if isinstance(rec, str) else "",
                owner_repo=owner if isinstance(owner, str) else "",
            )
        )

    for rule_cfg in coexist_rules:
        rule = rule_cfg.get("rule")
        if not isinstance(rule, str):
            continue
        require_all = rule_cfg.get("require_all")
        req = [x for x in require_all if isinstance(x, str)] if isinstance(require_all, list) else []
        if not req:
            continue
        if not all(occurrences.get(k) for k in req):
            continue

        sev = rule_cfg.get("severity")
        severity = sev if sev in {"pass", "warn", "fail"} else "warn"
        msg = rule_cfg.get("message")
        rec = rule_cfg.get("recommendation")
        owner = rule_cfg.get("owner_repo")
        paths: list[str] = []
        for k in req:
            paths.extend(sorted(set(occurrences.get(k, []))))
        findings.append(
            Finding(
                severity=severity,
                rule=rule,
                message=msg if isinstance(msg, str) else "DRY coexistence rule triggered",
                paths=tuple(paths[:80]),
                recommendation=rec if isinstance(rec, str) else "",
                owner_repo=owner if isinstance(owner, str) else "",
            )
        )

    return findings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qig_audit.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            QIG repo constellation audit runner.

            No-surprise-writes policy:
            - Does not modify repo files.
            - Only writes to workspace-level reports/.
            - May create/modify per-repo .venv via uv sync (allowed).
            - Does NOT install node deps automatically.
            """
        ).strip(),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "qig_audit_config.yaml"),
        help="Path to YAML config defining repo-specific audit rules.",
    )
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Force static-only scan: skip ALL subprocess commands (uv/ruff/pytest/node scripts).",
    )
    parser.add_argument(
        "--run-tools",
        action="store_true",
        help="Opt in to running subprocess tooling (ruff/pytest/node scripts). Default is static-only.",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Allow networked dependency syncing (uv sync). Implies --run-tools unless --no-network is set.",
    )

    return parser.parse_args()


def _run_repo_checks_all(repo_reports: list[RepoReport], args: argparse.Namespace) -> None:
    for repo in repo_reports:
        if repo.unexpected_repo:
            repo.add(
                Finding(
                    severity="warn",
                    rule="workspace.unexpected_repo",
                    message="Repo detected via qig-* heuristic; not in canonical scope list",
                    recommendation="Confirm whether this repo should be governed/audited or treated as archival.",
                    owner_repo=repo.name,
                )
            )

        try:
            run_tools = bool(args.run_tools) or bool(args.network)
            allow_network = bool(args.network)
            if bool(args.no_network):
                run_tools = False
                allow_network = False

            run_repo_checks(
                repo,
                timeout_s=int(args.timeout_s),
                run_tools=run_tools,
                allow_network=allow_network,
            )
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            repo.add(
                Finding(
                    severity="fail",
                    rule="runner.exception",
                    message=f"Audit runner exception while scanning repo: {exc}",
                    recommendation="Rerun with a smaller timeout or inspect the stack trace; fix runner robustness.",
                    owner_repo=repo.name,
                )
            )


def _run_workspace_scans(
    root: Path, repo_by_name: dict[str, RepoReport], config: AuditConfig
) -> list[Finding]:
    workspace_findings: list[Finding] = []
    for fn in (
        scan_frozen_facts,
        scan_frozen_constants_outside_verification,
        scan_potential_circularity,
    ):
        try:
            workspace_findings.extend(fn(root, repo_by_name))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            workspace_findings.append(
                Finding(
                    severity="fail",
                    rule="runner.exception.workspace_scan",
                    message=f"Workspace scan exception: {fn.__name__}: {exc}",
                    recommendation="Fix runner robustness so summary can be trusted.",
                    owner_repo="qig-consciousness",
                )
            )

    # Config-driven scans
    for fn in (scan_import_direction, scan_dry_collisions, scan_prohibitions):
        try:
            workspace_findings.extend(fn(root, repo_by_name, config))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            workspace_findings.append(
                Finding(
                    severity="fail",
                    rule="runner.exception.workspace_scan",
                    message=f"Workspace scan exception: {fn.__name__}: {exc}",
                    recommendation="Fix runner robustness so summary can be trusted.",
                    owner_repo="qig-consciousness",
                )
            )

    return workspace_findings


def run_repo_checks(
    repo: RepoReport,
    *,
    timeout_s: int,
    run_tools: bool,
    allow_network: bool,
) -> None:
    repo_path = repo.path

    # Entry points
    repo.entry_points = detect_entry_points(repo_path)
    if not repo.entry_points:
        repo.add(
            Finding(
                severity="warn",
                rule="entry_points.detect",
                message="No obvious entry point detected (best-effort scan)",
                recommendation="Add an index/README section listing canonical entry points.",
                owner_repo=repo.name,
            )
        )

    # Basic docs + lockfiles
    check_lockfiles(repo)
    check_docs_basics(repo)
    scan_secrets(repo, root=repo_path.parent)
    scan_node_barrels(repo)
    if is_scan_only_repo(repo.name):
        repo.add(
            Finding(
                severity="pass",
                rule="policy.scan_only",
                message="Repo marked scan-only; no subprocess commands executed",
                owner_repo=repo.name,
            )
        )
        return
    if not run_tools:
        repo.add(
            Finding(
                severity="pass",
                rule="policy.static_only",
                message="Static-only mode: skipping all subprocess tool runs (use --run-tools / --network to opt in)",
                owner_repo=repo.name,
            )
        )
        return
    repo.toolchain = detect_toolchain_versions(repo_path, min(timeout_s, 60))

    pyproject = repo_path / "pyproject.toml"
    package_json = repo_path / "package.json"

    if pyproject.exists():
        venv_python = repo_path / ".venv" / "bin" / "python"
        need_sync = True
        if venv_python.exists() and venv_has_modules(
            venv_python, repo_path, min(timeout_s, 60), modules=["ruff", "pytest"]
        ):
            need_sync = False

        if need_sync:
            if allow_network:
                res = run_cmd(["uv", "sync", "--frozen"], repo_path, timeout_s)
                repo.commands.append(res)
                if res.exit_code != 0:
                    repo.add(
                        Finding(
                            severity="fail",
                            rule="python.uv_sync_frozen",
                            message="uv sync --frozen failed",
                            paths=(repo.name,),
                            recommendation="Fix uv sync errors (dependency resolution / lockfile drift).",
                            owner_repo=repo.name,
                        )
                    )
                    # Don't try to run more commands if env didn't sync
                    return
            else:
                repo.add(
                    Finding(
                        severity="warn",
                        rule="python.uv_sync_skipped",
                        message="Skipping uv sync (no network). Tool runs may fail if deps are not already installed.",
                        recommendation="Re-run with --network to allow uv sync, or ensure tools exist in an existing venv.",
                        owner_repo=repo.name,
                    )
                )
        else:
            repo.add(
                Finding(
                    severity="pass",
                    rule="python.uv_sync_frozen",
                    message="Skipping uv sync: existing .venv has ruff+pytest",
                    owner_repo=repo.name,
                )
            )

        runner = [str(venv_python)] if venv_python.exists() else ["python"]

        for cmd, rule in [
            (runner + ["-m", "ruff", "check", "."], "python.ruff.check"),
            (runner + ["-m", "ruff", "format", "--check", "."], "python.ruff.format_check"),
            (runner + ["-m", "pytest", "-q"], "python.pytest"),
        ]:
            res = run_cmd(cmd, repo_path, timeout_s)
            repo.commands.append(res)
            if res.exit_code != 0:
                repo.add(
                    Finding(
                        severity="fail",
                        rule=rule,
                        message=f"Command failed: {res.cmd}",
                        recommendation="Inspect command output tail in report and fix.",
                        owner_repo=repo.name,
                    )
                )

        # Warn if 0 tests collected (best-effort: look for 'collected 0 items')
        py_res = next((c for c in repo.commands if c.cmd.endswith("pytest -q")), None)
        if py_res and "collected 0 items" in py_res.stdout_tail:
            repo.add(
                Finding(
                    severity="warn",
                    rule="tests.pytest.collected_zero",
                    message="pytest collected 0 tests",
                    recommendation="Add a minimal test suite or confirm intentional.",
                    owner_repo=repo.name,
                )
            )

    elif package_json.exists():
        node_modules = repo_path / "node_modules"
        if not node_modules.exists():
            repo.add(
                Finding(
                    severity="warn",
                    rule="node.deps.not_installed",
                    message="node_modules missing; skipping node lint/test to avoid surprise writes",
                    recommendation="Install deps manually if you want the audit to execute lint/test commands.",
                    owner_repo=repo.name,
                )
            )
            return

        # If deps already installed, we can run scripts without new writes.
        pkg = read_text(package_json) or ""
        has_lint = '"lint"' in pkg
        has_test = '"test"' in pkg

        if has_lint:
            res = run_cmd(["npm", "run", "lint"], repo_path, timeout_s)
            repo.commands.append(res)
            if res.exit_code != 0:
                repo.add(
                    Finding(
                        severity="fail",
                        rule="node.lint",
                        message="Node lint failed",
                        recommendation="Fix lint errors.",
                        owner_repo=repo.name,
                    )
                )
        else:
            repo.add(
                Finding(
                    severity="warn",
                    rule="node.lint.missing",
                    message="No lint script detected in package.json",
                    recommendation="Add a lint script (eslint + prettier) for maintainability.",
                    owner_repo=repo.name,
                )
            )

        if has_test:
            res = run_cmd(["npm", "run", "test"], repo_path, timeout_s)
            repo.commands.append(res)
            if res.exit_code != 0:
                repo.add(
                    Finding(
                        severity="fail",
                        rule="node.test",
                        message="Node tests failed",
                        recommendation="Fix test failures.",
                        owner_repo=repo.name,
                    )
                )
        else:
            repo.add(
                Finding(
                    severity="warn",
                    rule="node.test.missing",
                    message="No test script detected in package.json",
                    recommendation="Add a unit test runner (vitest/jest) to prevent drift.",
                    owner_repo=repo.name,
                )
            )

    else:
        repo.add(
            Finding(
                severity="warn",
                rule="repo.type.unknown",
                message="Repo type unknown (no pyproject.toml or package.json)",
                recommendation="Add minimal tooling configuration or mark repo archival explicitly.",
                owner_repo=repo.name,
            )
        )


def format_repo_report_md(repo: RepoReport) -> str:
    status = repo.status()
    status_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[status]

    lines: list[str] = []
    lines.append(f"# {repo.name} — {status_icon} {status.upper()}")
    lines.append("")
    lines.append(f"- **Path**: `{repo.path}`")
    if repo.unexpected_repo:
        lines.append("- **Unexpected repo**: yes (detected via qig-* heuristic)")
    lines.append("")

    if repo.toolchain:
        lines.append("## Toolchain (best-effort)")
        for k in sorted(repo.toolchain.keys()):
            lines.append(f"- **{k}**: `{repo.toolchain[k]}`")
        lines.append("")

    if repo.entry_points:
        lines.append("## Entry points (best-effort)")
        for ep in repo.entry_points:
            lines.append(f"- `{ep}`")
        lines.append("")

    lines.append("## Findings")
    if not repo.findings:
        lines.append("- ✅ No findings")
    else:
        for f in repo.findings:
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[f.severity]
            lines.append(f"- {icon} **{f.rule}**: {f.message}")
            if f.paths:
                for p in f.paths[:50]:
                    lines.append(f"  - `{p}`")
            if f.owner_repo:
                lines.append(f"  - **Owner repo**: `{f.owner_repo}`")
            if f.recommendation:
                lines.append(f"  - **Recommended fix**: {f.recommendation}")

    lines.append("")
    if repo.commands:
        lines.append("## Command results (tail)")
        for c in repo.commands:
            icon = "✅" if c.exit_code == 0 else "❌"
            lines.append(f"### {icon} `{c.cmd}`")
            lines.append(f"- **cwd**: `{c.cwd}`")
            lines.append(f"- **exit**: `{c.exit_code}`")
            lines.append(f"- **duration_s**: `{c.duration_s:.2f}`")
            if c.stdout_tail.strip():
                lines.append("```text")
                lines.append(c.stdout_tail)
                lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_summary_md(
    root: Path,
    repo_reports: list[RepoReport],
    workspace_findings: list[Finding],
) -> str:
    lines: list[str] = []
    lines.append("# QIG Constellation Audit — Summary")
    lines.append("")
    lines.append(f"- **Root**: `{root}`")
    lines.append("")

    unexpected = [r for r in repo_reports if r.unexpected_repo]
    if unexpected:
        lines.append("## Unexpected repos")
        for r in unexpected:
            lines.append(f"- `{r.name}`")
        lines.append("")

    lines.append("## Repo status")
    for r in repo_reports:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[r.status()]
        lines.append(f"- {icon} `{r.name}`")
    lines.append("")

    if workspace_findings:
        lines.append("## Workspace-level findings")
        for f in workspace_findings:
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[f.severity]
            lines.append(f"- {icon} **{f.rule}**: {f.message}")
            for p in f.paths[:50]:
                lines.append(f"  - `{p}`")
            if f.recommendation:
                lines.append(f"  - **Recommended fix**: {f.recommendation}")
        lines.append("")

    # Top 10 fails
    all_fails: list[Finding] = []
    for r in repo_reports:
        all_fails.extend([f for f in r.findings if f.severity == "fail"])
    all_fails.extend([f for f in workspace_findings if f.severity == "fail"])

    if all_fails:
        lines.append("## Top ❌ findings (first 10)")
        for f in all_fails[:10]:
            lines.append(f"- ❌ **{f.rule}**: {f.message}")
            for p in f.paths[:15]:
                lines.append(f"  - `{p}`")
        lines.append("")

    # Import-direction fails
    import_fails = [f for f in workspace_findings if f.rule == "governance.import_direction"]
    if import_fails:
        lines.append("## Import-direction violations")
        for f in import_fails:
            lines.append(f"- ❌ {f.message}")
            for p in f.paths[:20]:
                lines.append(f"  - `{p}`")
        lines.append("")

    # DRY collisions
    dry = [f for f in workspace_findings if f.rule.startswith("dry.")]
    if dry:
        lines.append("## DRY collisions")
        for f in dry:
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[f.severity]
            lines.append(f"- {icon} **{f.rule}**: {f.message}")
            for p in f.paths[:25]:
                lines.append(f"  - `{p}`")
            if f.owner_repo:
                lines.append(f"  - **Canonical home**: `{f.owner_repo}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def ensure_reports_dir(root: Path) -> Path:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def main() -> int:
    args = _parse_args()

    root = Path(args.root).resolve()
    reports_dir = ensure_reports_dir(root)

    repo_reports = iter_repo_dirs(root)
    repo_by_name = {r.name: r for r in repo_reports}

    # Mark canonical repos missing
    for expected in CANONICAL_REPOS:
        if expected not in repo_by_name:
            # Summary-only finding
            pass

    _run_repo_checks_all(repo_reports, args)

    config: AuditConfig = {}
    config_findings: list[Finding] = []
    try:
        config_path = _resolve_config_path(str(args.config))
        config = load_audit_config(config_path)
    except (OSError, ValueError, RuntimeError) as exc:
        config_findings.append(
            Finding(
                severity="fail",
                rule="runner.config",
                message=f"Failed to load audit config: {exc}",
                recommendation="Fix tools/qig_audit_config.yaml (or pass --config) so governance/DRY checks are reliable.",
                owner_repo="qig-consciousness",
            )
        )

    for repo in repo_reports:
        (reports_dir / f"{repo.name}.md").write_text(format_repo_report_md(repo), encoding="utf-8")

    workspace_findings = config_findings + _run_workspace_scans(root, repo_by_name, config)

    (reports_dir / "summary.md").write_text(
        format_summary_md(root, repo_reports, workspace_findings), encoding="utf-8"
    )

    any_fail = any(r.status() == "fail" for r in repo_reports) or any(
        f.severity == "fail" for f in workspace_findings
    )

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
