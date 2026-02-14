#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

VERIFICATION_DIR = Path("verification")

TLA2TOOLS_JAR = os.getenv("TLA2TOOLS_JAR", "")
VIPER_SILICON_PATH = os.getenv("VIPER_SILICON_PATH", "silicon")

PROVERS = {
    "agda": {
        "ext": ".agda",
        "cmd_builder": lambda file_path: (["agda", "--safe", str(file_path)], None),
        "placeholders": ["TODO:"]
    },
    "lean4": {
        "ext": ".lean",
        "cmd_builder": lambda file_path: (
            (["lake", "exe", "cache", "warmup"], file_path.parent.parent)
            if (file_path.parent.parent / "lakefile.lean").exists()
            else (None, None)
        ),
        "placeholders": ["TODO:"]
    },
    "isabelle": {
        "ext": ".thy",
        "cmd_builder": lambda file_path: (
            (["isabelle", "build", "-d", ".", "SemanticModels"], file_path.parent)
            if (file_path.parent / "ROOT").exists()
            else (None, None)
        ),
        "placeholders": ["TODO:"]
    },
    "viper": {
        "ext": ".vpr",
        "cmd_builder": lambda file_path: (
            ([VIPER_SILICON_PATH, "--z3Exe", "z3", str(file_path)], None)
            if VIPER_SILICON_PATH
            else (None, None)
        ),
        "placeholders": ["TODO:"]
    },
    "tlaplus": {
        "ext": ".tla",
        "cmd_builder": lambda file_path: (
            (["java", "-jar", TLA2TOOLS_JAR, "-deadlock", "-cleanup", str(file_path)], None)
            if TLA2TOOLS_JAR and Path(TLA2TOOLS_JAR).exists()
            else (None, None)
        ),
        "placeholders": ["TODO:"]
    },
    "spin": {
        "ext": ".pml",
        "cmd_builder": lambda file_path: (
            ["spin", "-a", str(file_path)], file_path.parent
        ),
        "placeholders": ["TODO:"]
    }
}

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'

def find_verification_files(prover: str) -> List[Path]:
    semantics_dir = VERIFICATION_DIR / "semantics"
    prover_dir = VERIFICATION_DIR / prover

    files = []
    ext = PROVERS[prover]["ext"]

    if semantics_dir.exists():
        files.extend(list(semantics_dir.glob(f"*{ext}")))

    if prover_dir.exists():
        files.extend(list(prover_dir.glob(f"*{ext}")))

    return files

def check_placeholders(file_path: Path, placeholders: List[str]) -> Tuple[bool, int]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        count = 0
        for placeholder in placeholders:
            count += content.count(placeholder)

        return count == 0, count
    except Exception as e:
        return False, 0

def type_check_file(file_path: Path, prover: str) -> Tuple[bool, str]:
    cmd_builder = PROVERS[prover]["cmd_builder"]
    result_tuple = cmd_builder(file_path)

    if result_tuple[0] is None:
        return False, f"Prover not configured (missing project config or tool)"

    full_cmd, cwd = result_tuple

    try:
        result = subprocess.run(
            full_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return True, "Type-check passed"
        else:
            return False, f"Type-check failed:\n{result.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return False, "Type-check timed out (5 minutes)"
    except FileNotFoundError:
        return False, f"Prover command not found (tool may not be installed): {full_cmd[0]}"
    except Exception as e:
        return False, f"Error running prover: {str(e)}"

def main():
    print("=== JAIDE v40 Formal Proof Type-Checking ===\n")
    print(f"Configuration:")
    print(f"  TLA2TOOLS_JAR={TLA2TOOLS_JAR or '(not set)'}")
    print(f"  VIPER_SILICON_PATH={VIPER_SILICON_PATH}")
    print()

    total_files = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_placeholders = 0

    for prover, config in PROVERS.items():
        print(f"Checking {prover.upper()} proofs...")

        files = find_verification_files(prover)
        if not files:
            print(f"  {Colors.YELLOW}[SKIP]{Colors.NC} No {prover} files found\n")
            continue

        for file_path in files:
            total_files += 1

            no_placeholders, ph_count = check_placeholders(file_path, config["placeholders"])
            total_placeholders += ph_count

            if not no_placeholders:
                print(f"  {Colors.RED}[FAIL]{Colors.NC} {file_path.name} - {ph_count} TODO: markers found")
                total_failed += 1
                continue

            type_ok, msg = type_check_file(file_path, prover)

            if type_ok:
                print(f"  {Colors.GREEN}[PASS]{Colors.NC} {file_path.name}")
                total_passed += 1
            elif "not configured" in msg or "not found" in msg:
                print(f"  {Colors.YELLOW}[SKIP]{Colors.NC} {file_path.name} - {msg}")
                total_skipped += 1
            else:
                print(f"  {Colors.RED}[FAIL]{Colors.NC} {file_path.name}")
                print(f"    {msg}")
                total_failed += 1

        print()

    print("=== Summary ===")
    print(f"Total files: {total_files}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Skipped (tool not configured): {total_skipped}")
    print(f"Total TODO: placeholders: {total_placeholders} (target: 0)")
    print()

    if total_placeholders > 0:
        print(f"{Colors.RED}[FAIL]{Colors.NC} TODO: placeholder markers detected")
        sys.exit(1)

    if total_failed > 0:
        print(f"{Colors.RED}[FAIL]{Colors.NC} Some proofs failed type-checking")
        sys.exit(1)

    if total_skipped > 0:
        print(f"{Colors.YELLOW}[WARN]{Colors.NC} Some proofs skipped (tools not installed)")

    print(f"{Colors.GREEN}[PASS]{Colors.NC} All active proofs type-check successfully")
    sys.exit(0)

if __name__ == "__main__":
    main()