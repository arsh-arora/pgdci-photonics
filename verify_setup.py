#!/usr/bin/env python
"""Verification script to check repository setup and imports."""

import sys
import importlib

def test_imports():
    """Test all module imports."""
    modules = [
        "src.obt1000.config",
        "src.obt1000.io_prep",
        "src.obt1000.baselines",
        "src.obt1000.ou_kalman",
        "src.obt1000.eval_metrics",
        "src.obt1000.plotting",
        "src.obt1000.utils",
        "src.pgcdi.data",
        "src.pgcdi.model",
        "src.pgcdi.physics_losses",
        "src.pgcdi.scheduler",
        "src.pgcdi.train",
        "src.pgcdi.sample",
    ]

    print("Testing module imports...")
    failed = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            failed.append((module_name, str(e)))

    if failed:
        print(f"\n❌ {len(failed)} modules failed to import")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return False
    else:
        print(f"\n✅ All {len(modules)} modules imported successfully!")
        return True

def check_dependencies():
    """Check if required packages are available."""
    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib.pyplot",
        "tqdm": "tqdm",
        "torch": "torch",
    }

    print("\nChecking dependencies...")
    missing = []
    for name, import_name in packages.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name:12s} {version}")
        except ImportError:
            print(f"✗ {name:12s} NOT INSTALLED")
            missing.append(name)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -e .")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def check_structure():
    """Check directory structure."""
    import os

    print("\nChecking directory structure...")
    required = [
        "data",
        "out",
        "src/obt1000",
        "src/pgcdi",
        "scripts",
        "requirements.txt",
        "pyproject.toml",
        "Makefile",
        "README.md",
    ]

    missing = []
    for item in required:
        if os.path.exists(item):
            print(f"✓ {item}")
        else:
            print(f"✗ {item}")
            missing.append(item)

    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        return False
    else:
        print("\n✅ Directory structure complete!")
        return True

def main():
    print("=" * 60)
    print("Optical Bead Trajectory 1000fps - Repository Verification")
    print("=" * 60)

    results = []
    results.append(("Structure", check_structure()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Imports", test_imports()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s} {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All checks passed! Repository is ready to use.")
        print("\nNext steps:")
        print("  1. Place your .mat file in data/")
        print("  2. Update src/obt1000/config.py if needed")
        print("  3. Run: make baselines")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
