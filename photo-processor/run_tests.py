#!/usr/bin/env python3
"""
Test runner script for AI Photo Processor
"""
import argparse
import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install -r test-requirements.txt")
        return False


def run_unit_tests(verbose=False, coverage=True):
    """Run unit tests"""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ])
    
    cmd.append("-m")
    cmd.append("unit")
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests"""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    cmd.append("-m")
    cmd.append("integration")
    
    return run_command(cmd, "Integration Tests")


def run_schema_tests(verbose=False):
    """Run schema validation tests"""
    cmd = ["python", "-m", "pytest", "tests/unit/test_schemas.py"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, "Schema Validation Tests")


def run_all_tests(verbose=False, coverage=True):
    """Run all tests"""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    return run_command(cmd, "All Tests")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function"""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, f"Specific Test: {test_path}")


def run_tests_with_markers(markers, verbose=False):
    """Run tests with specific markers"""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    for marker in markers:
        cmd.extend(["-m", marker])
    
    return run_command(cmd, f"Tests with markers: {', '.join(markers)}")


def lint_code():
    """Run code linting"""
    print("\n" + "="*60)
    print("Running Code Quality Checks")
    print("="*60)
    
    # Check if files exist
    python_files = [
        "ai_analyzer.py",
        "image_processor.py", 
        "immich_client.py",
        "schemas.py",
        "main.py"
    ]
    
    existing_files = [f for f in python_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No Python files found to lint")
        return False
    
    # Run flake8 if available
    try:
        cmd = ["python", "-m", "flake8"] + existing_files + ["--max-line-length=100"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Code style check passed")
            return True
        else:
            print("❌ Code style issues found:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("⚠️  flake8 not found, skipping code style check")
        print("Install with: pip install flake8")
        return True


def check_test_coverage():
    """Check test coverage requirements"""
    print("\n" + "="*60)
    print("Checking Test Coverage")
    print("="*60)
    
    # Run tests with coverage
    cmd = [
        "python", "-m", "pytest", 
        "--cov=.", 
        "--cov-report=term-missing",
        "--cov-fail-under=90",
        "--quiet"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Test coverage meets requirements (≥90%)")
            return True
        else:
            print("❌ Test coverage below requirements:")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ Error checking coverage: {e}")
        return False


def install_test_dependencies():
    """Install test dependencies"""
    cmd = ["pip", "install", "-r", "test-requirements.txt"]
    return run_command(cmd, "Installing Test Dependencies")


def main():
    parser = argparse.ArgumentParser(description="AI Photo Processor Test Runner")
    parser.add_argument(
        "test_type", 
        choices=["unit", "integration", "schema", "all", "specific", "markers", "lint", "coverage", "install"],
        help="Type of tests to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--path", help="Specific test path (for 'specific' test type)")
    parser.add_argument("--markers", nargs="+", help="Test markers to run (for 'markers' test type)")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = True
    
    if args.test_type == "install":
        success = install_test_dependencies()
    
    elif args.test_type == "unit":
        success = run_unit_tests(args.verbose, not args.no_coverage)
    
    elif args.test_type == "integration":
        success = run_integration_tests(args.verbose)
    
    elif args.test_type == "schema":
        success = run_schema_tests(args.verbose)
    
    elif args.test_type == "all":
        success = run_all_tests(args.verbose, not args.no_coverage)
        if success:
            success = lint_code()
    
    elif args.test_type == "specific":
        if not args.path:
            print("❌ --path is required for specific tests")
            sys.exit(1)
        success = run_specific_test(args.path, args.verbose)
    
    elif args.test_type == "markers":
        if not args.markers:
            print("❌ --markers is required for marker-based tests")
            sys.exit(1)
        success = run_tests_with_markers(args.markers, args.verbose)
    
    elif args.test_type == "lint":
        success = lint_code()
    
    elif args.test_type == "coverage":
        success = check_test_coverage()
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("🎉 All tests completed successfully!")
        print("\nTest Coverage Report: htmlcov/index.html")
    else:
        print("💥 Some tests failed!")
        print("\nCheck the output above for details.")
    print("="*60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()