#!/usr/bin/env python3
"""
Test runner for the enhanced photo processor
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run all tests and generate coverage report"""
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=" * 60)
    print("Running Enhanced Photo Processor Tests")
    print("=" * 60)
    
    # Test commands
    test_commands = [
        # Unit tests
        {
            'name': 'Unit Tests - Recipe Storage',
            'cmd': ['pytest', 'tests/unit/test_recipe_storage.py', '-v', '--tb=short']
        },
        {
            'name': 'Unit Tests - Enhanced Immich Client',
            'cmd': ['pytest', 'tests/unit/test_immich_client_v2.py', '-v', '--tb=short']
        },
        {
            'name': 'Unit Tests - Main Processor',
            'cmd': ['pytest', 'tests/unit/test_main_v2.py', '-v', '--tb=short']
        },
        # Integration tests
        {
            'name': 'Integration Tests - Dual Upload',
            'cmd': ['pytest', 'tests/integration/test_dual_upload_integration.py', '-v', '--tb=short']
        },
        # All tests with coverage
        {
            'name': 'All Tests with Coverage',
            'cmd': [
                'pytest',
                'tests/',
                '--cov=.',
                '--cov-report=html',
                '--cov-report=term-missing',
                '-v'
            ]
        }
    ]
    
    failed_tests = []
    
    for test_info in test_commands:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_info['name']}")
        print('=' * 60)
        
        try:
            result = subprocess.run(
                test_info['cmd'],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {test_info['name']}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            failed_tests.append(test_info['name'])
        
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if failed_tests:
        print(f"❌ {len(failed_tests)} test suite(s) failed:")
        for test in failed_tests:
            print(f"   - {test}")
        return 1
    else:
        print("✅ All tests passed!")
        print("\nCoverage report generated in htmlcov/index.html")
        return 0

if __name__ == "__main__":
    sys.exit(run_tests())