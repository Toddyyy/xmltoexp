#!/usr/bin/env python3
"""
Setup script to integrate MIREX tokenizer/detokenizer into your repo.
This script helps configure the MIREX submodule for seamless integration.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def check_mirex_directory():
    """Check if MIREX directory exists and has required structure."""
    mirex_path = Path("MIREX")

    if not mirex_path.exists():
        print("❌ MIREX directory not found")
        return False

    required_dirs = ["tokenizer", "detokenizer"]
    required_files = [
        "tokenizer/base_score_tokenizer.py",
        "detokenizer/detokenizer.py"
    ]

    missing_items = []

    for dir_name in required_dirs:
        if not (mirex_path / dir_name).exists():
            missing_items.append(f"Directory: {dir_name}")

    for file_path in required_files:
        if not (mirex_path / file_path).exists():
            missing_items.append(f"File: {file_path}")

    if missing_items:
        print("❌ Missing MIREX components:")
        for item in missing_items:
            print(f"   • {item}")
        return False

    print("✅ MIREX directory structure is valid")
    return True


def install_mirex_requirements():
    """Install MIREX requirements if requirements.txt exists."""
    req_file = Path("MIREX_Tokenizer/requirements.txt")

    if not req_file.exists():
        print("⚠️ MIREX_Tokenizer/requirements.txt not found, skipping dependency installation")
        return True

    print("📦 Installing MIREX requirements...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(req_file)
        ], check=True)
        print("✅ MIREX requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install MIREX requirements: {e}")
        return False


def create_mirex_init_files():
    """Create __init__.py files in MIREX subdirectories if they don't exist."""
    init_files = [
        "MIREX_Tokenizer/__init__.py",
        "MIREX_Tokenizer/tokenizer/__init__.py",
        "MIREX_Tokenizer/detokenizer/__init__.py"
    ]

    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.touch()
            print(f"✅ Created {init_file}")


def test_mirex_imports():
    """Test if MIREX modules can be imported successfully."""
    print("🧪 Testing MIREX imports...")

    # Add MIREX to path temporarily
    mirex_path = Path("MIREX").resolve()
    if str(mirex_path) not in sys.path:
        sys.path.insert(0, str(mirex_path))

    try:
        from tokenizer.base_score_tokenizer import ScoreNoteToken
        print("✅ ScoreNoteToken import successful")
    except ImportError as e:
        print(f"❌ Failed to import ScoreNoteToken: {e}")
        return False

    try:
        from detokenizer.detokenizer import RenderingPerformanceNoteToken, detokenize_to_midi
        print("✅ Detokenizer imports successful")
    except ImportError as e:
        print(f"❌ Failed to import detokenizer modules: {e}")
        return False

    print("✅ All MIREX imports working correctly")
    return True


def setup_environment_script():
    """Create a script to set up environment variables."""

    # Create bash script for Unix systems
    bash_script_content = """#!/bin/bash
# Environment setup for MIREX integration

# Add MIREX to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/MIREX"

echo "Environment configured for MIREX integration"
echo "   PYTHONPATH includes: $(pwd)/MIREX"

# Run your command
exec "$@"
"""

    # Create batch script for Windows
    batch_script_content = """@echo off
REM Environment setup for MIREX integration

REM Add MIREX to Python path
set PYTHONPATH=%PYTHONPATH%;%CD%\\MIREX

echo Environment configured for MIREX integration
echo    PYTHONPATH includes: %CD%\\MIREX

REM Run your command
%*
"""

    # Create appropriate script based on OS
    if os.name == 'nt':  # Windows
        script_path = Path("run_with_mirex.bat")
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(batch_script_content)
            print(f"Created environment script: {script_path}")
            print("   Usage: run_with_mirex.bat python generate_from_xml.py [args]")
        except UnicodeEncodeError:
            # Fallback to ASCII only
            with open(script_path, 'w', encoding='ascii', errors='replace') as f:
                f.write(batch_script_content)
            print(f"Created environment script: {script_path} (ASCII mode)")
    else:  # Unix/Linux/Mac
        script_path = Path("run_with_mirex.sh")
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(bash_script_content)
            os.chmod(script_path, 0o755)
            print(f"Created environment script: {script_path}")
            print("   Usage: ./run_with_mirex.sh python generate_from_xml.py [args]")
        except UnicodeEncodeError:
            # Fallback to ASCII only
            with open(script_path, 'w', encoding='ascii', errors='replace') as f:
                f.write(bash_script_content)
            os.chmod(script_path, 0o755)
            print(f"Created environment script: {script_path} (ASCII mode)")


def main():
    print("🔧 MIREX Integration Setup")
    print("=" * 50)

    # Check current directory
    if not Path("generate_from_xml.py").exists():
        print("❌ Please run this script from your project root directory")
        print("   (The directory containing generate_from_xml.py)")
        return 1

    # Step 1: Check MIREX directory
    print("\n1. Checking MIREX directory...")
    if not check_mirex_directory():
        print("\n💡 To fix this:")
        print("   • Copy the MIREX directory to your project root")
        print("   • Ensure it contains tokenizer/ and detokenizer/ subdirectories")
        return 1

    # Step 2: Create __init__.py files
    print("\n2. Creating Python package files...")
    create_mirex_init_files()

    # Step 3: Install requirements
    print("\n3. Installing requirements...")
    if not install_mirex_requirements():
        print("⚠️ Requirements installation failed, but continuing...")

    # Step 4: Test imports
    print("\n4. Testing imports...")
    if not test_mirex_imports():
        print("\n💡 Import test failed. Possible solutions:")
        print("   • Check that all required MIREX files are present")
        print("   • Install missing Python dependencies")
        print("   • Verify MIREX code compatibility")
        return 1

    # Step 5: Create helper scripts
    print("\n5. Creating helper scripts...")
    setup_environment_script()

    print("\n" + "=" * 50)
    print("MIREX integration setup complete!")
    print("\nUsage:")
    print("   # Method 1: Direct execution (recommended)")
    print("   python generate_from_xml.py [args]")
    print("")
    if os.name == 'nt':
        print("   # Method 2: Using environment script")
        print("   run_with_mirex.bat python generate_from_xml.py [args]")
        print("")
        print("   # Method 3: Manual environment setup")
        print("   set PYTHONPATH=%PYTHONPATH%;%CD%\\MIREX")
    else:
        print("   # Method 2: Using environment script")
        print("   ./run_with_mirex.sh python generate_from_xml.py [args]")
        print("")
        print("   # Method 3: Manual environment setup")
        print("   export PYTHONPATH=\"$PYTHONPATH:$(pwd)/MIREX\"")
    print("   python generate_from_xml.py [args]")

    return 0


if __name__ == "__main__":
    exit(main())