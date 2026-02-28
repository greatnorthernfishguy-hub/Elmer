#!/usr/bin/env bash
# Elmer — Install and register with ET Module Manager
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# What it does:
#   1. Installs Python dependencies
#   2. Registers Elmer with the ET Module Manager
#   3. Creates data directories
#   4. Verifies the installation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ID="elmer"
DATA_DIR="${HOME}/.elmer"
ET_MODULES_DIR="${HOME}/.et_modules"

echo "=== Elmer Installer ==="
echo "Install directory: ${SCRIPT_DIR}"
echo ""

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
if command -v pip3 &>/dev/null; then
    pip3 install -r "${SCRIPT_DIR}/requirements.txt" --quiet
elif command -v pip &>/dev/null; then
    pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet
else
    echo "WARNING: pip not found. Install dependencies manually."
fi

# 2. Create data directories
echo "[2/4] Creating data directories..."
mkdir -p "${DATA_DIR}"
mkdir -p "${ET_MODULES_DIR}/shared_learning"

# 3. Register with ET Module Manager
echo "[3/4] Registering with ET Module Manager..."
python3 -c "
import sys, json
sys.path.insert(0, '${SCRIPT_DIR}')
from et_modules.manager import ETModuleManager, ModuleManifest

manifest = ModuleManifest.from_file('${SCRIPT_DIR}/et_module.json')
if manifest:
    manifest.install_path = '${SCRIPT_DIR}'
    manager = ETModuleManager()
    manager.register(manifest)
    print(f'  Registered: {manifest.module_id} v{manifest.version}')
else:
    print('  WARNING: Could not load et_module.json')
" 2>/dev/null || echo "  Standalone mode (ET Module Manager unavailable)"

# 4. Verify
echo "[4/4] Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from ng_ecosystem import SubstrateSignal, SignalType
from core.config import load_config
from core.base_socket import ElmerSocket

cfg = load_config('${SCRIPT_DIR}/config.yaml')
sig = SubstrateSignal.create(
    source_socket='installer',
    signal_type=SignalType.HEALTH,
    payload={'status': 'installed'},
)
print(f'  Config loaded: {cfg.module_id} v{cfg.version}')
print(f'  SubstrateSignal: {sig.signal_id[:8]}... ({sig.signal_type.value})')
print(f'  All imports OK')
"

echo ""
echo "=== Elmer installed successfully ==="
echo "  Run tests:  cd ${SCRIPT_DIR} && python3 -m pytest tests/ -v"
echo "  Start hook: python3 -c \"from elmer_hook import ElmerHook; h = ElmerHook(); h.start()\""
