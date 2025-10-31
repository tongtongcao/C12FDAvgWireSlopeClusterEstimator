#!/bin/tcsh
# ==============================
# JLab Python venv + PyTorch
# CPU-only vs GPU(CUDA 12.4.1)
# Includes Lightning, matplotlib, scipy
# ==============================

# === User options ===
set use_cuda = 0       # 1 = use CUDA, 0 = CPU only

if ( $use_cuda == 1 ) then
    set venv_dir = /work/clas12/users/caot/ai/venvs/torch-cu124
else
    set venv_dir = /work/clas12/users/caot/ai/venvs/torch-cpu
endif

# === Load modules ===
if ( $use_cuda == 1 ) then
    echo "🔹 Loading CUDA 12.4.1..."
	module unload cuda
	module load cuda/12.4.1
else
    echo "🔹 CPU only mode (no CUDA module loaded)"
endif

# Python module
module unload pymods
module load pymods/3.9

# === Create or reuse venv ===
if ( ! -d $venv_dir ) then
    echo "🔹 Creating new venv at $venv_dir"
    python -m venv $venv_dir
else
    echo "🔹 Reusing existing venv at $venv_dir"
endif

source $venv_dir/bin/activate.csh

# === Upgrade pip ===
pip install --upgrade pip

# === Install PyTorch ===
if ( $use_cuda == 1 ) then
    echo "🔹 Installing PyTorch with CUDA 12.4 support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
	pip install torch-geometric
else
    echo "🔹 Installing CPU-only PyTorch"
    pip install torch torchvision torchaudio
	pip install torch-geometric
endif

# === Install extra packages ===
pip install lightning matplotlib scipy
pip install scikit-learn networkx networkx
pip install pytz

# === Verify installation ===
python - << 'EOF'
import torch, lightning, matplotlib, scipy
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "Available:", torch.cuda.is_available())
print("Lightning:", lightning.__version__)
print("matplotlib:", matplotlib.__version__)
print("scipy:", scipy.__version__)
EOF

