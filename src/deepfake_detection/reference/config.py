
import torch

from deepfake_detection.config.config import _project_root

# ── Dataset paths ──────────────────────────────────────────────────────────────
BASE             = _project_root()  # datasets/<name>/embedding and models/<run> live under this root (DEEPFAKE_PROJECT_ROOT)
POLYGLOT_ROOT    = BASE / "datasets" / "polyglotfake" / "embedding"
HIDF_ROOT        = BASE / "datasets" / "hidf" / "embedding"
CELEBDFV2_ROOT   = BASE / "datasets" / "celebdfv2" / "embedding"
FAKEAVCELEB_ROOT = BASE / "datasets" / "fakeavceleb" / "embedding"
FFPP_ROOT        = BASE / "datasets" / "ffpp" / "embedding"
SAVE_BASE        = BASE / "models"

# ── Datasets where audio is NOT a manipulation signal ──────────────────────────
# MoE gate is forced to α=1.0 (visual expert); audio features zeroed on load.
AUDIO_UNAVAILABLE_DATASETS = {"celebdfv2", "ffpp"}

# ── Feature dimensions ─────────────────────────────────────────────────────────
AUDIO_DIM = 1024   # Whisper encoder (medium) output dim
VIDEO_DIM = 768    # VideoMAE-base last_hidden_state mean-pooled dim

# ── Model architecture ─────────────────────────────────────────────────────────
HIDDEN_DIM  = 256   # shared projection / GRU hidden size
GRU_LAYERS  = 2
DROPOUT     = 0.3
GATE_HIDDEN = 64    # ModalityReliabilityGate MLP hidden size

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS          = 60
BATCH_SIZE      = 16
LR              = 1e-4
WEIGHT_DECAY    = 1e-3
PATIENCE        = 12
VAL_RATIO       = 0.15
SEED            = 42
LABEL_SMOOTHING = 0.05
MODAL_DROP_P    = 0.25   # probability of zeroing one modality per sample
FEAT_NOISE      = 0.01   # Gaussian noise std added to features during training
MIXUP_ALPHA     = 0.2    # mixup beta-distribution alpha; 0 disables mixup

# ── Loss & DANN regularisation ─────────────────────────────────────────────────
DANN_LAMBDA            = 0.1    # domain-adversarial loss weight
FOCAL_GAMMA            = 1.0    # focal loss focusing parameter
FOCAL_ALPHA            = 0.75   # focal loss weight on the fake class (class 1)
GATE_SUPERVISION_WEIGHT = 0.1   # weight on soft-BCE gate supervision auxiliary loss

# ── Stochastic Weight Averaging ────────────────────────────────────────────────
SWA_START = 0.75   # fraction of EPOCHS after which SWA begins
SWA_LR    = 5e-5

# ── Debug ──────────────────────────────────────────────────────────────────────
SANITY_FRACTION = None   # e.g. 0.05 to run a quick smoke-test on 5 % of data

# ── Performance defaults ─────────────────────────────────────────────────────
# Conservative RTX-class GPU defaults. These can be overridden from YAML or CLI
# under the "performance" section without changing experiment protocols.
AMP                  = True
AMP_DTYPE            = "float16"
TORCH_COMPILE        = False
NUM_WORKERS          = 8
PIN_MEMORY           = True
PERSISTENT_WORKERS   = True
PREFETCH_FACTOR      = 4
CACHE_SPLITS         = True
CACHE_NORM_STATS     = True
CACHE_DIR            = BASE / "temp" / "training_cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
