#!/bin/bash
# ============================================================================
# Script central : execute le pipeline des NOUVEAUX modeles legers
# (LEAR, XGBoost, DNN, DLinear, CNN-LSTM) uniquement.
# Pas de "set -e" : un echec sur une etape ne bloque pas la suivante.
# ============================================================================

log() {
    echo ""
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "=================================================================="
}

run_step() {
    local description="$1"
    local script="$2"
    log "$description"
    uv run "$script"
    if [ $? -ne 0 ]; then
        echo "!!! ECHEC : $script -- on continue avec l'etape suivante"
    else
        echo ">>> OK : $script"
    fi
}

# ----------------------------------------------------------------------------
# 1. Recherche Optuna pour les 5 modeles legers (missing_ratio=0.0)
# ----------------------------------------------------------------------------
run_step "Recherche Optuna modeles legers (5 modeles x 5 datasets)" "optuna_search_light_models.py"

# ----------------------------------------------------------------------------
# 2. Benchmark complet (5 modeles x 5 datasets x 6 ratios x 5 seeds)
# ----------------------------------------------------------------------------
run_step "Benchmark modeles legers (5 modeles x 5 datasets x 6 ratios x 5 seeds)" "run_light_models_benchmark.py"

log "PIPELINE MODELES LEGERS TERMINE"