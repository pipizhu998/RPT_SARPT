from __future__ import annotations

from dataclasses import dataclass

from baseline.adabn import adapt_batch_norm, evaluate_model_per_batch_adabn
from baseline.cotta import evaluate_model_cotta
from baseline.eata import evaluate_model_eata
from baseline.tent import evaluate_model_tent
from rpt_sarpt import evaluate_model_rpt


@dataclass
class TestAdaptConfig:
    method: str = "none"
    adabn_reset_stats: bool = True
    adabn_momentum: float | None = None
    adapt_max_batches: int | None = None
    eata_lr: float = 2.5e-4
    eata_steps: int = 1
    eata_optimizer: str = "sgd"
    eata_weight_decay: float = 0.0
    eata_episodic: bool = False
    eata_e_margin: float = 0.9210340371976183
    eata_d_margin: float = 0.05
    eata_fisher_size: int = 2000
    eata_fisher_alpha: float = 2000.0
    eata_fisher_clip_by_norm: float | None = None
    cotta_lr: float = 1e-3
    cotta_steps: int = 1
    cotta_optimizer: str = "adam"
    cotta_weight_decay: float = 0.0
    cotta_episodic: bool = False
    cotta_mt_alpha: float = 0.999
    cotta_rst_m: float = 0.01
    cotta_ap: float = 0.92
    cotta_augmentation_views: int = 32
    cotta_gaussian_std: float = 0.005
    cotta_soft_augmentations: bool = False
    cotta_beta: float = 0.9
    tent_lr: float = 1e-3
    tent_steps: int = 1
    tent_optimizer: str = "adam"
    tent_weight_decay: float = 0.0
    tent_episodic: bool = False
    rpt_lr: float = 1e-3
    rpt_steps: int = 1
    rpt_optimizer: str = "adam"
    rpt_weight_decay: float = 0.0
    rpt_episodic: bool = False
    rpt_jsd_weight: float = 0.1
    rpt_augmix_severity: int = 3
    rpt_augmix_width: int = 3
    rpt_augmix_depth: int = -1
    rpt_augmix_alpha: float = 1.0
    rpt_augmix_all_ops: bool = True
    rpt_source_anchor_weight: float = 0.01
    rpt_precompute_augmix: bool = True
    rpt_augmix_cache_dir: str | None = "outputs/cache/ra_rpt_augmix"
    rpt_rebuild_augmix_cache: bool = False
    sweep_enabled: bool = False
    sweep_name: str | None = None
    rpt_lr_values: list[float] | None = None
    rpt_jsd_weight_values: list[float] | None = None
    rpt_source_anchor_weight_values: list[float] | None = None
