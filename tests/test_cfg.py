"""Config loader smoke test."""

from utils.cfg import load_config


def test_load_config_core_keys():
    cfg = load_config()
    # Spot-check the sections we rely on everywhere.
    assert cfg.unet.num_classes == 6
    assert len(cfg.unet.class_info) == 6
    assert cfg.yolo.model.endswith(".pt")
    assert cfg.device.hsa_override_gfx_version == "10.3.0"
    assert cfg.inference.mask_alpha > 0.0


def test_attribute_and_dict_access():
    cfg = load_config()
    assert cfg["unet"]["lr"] == cfg.unet.lr
    assert cfg["paths"]["results_dir"] == cfg.paths.results_dir
