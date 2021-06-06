import sys
import yaml
from omegaconf import DictConfig, OmegaConf


f = open('E:\\cldc\\config\\config.yaml', 'r')
cfg = OmegaConf.create(yaml.safe_load(f))
f.close()

if __name__ == "__main__":
    print(cfg)
    print(type(list(cfg.model.fl_alpha)))