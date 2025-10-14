import json
import os

class Config:
    def __init__(self):
        self.cfg_dict = {}

    def update_from_file(self, cfg_file):
        if not os.path.exists(cfg_file):
            raise FileNotFoundError(f"Config file not found: {cfg_file}")
        with open(cfg_file, 'r') as f:
            self.cfg_dict = json.load(f)
        print(f"[INFO] Config loaded from {cfg_file}")

    def __getattr__(self, name):
        # supaya bisa diakses seperti cfg.MODEL atau cfg.DATASET
        if name in self.cfg_dict:
            return self.cfg_dict[name]
        raise AttributeError(f"No such attribute: {name}")

cfg = Config()

def update_config_from_file(cfg_file):
    cfg.update_from_file(cfg_file)
