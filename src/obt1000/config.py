from dataclasses import dataclass

@dataclass
class Paths:
    mat_path: str = "data/300fps_15k.mat"
    mat_key: str | None = None  # auto-detect a 1D vector if None
    out_dir: str = "out"

@dataclass
class Sampling:
    fps_in: float = 300.0
    fps_out: float = 1000.0
    px_to_nm: float = 35.0

@dataclass
class OU:
    # safeguards; estimates will overwrite these
    min_a: float = 1e-6
    max_a: float = 0.999999

@dataclass
class Train:
    epochs: int = 20
    batch_size: int = 4
    T: int = 15000     # sequence length for synthetic data
    lr: float = 2e-4
    save_path: str = "out/pgcdi.pt"

@dataclass
class Config:
    paths: Paths = Paths()
    sampling: Sampling = Sampling()
    ou: OU = OU()
    train: Train = Train()

cfg = Config()
