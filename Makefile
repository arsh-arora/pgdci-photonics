PY := python

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .

baselines:
	$(PY) scripts/run_baselines.py

ou:
	$(PY) scripts/run_ou.py

train:
	$(PY) scripts/run_train_pgcdi.py

sample:
	$(PY) scripts/run_sample_pgcdi.py

eval:
	$(PY) scripts/run_eval_all.py

all: baselines ou train sample eval
