PYTHON = python3
SCRIPT = turek_traction_l2_error.py

all: run_stokes run_navier

run_stokes:
	STOKES=True MAX_REF=5 ORDER=2 $(PYTHON) $(SCRIPT)

run_navier:
	STOKES=False MAX_REF=5 ORDER=2 $(PYTHON) $(SCRIPT)
