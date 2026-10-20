# rover_sim — the per-stack sim (docker/compose.yaml), driven by scripts/sim.py.
#
# make images      each stack's image from its own repo's Dockerfile at the docker/stacks.env
#                  pins (scripts/build_images.sh --help). STACKS="mighty dlio" picks stacks,
#                  FORCE=1 rebuilds, REUSE="mighty=mighty-hw:main-3cf7320 ..." retags an
#                  existing image instead, JOBS=N caps colcon in our own targets, PARALLEL=N
#                  builds N stacks at once. SRC="mighty=~/code/mighty ..." builds a local
#                  checkout into the stack's tag (sim.py warns while it is in use).
# make up          the sim-* containers, idle (sessions docker-exec their nodes into them);
#                  GPU=1 adds sim-elevation (compose profile gpu)
# make down        stop the session's nodes, then remove the containers of every profile
# make session S=<name>  run config/sessions/<name>.yaml (scripts/sim.py up); ODOMETRY=truth
#                  or ODOMETRY=truth:JR81,RR83 puts those agents on ground truth (NOT PARITY)
# make status      scripts/sim.py status
# make test        the unit tests (python3 -m unittest discover -s tests)
# Compose goes through `sim.py compose`: it adds the pin file, the run dir it has checked and
# the CPU budget, and screens docker/compose.local.yaml. X access for the containers
# (xhost +SI:localuser:root, never +local: on this shared host) is granted by `sim.py up`
# when a session opens windows.

.PHONY: images up down session status test stack-up stack-down
images:     ; scripts/build_images.sh $(STACKS) $(if $(FORCE),--force) $(foreach r,$(REUSE),--reuse $(r)) \
                $(foreach s,$(SRC),--src $(s)) $(if $(JOBS),--jobs $(JOBS)) $(if $(PARALLEL),--parallel $(PARALLEL))
up:         ; python3 scripts/sim.py compose $(if $(GPU),--profile gpu) up -d
down:       ; python3 scripts/sim.py down --containers
session:    ; $(if $(S),python3 scripts/sim.py up $(S)$(if $(ODOMETRY), --odometry $(ODOMETRY)),$(error usage: make session S=<name>, for config/sessions/<name>.yaml))
status:     ; python3 scripts/sim.py status
test:       ; python3 -m unittest discover -s tests
# The names these had while the old single-container sim had up/down.
stack-up: up
stack-down: down
