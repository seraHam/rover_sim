# rover_sim — build/run the rover-sim container.
#
# JOBS caps build parallelism in docker/Dockerfile (default 1; make build JOBS=N).
# The fallback is in the recipe (${JOBS:-1}) so an empty JOBS= also means 1,
# not unbounded `make -j`.
#
# MAPPER=elevation also builds elevation_mapping_cupy, the rovers' mapper (needs
# an NVIDIA GPU to run). The ACL mapper is always built.
#
# MIGHTY_SRC, ACL_MAPPING_SRC, MPC_SRC, ELEVATION_SRC: build that repo from a
# local checkout instead of its pinned commit (make build ACL_MAPPING_SRC=~/code/acl-mapping).
#
# make up adds docker/compose.gpu.yaml when the image has the elevation mapper,
# and docker/compose.local.yaml (gitignored, this machine's overrides) if it exists.
IMAGE = rover-sim:local
$(if $(filter-out acl elevation,$(MAPPER)),$(error MAPPER must be acl or elevation))
TARGET = $(if $(filter elevation,$(MAPPER)),runtime-elevation,runtime)
GPU_COMPOSE = $(if $(findstring elevation,$(shell docker image inspect -f '{{index .Config.Labels "rover_sim.mappers"}}' $(IMAGE) 2>/dev/null)),-f docker/compose.gpu.yaml)
LOCAL_COMPOSE = $(if $(wildcard docker/compose.local.yaml),-f docker/compose.local.yaml)
COMPOSE = docker compose -f docker/compose.yaml $(GPU_COMPOSE) $(LOCAL_COMPOSE)
LOCAL_SRC = $(if $(MIGHTY_SRC),--build-context mighty-src=$(abspath $(MIGHTY_SRC))) \
            $(if $(ACL_MAPPING_SRC),--build-context acl-mapping-src=$(abspath $(ACL_MAPPING_SRC))) \
            $(if $(MPC_SRC),--build-context mpc-src=$(abspath $(MPC_SRC))) \
            $(if $(ELEVATION_SRC),--build-context elevation-src=$(abspath $(ELEVATION_SRC)))

.PHONY: build up down shell logs
build: ; docker buildx build --load -t $(IMAGE) -f docker/Dockerfile --target $(TARGET) \
           --ssh default$(if $(SSH_KEY),=$(SSH_KEY)) --build-arg JOBS=$${JOBS:-1} $(LOCAL_SRC) .
up:    ; xhost +local: >/dev/null 2>&1 || true; $(COMPOSE) up -d
down:  ; $(COMPOSE) down
shell: ; docker exec -it rover-sim bash
logs:  ; $(COMPOSE) logs -f

# ---------- per-stack sim (docker/compose.stack.yaml), alongside the above until PR 6 ----------
# make images      each stack's image from its own repo's Dockerfile at the docker/stacks.env
#                  pins (scripts/build_images.sh --help). STACKS="mighty dlio" picks stacks,
#                  FORCE=1 rebuilds, REUSE="mighty=mighty-hw:main-3cf7320 ..." retags an
#                  existing image instead, JOBS=N caps colcon in our own targets, PARALLEL=N
#                  builds N stacks at once. SRC="mighty=~/code/mighty ..." builds a local
#                  checkout into the stack's tag (sim.py warns while it is in use); not
#                  MIGHTY_SRC/ELEVATION_SRC, which are the legacy build's.
# make stack-up    the sim-* containers, idle (sessions docker-exec their nodes into them);
#                  GPU=1 adds sim-elevation (compose profile gpu)
# make stack-down  stop the session's nodes, then remove the containers of every profile
# make session S=<name>  run config/sessions/<name>.yaml (scripts/sim.py up)
# Compose goes through `sim.py compose`: it adds the pin file, the run dir it has checked and
# the CPU budget, and skips a compose.local.yaml written for the legacy service. X access
# for the containers (xhost +SI:localuser:root, never +local: on this shared host) is
# granted by `sim.py up` when a session opens windows.

.PHONY: images stack-up stack-down session
images:     ; scripts/build_images.sh $(STACKS) $(if $(FORCE),--force) $(foreach r,$(REUSE),--reuse $(r)) \
                $(foreach s,$(SRC),--src $(s)) $(if $(JOBS),--jobs $(JOBS)) $(if $(PARALLEL),--parallel $(PARALLEL))
stack-up:   ; python3 scripts/sim.py compose $(if $(GPU),--profile gpu) up -d
stack-down: ; python3 scripts/sim.py down --containers
session:    ; $(if $(S),python3 scripts/sim.py up $(S),$(error usage: make session S=<name>, for config/sessions/<name>.yaml))
