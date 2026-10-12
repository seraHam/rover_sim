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
