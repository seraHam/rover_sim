# rover_sim — build/run the rover-sim container.
#
# JOBS caps build parallelism in docker/Dockerfile (default 1; make build JOBS=N).
# The fallback is in the recipe (${JOBS:-1}) so an empty JOBS= also means 1,
# not unbounded `make -j`.
#
# MIGHTY_SRC, ACL_MAPPING_SRC, MPC_SRC: build that repo from a local checkout
# instead of its pinned commit (make build ACL_MAPPING_SRC=~/code/acl-mapping).
COMPOSE = docker compose -f docker/compose.yaml
LOCAL_SRC = $(if $(MIGHTY_SRC),--build-context mighty-src=$(abspath $(MIGHTY_SRC))) \
            $(if $(ACL_MAPPING_SRC),--build-context acl-mapping-src=$(abspath $(ACL_MAPPING_SRC))) \
            $(if $(MPC_SRC),--build-context mpc-src=$(abspath $(MPC_SRC)))

.PHONY: build up down shell logs
build: ; docker buildx build --load -t rover-sim:local -f docker/Dockerfile \
           --ssh default$(if $(SSH_KEY),=$(SSH_KEY)) --build-arg JOBS=$${JOBS:-1} $(LOCAL_SRC) .
up:    ; xhost +local: >/dev/null 2>&1 || true; $(COMPOSE) up -d
down:  ; $(COMPOSE) down
shell: ; docker exec -it rover-sim bash
logs:  ; $(COMPOSE) logs -f
