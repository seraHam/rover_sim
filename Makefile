# rover_sim — build/run the rover-sim container.
#
# JOBS caps build parallelism in docker/Dockerfile (default 1; make build JOBS=N).
# The fallback is in the recipe (${JOBS:-1}) so an empty JOBS= also means 1,
# not unbounded `make -j`.
COMPOSE = docker compose -f docker/compose.yaml

.PHONY: build up down shell logs
build: ; $(COMPOSE) build --ssh default$(if $(SSH_KEY),=$(SSH_KEY)) --build-arg JOBS=$${JOBS:-1}
up:    ; xhost +local: >/dev/null 2>&1 || true; $(COMPOSE) up -d
down:  ; $(COMPOSE) down
shell: ; docker exec -it rover-sim bash
logs:  ; $(COMPOSE) logs -f
