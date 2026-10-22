#!/usr/bin/env bash
# build_images.sh — build the per-stack sim images, each from its own repo's Dockerfile at
# the pins in docker/stacks.env, so the sim runs the robots' images, not a re-packaging.
#
#   scripts/build_images.sh [UNIT...] [--force] [--src UNIT=DIR]... [--reuse UNIT=IMAGE]...
#                           [--keep-external] [--jobs N] [--parallel N] [--prepare-only] [--list]
#
# UNITs: mighty mapping dlio drive gazebo elevation isaac, where dlio = dlio-jr + dlio-rr (the
# JR02- and RR08-calibrated images). No UNIT = all of them but isaac (Isaac Sim, for
# `simulator: isaac` sessions: built only when named).
#
# Per unit:
#   1. fetch its repo at the pin into .cache/src/<repo>@<sha> (gitignored) and do the
#      repo's own prep: dlio_ws imports src/ from dlio.repos; rover becomes a workspace
#      .cache/ws/rover@<sha>/src/rover with its acl_ctrl_safety submodule;
#   2. apply docker/patches/<repo>/*.patch in order (fixes not upstream yet; delete a patch
#      once the pin carries it);
#   3. docker buildx build --load with that repo's Dockerfile and build args, every
#      `FROM ros:humble-ros-base` pinned to ROS_BASE_IMAGE, tagged as in stacks.env;
#   4. check the image's rmw_zenoh + vendor builds against stacks.env (dpkg-query).
# gazebo, elevation and isaac are this repo's own docker/Dockerfile targets.
#
# A unit is skipped when its tag exists and this script built it from the same inputs
# (pin, patches, build args: the rover_sim.inputs label). An image without that label
# (tagged by hand or with --reuse) is rebuilt from source, unless --keep-external. For
# gazebo and elevation, built from this checkout, a skip also needs the provenance labels
# to match (the last commit that changed their inputs, and whether those have uncommitted
# edits); otherwise the image is rebuilt from cache to relabel it. --force rebuilds
# anyway; a rebuild with unchanged inputs is cheap, since BuildKit's layer cache holds.
#
#   --src UNIT=DIR      build from a local checkout instead of the pin: mighty, mapping,
#                       dlio (a dlio_ws root with src/ imported), drive (a rover checkout with
#                       acl_ctrl_safety initialised), elevation, gazebo (a vcs import of
#                       rover_sim.sim.repos). Always builds, and takes over the stack's tag
#                       labelled rover_sim.source=local:<dir>, which sim.py up/status warn
#                       about; the next plain run restores the pin. Patches go to a staged
#                       copy, never into the checkout.
#   --reuse UNIT=IMAGE  retag an existing image instead of building, e.g.
#                       --reuse mighty=mighty-hw:main-3cf7320. Refused if the image's
#                       revision labels name another commit than the pin.
#   --keep-external     keep images without a rover_sim.inputs label (earlier --reuse)
#   --jobs N            JOBS build arg (default 4): caps make parallelism in this repo's
#                       gazebo target (the upstream Dockerfiles take none). Always passed,
#                       so a plain run reuses the compile layers of the last run with the
#                       same N; any other N rebuilds them (only them: the apt layers above
#                       do not depend on it)
#   --parallel N        build up to N units at once; logs go to .cache/logs/<time>/
#   --prepare-only      fetch and patch, print the build commands, build nothing
#   --list              show each unit's tag and whether a build would run
#   SSH_KEY=<key>       forward this key instead of the ssh agent (--ssh default=<key>)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACKS_ENV="$ROOT/docker/stacks.env"
PATCHES="$ROOT/docker/patches"
CACHE="$ROOT/.cache"
PYTHON=${PYTHON:-python3}
UNITS_ALL=(mighty mapping dlio-jr dlio-rr drive gazebo elevation)
UNITS_NAMED=(isaac)              # built only when named

log() { printf '[images] %s\n' "$*" >&2; }
die() { printf '[images] error: %s\n' "$*" >&2; exit 1; }
usage() { awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$0"; }

# ---------- pins -----------------------------------------------------------------------
# stacks.env is parsed, not sourced: it must stay plain KEY=VALUE for sim.py and compose.
# Values land in PIN_<KEY> so the caller's environment can never shadow a pin.
load_pins() {
  local line key val n=0 p
  [[ -r $STACKS_ENV ]] || die "no $STACKS_ENV"
  while IFS= read -r line || [[ -n $line ]]; do
    n=$((n + 1))
    [[ $line =~ ^[[:space:]]*(#.*)?$ ]] && continue
    [[ $line =~ ^([A-Z][A-Z0-9_]*)=(.+)$ ]] || die "stacks.env:$n: expected KEY=VALUE"
    key=${BASH_REMATCH[1]} val=${BASH_REMATCH[2]}
    case $val in
      *[[:space:]\"\'\$\`\\]*) die "stacks.env:$n: $key: plain values only (no quotes, spaces or \$)" ;;
    esac
    printf -v "PIN_$key" '%s' "$val"
  done <"$STACKS_ENV"
  for key in ROS_BASE_IMAGE ROS_SNAPSHOT RMW_ZENOH_CPP_VER ZENOH_CPP_VENDOR_VER ROS_SNAPSHOT_KEY_FPR \
             MIGHTY_REPO MIGHTY_SHA MIGHTY_IMAGE MAPPING_REPO MAPPING_SHA MAPPING_IMAGE \
             DLIO_REPO DLIO_SHA DLIO_JR_ROBOT DLIO_JR_IMAGE DLIO_RR_ROBOT DLIO_RR_IMAGE \
             DRIVE_REPO DRIVE_SHA DRIVE_IMAGE GAZEBO_IMAGE ISAAC_IMAGE \
             ELEVATION_REPO ELEVATION_SHA ROS2_NUMPY_REPO ROS2_NUMPY_SHA ELEVATION_IMAGE; do
    p="PIN_$key"
    [[ -n ${!p:-} ]] || die "stacks.env: $key is missing"
  done
  for key in MIGHTY MAPPING DLIO DRIVE ELEVATION ROS2_NUMPY; do
    p="PIN_${key}_SHA"
    [[ ${!p} =~ ^[0-9a-f]{40}$ ]] || die "stacks.env: ${key}_SHA must be a full commit sha"
  done
  [[ $PIN_ROS_BASE_IMAGE == *@sha256:* ]] || die "stacks.env: ROS_BASE_IMAGE must be pinned by digest"
  # A tag names its content: a pin bump that forgets the tag would silently reuse the old
  # image (the skip check trusts the tag).
  tag_carries PIN_MIGHTY_IMAGE "${PIN_MIGHTY_SHA:0:7}"
  tag_carries PIN_MAPPING_IMAGE "${PIN_MAPPING_SHA:0:7}"
  tag_carries PIN_DLIO_JR_IMAGE "${PIN_DLIO_SHA:0:7}-$PIN_DLIO_JR_ROBOT"
  tag_carries PIN_DLIO_RR_IMAGE "${PIN_DLIO_SHA:0:7}-$PIN_DLIO_RR_ROBOT"
  tag_carries PIN_DRIVE_IMAGE "${PIN_DRIVE_SHA:0:7}"
  tag_carries PIN_ELEVATION_IMAGE "${PIN_ELEVATION_SHA:0:7}"
}
tag_carries() {  # $1: PIN_*_IMAGE variable name, $2: the tag it must have
  [[ ${!1} == *":$2" ]] || die "stacks.env: ${1#PIN_}=${!1} does not match its pin (want tag :$2)"
}

# ---------- units ----------------------------------------------------------------------
# Sets U_* for unit $1. U_LAYOUT: repo (repo-root context), dlio (repo root + src/ from
# dlio.repos), ws (workspace root, repo at src/rover), self (this repo).
unit_spec() {
  U=$1 U_TARGET='' U_SSH=0 U_SRC='' U_LOCAL_CTX='' U_ROBOT='' U_ARGS=()
  case $1 in
    mighty)  # mpc (private GitLab) is cloned inside the build
      U_KIND=pin U_LAYOUT=repo U_NAME=mighty U_DOCKERFILE=docker/Dockerfile.hw U_SSH=1
      U_REPO_URL=$PIN_MIGHTY_REPO U_SHA=$PIN_MIGHTY_SHA U_IMAGE=$PIN_MIGHTY_IMAGE ;;
    mapping)  # acl-mapping + map_fusion (private GitLab) are cloned inside the build
      U_KIND=pin U_LAYOUT=repo U_NAME=mapping_ws U_DOCKERFILE=docker/Dockerfile.mapping U_SSH=1
      U_REPO_URL=$PIN_MAPPING_REPO U_SHA=$PIN_MAPPING_SHA U_IMAGE=$PIN_MAPPING_IMAGE ;;
    dlio-jr|dlio-rr)  # build.sh: ROBOT_NAME picks cfg_overlay/<robot>/; WS as its compose
      local cls=${1#dlio-}
      local robot="PIN_DLIO_${cls^^}_ROBOT" image="PIN_DLIO_${cls^^}_IMAGE"
      U_KIND=pin U_LAYOUT=dlio U_NAME=dlio_ws U_DOCKERFILE=docker/Dockerfile
      U_REPO_URL=$PIN_DLIO_REPO U_SHA=$PIN_DLIO_SHA U_IMAGE=${!image} U_ROBOT=${!robot}
      U_ARGS=("WS=/home/swarm/code/dlio_ws" "ROBOT_NAME=$U_ROBOT") ;;
    drive)  # rover_bringup/docker/compose.yaml: context = workspace root, WS as there
      U_KIND=pin U_LAYOUT=ws U_NAME=rover U_DOCKERFILE=src/rover/rover_bringup/docker/Dockerfile
      U_REPO_URL=$PIN_DRIVE_REPO U_SHA=$PIN_DRIVE_SHA U_IMAGE=$PIN_DRIVE_IMAGE
      U_ARGS=("WS=/home/swarm/code/rover_ws") ;;
    gazebo)  # sim-only deps: rover_sim.sim.repos (public), or --src gazebo= a vcs import of it
      U_KIND=self U_LAYOUT=self U_NAME=rover_sim U_DOCKERFILE=docker/Dockerfile U_TARGET=gazebo
      U_REPO_URL='' U_SHA='' U_IMAGE=$PIN_GAZEBO_IMAGE U_LOCAL_CTX=stack-sim-src ;;
    elevation)  # --src elevation= an elevation_mapping_cupy checkout
      U_KIND=self U_LAYOUT=self U_NAME=rover_sim U_DOCKERFILE=docker/Dockerfile U_TARGET=elevation
      U_REPO_URL=$PIN_ELEVATION_REPO U_SHA=$PIN_ELEVATION_SHA U_IMAGE=$PIN_ELEVATION_IMAGE
      U_LOCAL_CTX=stack-elevation-src
      U_ARGS=("STACK_ELEVATION_GIT_URL=$PIN_ELEVATION_REPO" "STACK_ELEVATION_GIT_REF=$PIN_ELEVATION_SHA"
              "STACK_ROS2_NUMPY_GIT_URL=$PIN_ROS2_NUMPY_REPO" "STACK_ROS2_NUMPY_GIT_REF=$PIN_ROS2_NUMPY_SHA") ;;
    isaac)  # FROM NVIDIA's Isaac Sim image (docker/Dockerfile ISAAC_BASE_IMAGE) + stack-base's rmw_zenoh
      U_KIND=self U_LAYOUT=self U_NAME=rover_sim U_DOCKERFILE=docker/Dockerfile U_TARGET=isaac
      U_REPO_URL='' U_SHA='' U_IMAGE=$PIN_ISAAC_IMAGE U_LOCAL_CTX=stack-sim-src ;;
    *) die "unknown unit $1" ;;
  esac
  # The fleet RMW pin goes to every Dockerfile that declares it (mighty, mapping, drive,
  # patched dlio, ours): stacks.env stays the one place it is set.
  U_ARGS+=("ROS_SNAPSHOT=$PIN_ROS_SNAPSHOT" "RMW_ZENOH_CPP_VER=$PIN_RMW_ZENOH_CPP_VER"
           "ZENOH_CPP_VENDOR_VER=$PIN_ZENOH_CPP_VENDOR_VER" "ROS_SNAPSHOT_KEY_FPR=$PIN_ROS_SNAPSHOT_KEY_FPR")
  if [[ $U_KIND == self ]]; then U_ARGS+=("STACK_BASE_IMAGE=$PIN_ROS_BASE_IMAGE"); fi
  if [[ -n ${SRC[$1]:-} ]]; then
    U_SRC=${SRC[$1]}
    U_SRC=${U_SRC/#\~/$HOME}
    [[ -d $U_SRC ]] || die "--src ${1%-[jr]r}=$U_SRC is not a directory"
    U_SRC=$(cd "$U_SRC" && pwd)
  fi
  return 0
}

add_unit() {  # $1: unit or stack name -> WANT
  case $1 in
    dlio) WANT[dlio-jr]=1 WANT[dlio-rr]=1 ;;
    mighty|mapping|dlio-jr|dlio-rr|drive|gazebo|elevation|isaac) WANT[$1]=1 ;;
    *) die "unknown unit '$1' (units: mighty mapping dlio dlio-jr dlio-rr drive gazebo elevation isaac)" ;;
  esac
}

# ---------- sources --------------------------------------------------------------------
patch_files() {  # $1: repo name -> docker/patches/<repo>/*.patch, in apply order
  local f
  for f in "$PATCHES/$1"/*.patch; do
    if [[ -f $f ]]; then printf '%s\n' "$f"; fi
  done
}
patch_hash() {  # $1: repo name -> hash of its patch set (names + contents)
  local f
  { while IFS= read -r f; do basename "$f"; cat "$f"; done < <(patch_files "$1"); } \
    | sha256sum | cut -c1-16
}
tree_state() {  # $1: git checkout -> hash of HEAD plus every tracked and untracked change
  ( cd "$1" && { git rev-parse HEAD
                 git diff --binary HEAD
                 git ls-files -z --others --exclude-standard | xargs -0r sha256sum --; } \
      | sha256sum | cut -c1-16 )
}
# What docker/Dockerfile's stack targets COPY from this repo. Only these decide whether
# gazebo/elevation are stale, so editing scripts/ or config/ does not rebuild them.
SELF_INPUTS=(docker/Dockerfile docker/keys rover_sim.sim.repos src/rover_sim)
self_state() {  # hash of SELF_INPUTS as they are on disk, tracked or not
  ( cd "$ROOT" && git ls-files -z --cached --others --exclude-standard -- "${SELF_INPUTS[@]}" \
      | sort -zu | { xargs -0r sha256sum -- 2>/dev/null || true; } | sha256sum | cut -c1-16 )
}

self_provenance() {  # -> "<revision> <dirty>" of SELF_INPUTS: the last commit that changed
  # them, and whether they differ from HEAD now. Commits that touch nothing the image
  # COPYs leave both alone, so they neither rebuild nor relabel gazebo/elevation.
  local rev dirty=false
  rev=$(git -C "$ROOT" log -1 --format=%H -- "${SELF_INPUTS[@]}" 2>/dev/null) || rev=''
  if [[ -z $rev ]]; then rev=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null) || rev=unknown; fi
  if [[ -n $(git -C "$ROOT" status --porcelain -- "${SELF_INPUTS[@]}" 2>/dev/null) ]]; then dirty=true; fi
  printf '%s %s\n' "$rev" "$dirty"
}

clone_at() {  # $1 url, $2 sha, $3 dest: a detached checkout of $2, created atomically
  local url=$1 sha=$2 dest=$3 tmp
  if [[ -e $dest ]]; then return 0; fi
  log "fetch $url @ ${sha:0:7}"
  mkdir -p "$CACHE/tmp" "$(dirname "$dest")"
  tmp=$(mktemp -d "$CACHE/tmp/clone.XXXXXX")
  # blob:none: full history (no shallow-by-sha surprises), blobs only for the checkout.
  git clone -q --filter=blob:none --no-checkout "$url" "$tmp/repo"
  if ! git -C "$tmp/repo" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$tmp/repo" fetch -q origin "$sha"  # no longer on a branch
  fi
  git -C "$tmp/repo" -c advice.detachedHead=false checkout -q --detach "$sha"
  mv "$tmp/repo" "$dest"
  rmdir "$tmp"
}

repos_list() {  # $1: a .repos file -> "name url version" lines (vcs is not on the host)
  "$PYTHON" -I - "$1" <<'PY'
import sys
import yaml
repos = (yaml.safe_load(open(sys.argv[1])) or {}).get('repositories') or {}
for name, r in repos.items():
    if r.get('type', 'git') != 'git':
        sys.exit('%s: %s is not a git repo' % (sys.argv[1], name))
    print(name, r['url'], r['version'])
PY
}

dlio_import() {  # $1: dlio_ws root. `vcs import src < dlio.repos`, checked on every run
  local dir=$1 out name url ver
  out=$(repos_list "$dir/dlio.repos") || die "cannot read $dir/dlio.repos (needs $PYTHON with PyYAML)"
  while read -r name url ver; do
    [[ -n $name ]] || continue
    [[ $ver =~ ^[0-9a-f]{40}$ ]] || die "$dir/dlio.repos: $name is not pinned to a commit sha ($ver)"
    clone_at "$url" "$ver" "$dir/src/$name"
    if [[ $(git -C "$dir/src/$name" rev-parse HEAD) != "$ver" || -n $(git -C "$dir/src/$name" status --porcelain) ]]; then
      die "$dir/src/$name is not a clean $ver; delete $dir to re-fetch"
    fi
  done <<<"$out"
}

git_apply() {  # git apply in $1. The ceiling keeps a staged copy (not a repo) from being
  # taken as part of rover_sim's own repo, which would make git apply skip every path.
  GIT_CEILING_DIRECTORIES="$(dirname "$1")" git -C "$1" apply "${@:2}"
}
apply_patches() {  # $1: tree, $2: repo name
  local dir=$1 p
  while IFS= read -r p; do
    if git_apply "$dir" --check "$p" 2>/dev/null; then
      git_apply "$dir" "$p"
      log "$2: applied $(basename "$p")"
    elif git_apply "$dir" --reverse --check "$p" 2>/dev/null; then
      log "$2: $(basename "$p") is already in this source, skipped"
    else
      git_apply "$dir" --check "$p" || true
      die "$2: $(basename "$p") does not apply to $dir"
    fi
  done < <(patch_files "$2")
}

pin_tree() {  # $1 checkout, $2 sha, $3 repo name: make it exactly <sha> + docker/patches/<repo>
  local dir=$1 sha=$2 name=$3 stamp want
  stamp="$(git -C "$dir" rev-parse --absolute-git-dir)/rover_sim-prepared"
  if [[ $name == rover ]]; then git -C "$dir" submodule update -q --init -- acl_ctrl_safety; fi
  want="$sha $(patch_hash "$name")"
  if [[ -r $stamp && "$(<"$stamp")" == "$want $(tree_state "$dir")" ]]; then return 0; fi
  log "$name: reset to ${sha:0:7} + patches"
  git -C "$dir" reset -q --hard "$sha"
  git -C "$dir" clean -q -fd  # not -x: keeps ignored paths, i.e. dlio_ws's src/ imports
  if [[ $name == rover ]]; then git -C "$dir" submodule update -q --init -- acl_ctrl_safety; fi
  apply_patches "$dir" "$name"
  printf '%s %s\n' "$want" "$(tree_state "$dir")" >"$stamp"
}

stage() {  # $1 local checkout -> $2: a copy the patches can go into
  command -v rsync >/dev/null || die "rsync is needed to stage $1"
  mkdir -p "$2"
  rsync -a --delete --exclude=.git --exclude=/build/ --exclude=/install/ --exclude=/log/ \
        --exclude=__pycache__/ "$1/" "$2/"
}

declare -A CTX=() PREPARED=() KEY=()
prepare_unit() {  # fetch or stage, then patch: sets CTX[$U], the build context
  local id="$U_NAME:${U_SRC:-pin}" dir
  if [[ -n ${PREPARED[$id]:-} ]]; then CTX[$U]=${PREPARED[$id]}; return 0; fi
  case $U_LAYOUT in
    self) dir=$ROOT ;;
    repo|dlio)
      if [[ -n $U_SRC ]]; then
        if [[ $U_LAYOUT == dlio && ! -d $U_SRC/src ]]; then
          die "--src dlio=$U_SRC has no src/ (run: vcs import src < dlio.repos, in it)"
        fi
        dir=$U_SRC
        if [[ -n $(patch_files "$U_NAME") ]]; then
          dir="$CACHE/stage/$U_NAME"
          stage "$U_SRC" "$dir"
          apply_patches "$dir" "$U_NAME"
        fi
      else
        dir="$CACHE/src/$U_NAME@$U_SHA"
        clone_at "$U_REPO_URL" "$U_SHA" "$dir"
        pin_tree "$dir" "$U_SHA" "$U_NAME"
      fi
      if [[ $U_LAYOUT == dlio && -z $U_SRC ]]; then dlio_import "$dir"; fi ;;
    ws)
      if [[ -n $U_SRC ]]; then
        [[ -f $U_SRC/acl_ctrl_safety/package.xml ]] \
          || die "--src drive=$U_SRC: run git submodule update --init acl_ctrl_safety in it first"
        dir="$CACHE/stage/rover"
        stage "$U_SRC" "$dir/src/rover"
        apply_patches "$dir/src/rover" rover
      else
        dir="$CACHE/ws/rover@$U_SHA"
        clone_at "$U_REPO_URL" "$U_SHA" "$dir/src/rover"
        pin_tree "$dir/src/rover" "$U_SHA" rover
      fi ;;
  esac
  PREPARED[$id]=$dir
  CTX[$U]=$dir
}

# ---------- images ---------------------------------------------------------------------
unit_key() {  # what decides the image -> its rover_sim.inputs label
  local src
  if [[ -n $U_SRC ]]; then echo local; return 0; fi
  if [[ $U_KIND == self ]]; then
    src="rover_sim:$(self_state 2>/dev/null || echo "nogit-$(date +%s%N)")"
  else
    src="$U_REPO_URL@$U_SHA patches:$(patch_hash "$U_NAME")"
  fi
  printf '%s\n' "$U" "$U_DOCKERFILE" "$U_TARGET" "$PIN_ROS_BASE_IMAGE" "$src" "${U_ARGS[@]}" \
    | sha256sum | cut -c1-16
}
image_label() {  # $1 image, $2 label -> value, empty if unset
  local v
  v=$(docker image inspect -f "{{index .Config.Labels \"$2\"}}" "$1" 2>/dev/null) || true
  [[ $v == '<no value>' ]] && v=''
  printf '%s' "$v"
}
unit_state() {  # $1 wanted key -> missing | current | relabel | stale | external
  local have
  if ! docker image inspect "$U_IMAGE" >/dev/null 2>&1; then echo missing; return 0; fi
  have=$(image_label "$U_IMAGE" rover_sim.inputs)
  if [[ -z $have ]]; then echo external
  elif [[ $have != "$1" ]]; then echo stale
  elif [[ $U_KIND == self && -z $U_SRC ]] && [[ "$(self_provenance)" != \
         "$(image_label "$U_IMAGE" org.opencontainers.image.revision) $(image_label "$U_IMAGE" rover_sim.dirty)" ]]; then
    echo relabel
  else echo current; fi
}

unit_labels() {  # -> LABELS: provenance, and the inputs key the skip check reads
  # image.source/revision name the repo whose Dockerfile built the image.
  local url=$U_REPO_URL src=pin rev=$U_SHA dirty=false names
  if [[ $U_KIND == self ]]; then
    src=rover_sim${U_SRC:++local:$U_SRC}
    url=$(git -C "$ROOT" remote get-url origin 2>/dev/null) || url=rover_sim
    read -r rev dirty < <(self_provenance)
  elif [[ -n $U_SRC ]]; then
    src="local:$U_SRC"
    rev=$(git -C "$U_SRC" rev-parse HEAD 2>/dev/null) || rev=unknown
    if [[ -n $(git -C "$U_SRC" status --porcelain 2>/dev/null) ]]; then dirty=true; fi
  fi
  U_REV=$rev U_DIRTY=$dirty
  LABELS=(--label "rover_sim.unit=$U" --label "rover_sim.inputs=${KEY[$U]}"
          --label "rover_sim.source=$src" --label "rover_sim.dirty=$dirty"
          --label "org.opencontainers.image.source=$url"
          --label "org.opencontainers.image.revision=$rev")
  if [[ $U_KIND == pin ]]; then
    names=$(patch_files "$U_NAME" | xargs -r -n1 basename | paste -sd, -)
    LABELS+=(--label "rover_sim.patches=${names:-none}")
  fi
  if [[ -n $U_ROBOT ]]; then LABELS+=(--label "rover_sim.robot=$U_ROBOT"); fi
  if [[ $U == elevation ]]; then
    if [[ -n $U_SRC ]]; then
      rev=$(git -C "$U_SRC" rev-parse HEAD 2>/dev/null) || rev=unknown
      LABELS+=(--label "rover_sim.elevation.ref=local:$U_SRC@$rev")
    else
      LABELS+=(--label "rover_sim.elevation.ref=$U_SHA")
    fi
  fi
  return 0
}

build_cmd() {  # $1 unit (prepared) -> CMD
  unit_spec "$1"
  local ctx=${CTX[$U]} a
  CMD=(docker buildx build --load -f "$ctx/$U_DOCKERFILE" -t "$U_IMAGE"
       --build-context "ros:humble-ros-base=docker-image://$PIN_ROS_BASE_IMAGE")
  if [[ -n $U_TARGET ]]; then CMD+=(--target "$U_TARGET"); fi
  CMD+=("${SSH_FLAG[@]}")
  for a in "${U_ARGS[@]}"; do CMD+=(--build-arg "$a"); done
  if [[ $U_KIND == self ]]; then CMD+=(--build-arg "JOBS=$JOBS"); fi
  # Ours: a local checkout replaces the stage that clones the pin.
  if [[ $U_KIND == self && -n $U_SRC ]]; then CMD+=(--build-context "$U_LOCAL_CTX=$U_SRC"); fi
  unit_labels
  CMD+=("${LABELS[@]}")
  # mighty's Dockerfile.hw sets its own revision and dirty labels from these (mighty#16; its
  # last instruction, so they bust no cache): the same commit and state as rover_sim's labels.
  if [[ $U == mighty ]]; then CMD+=(--build-arg "GIT_SHA=$U_REV" --build-arg "GIT_DIRTY=$U_DIRTY"); fi
  if (( PARALLEL > 1 )); then CMD+=(--progress=plain); fi
  CMD+=("$ctx")
}
build_unit() {
  build_cmd "$1"
  log "[$U] build $U_IMAGE from ${U_SRC:-${CTX[$U]}}"
  "${CMD[@]}"
}

verify_unit() {  # $1 unit -> 0 when its image carries exactly the stacks.env rmw_zenoh pair
  local want                                    # (what it has instead: VERIFY_GOT)
  unit_spec "$1"
  want=$(printf '%s\n' "ros-humble-rmw-zenoh-cpp=$PIN_RMW_ZENOH_CPP_VER" \
                       "ros-humble-zenoh-cpp-vendor=$PIN_ZENOH_CPP_VENDOR_VER")
  # A throwaway, network-less container named like the sim's own. isaac has no ROS packages:
  # its libraries are stack-base's, and it carries the pin stack-base verified with them.
  if [[ $1 == isaac ]]; then
    VERIFY_GOT=$(docker run --rm --network none --name "rover-sim-verify-$1-$$" --entrypoint cat \
                   "$U_IMAGE" /etc/rover_sim/rmw-zenoh.pin 2>/dev/null | awk -F'\t' '{print $1 "=" $2}' | sort) || true
  else
    VERIFY_GOT=$(docker run --rm --network none --name "rover-sim-verify-$1-$$" --entrypoint dpkg-query \
                   "$U_IMAGE" -W -f '${Package}=${Version}\n' ros-humble-rmw-zenoh-cpp ros-humble-zenoh-cpp-vendor \
                   2>/dev/null | sort) || true
  fi
  [[ $VERIFY_GOT == "$want" ]]
}

pin_file_arg() {  # $1 file in the pinned repo, $2 ARG name -> its default, from the cached
  # checkout (empty when there is none yet)
  local dir="$CACHE/src/$U_NAME@$U_SHA"
  [[ -d $dir ]] || return 0
  git -C "$dir" show "$U_SHA:$1" 2>/dev/null | sed -n "s/^ARG $2=\(.*\)$/\1/p" | head -n1
}
reuse_check() {  # $1 unit, $2 image: refuse an image whose labels name another commit than the pin
  local rev want label got
  unit_spec "$1"
  case $U_KIND in
    pin) label=org.opencontainers.image.revision want=$U_SHA ;;
    self) label=rover_sim.elevation.ref want=$U_SHA ;;
  esac
  if [[ -n $want ]]; then
    rev=$(image_label "$2" "$label")
    if [[ -n $rev && $rev != "$want" ]]; then
      die "--reuse $1=$2: its $label is $rev, the pin is $want (use an image built at the pin, or bump the pin)"
    elif [[ -z $rev ]]; then
      log "[$1] $2 has no $label label: cannot verify it is ${want:0:7}; sim.py will warn that it is unverified"
    fi
  fi
  if [[ $1 == mapping ]]; then  # map_fusion is cloned inside the build at the Dockerfile's ARG
    want=$(pin_file_arg docker/Dockerfile.mapping MAP_FUSION_REF)
    got=$(image_label "$2" org.mit-acl.map-fusion.ref)
    if [[ -n $want && -n $got && $got != "$want" ]]; then
      die "--reuse mapping=$2: its map_fusion is $got, mapping_ws ${U_SHA:0:7} pins $want"
    fi
  fi
  return 0
}

# ---------- main -----------------------------------------------------------------------
# JOBS is fixed, not derived from the host: it is in the compile layers' cache keys.
FORCE=0 PREPARE_ONLY=0 LIST=0 JOBS=4 PARALLEL=1 KEEP_EXTERNAL=0
declare -A WANT=() REUSE=() ACTION=() SRC=()
while (($#)); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --force) FORCE=1 ;;
    --reuse)
      [[ ${2:-} == ?*=?* ]] || die "--reuse needs UNIT=IMAGE"
      [[ ${2%%=*} != dlio ]] || die "--reuse: dlio has two images, use dlio-jr=IMAGE / dlio-rr=IMAGE"
      REUSE[${2%%=*}]=${2#*=}
      shift ;;
    --src)
      [[ ${2:-} == ?*=?* ]] || die "--src needs UNIT=DIR"
      case ${2%%=*} in
        dlio) SRC[dlio-jr]=${2#*=} SRC[dlio-rr]=${2#*=} ;;
        mighty|mapping|drive|gazebo|elevation) SRC[${2%%=*}]=${2#*=} ;;
        *) die "--src: unknown unit '${2%%=*}' (mighty mapping dlio drive gazebo elevation)" ;;
      esac
      shift ;;
    --keep-external) KEEP_EXTERNAL=1 ;;
    --jobs)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || die "--jobs needs a positive number"
      JOBS=$2
      shift ;;
    --parallel)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || die "--parallel needs a positive number"
      PARALLEL=$2
      shift ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    --list) LIST=1 ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) add_unit "$1" ;;
  esac
  shift
done

load_pins
for v in MIGHTY_SRC MAPPING_SRC DLIO_SRC DRIVE_SRC ELEVATION_SRC GAZEBO_SRC; do
  if [[ -n ${!v:-} ]]; then log "note: $v is set but not used; local builds: --src UNIT=DIR"; fi
done
for u in "${!REUSE[@]}"; do add_unit "$u"; done
if (( ${#WANT[@]} == 0 )); then for u in "${UNITS_ALL[@]}"; do WANT[$u]=1; done; fi
UNITS=()
for u in "${UNITS_ALL[@]}" "${UNITS_NAMED[@]}"; do
  if [[ -n ${WANT[$u]:-} ]]; then UNITS+=("$u"); fi
done

if (( LIST )); then
  printf '%-10s %-34s %-9s %s\n' UNIT IMAGE STATE NOTE
  for u in "${UNITS[@]}"; do
    unit_spec "$u"
    k=$(unit_key)
    s=$(unit_state "$k")
    case $s in
      missing) n='builds' ;;
      current) n='up to date' ;;
      relabel) n='inputs unchanged, provenance labels out of date: cached rebuild' ;;
      stale) n='inputs changed: rebuilds' ;;
      external) n="not built here (reused/hand-tagged): $( ((KEEP_EXTERNAL)) && echo kept || echo 'rebuilds (--keep-external keeps it)')" ;;
    esac
    if [[ -n $U_SRC ]]; then n="--src: builds $U_SRC"; fi
    printf '%-10s %-34s %-9s %s\n' "$u" "$U_IMAGE" "$s" "$n"
  done
  exit 0
fi

# One run at a time: the cache trees are reset and patched in place.
mkdir -p "$CACHE"
exec 9>"$CACHE/.lock"
flock -n 9 || die "another build_images.sh is running"
rm -rf "$CACHE/tmp"

SSH_FLAG=()
if [[ -n ${SSH_KEY:-} ]]; then
  [[ -r $SSH_KEY ]] || die "SSH_KEY=$SSH_KEY is not readable"
  SSH_FLAG=(--ssh "default=$SSH_KEY")
elif [[ -S ${SSH_AUTH_SOCK:-} ]]; then
  SSH_FLAG=(--ssh default)
fi
# Host-side clones must fail, not prompt, without a usable key.
: "${GIT_SSH_COMMAND:=ssh -o BatchMode=yes${SSH_KEY:+ -i $SSH_KEY -o IdentitiesOnly=yes}}"
export GIT_SSH_COMMAND GIT_TERMINAL_PROMPT=0

BUILD=()
for u in "${UNITS[@]}"; do
  unit_spec "$u"
  if [[ -n ${REUSE[$u]:-} ]]; then
    docker image inspect "${REUSE[$u]}" >/dev/null 2>&1 || die "--reuse $u=${REUSE[$u]}: no such image"
    reuse_check "$u" "${REUSE[$u]}"
    ACTION[$u]="reuse ${REUSE[$u]}"
    continue
  fi
  KEY[$u]=$(unit_key)
  state=$(unit_state "${KEY[$u]}")
  if (( ! FORCE )) && [[ -z $U_SRC ]] && [[ $state == current || ( $state == external && KEEP_EXTERNAL -eq 1 ) ]]; then
    ACTION[$u]="skip ($state)"
    continue
  fi
  case $state in
    relabel) log "[$u] provenance labels out of date (inputs unchanged): cached rebuild" ;;
    external) log "[$u] $U_IMAGE was not built by this script (reused or hand-tagged): building from source" ;;
  esac
  if (( U_SSH )) && (( ${#SSH_FLAG[@]} == 0 )); then
    die "$u clones private repos inside its build: start an ssh agent holding the GitHub/GitLab key, or set SSH_KEY"
  fi
  prepare_unit
  BUILD+=("$u")
  ACTION[$u]=build
done

if (( PREPARE_ONLY )); then
  for u in "${BUILD[@]}"; do
    build_cmd "$u"
    printf '%s:' "$u"; printf ' %q' "${CMD[@]}"; printf '\n'
  done
  for u in "${UNITS[@]}"; do [[ ${ACTION[$u]} == build ]] || log "$u: ${ACTION[$u]}"; done
  exit 0
fi

FAILED=()
if (( PARALLEL > 1 && ${#BUILD[@]} > 1 )); then
  logdir="$CACHE/logs/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$logdir"
  for u in "${BUILD[@]}"; do
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n || true; done
    log "[$u] building, log: $logdir/$u.log"
    ( trap 'echo $? >"$logdir/$u.rc"' EXIT; build_unit "$u" ) >"$logdir/$u.log" 2>&1 &
  done
  wait || true
  for u in "${BUILD[@]}"; do
    if [[ $(cat "$logdir/$u.rc" 2>/dev/null) != 0 ]]; then
      FAILED+=("$u")
      ACTION[$u]='build FAILED'
      log "[$u] FAILED; tail of $logdir/$u.log:"
      tail -n 25 "$logdir/$u.log" >&2
    fi
  done
else
  for u in "${BUILD[@]}"; do build_unit "$u"; done
fi

for u in "${!REUSE[@]}"; do
  unit_spec "$u"
  docker tag "${REUSE[$u]}" "$U_IMAGE"
  log "[$u] $U_IMAGE = ${REUSE[$u]} (reused: not built from the pin by this script)"
done

printf '\n%-10s %-34s %-13s %-9s %s\n' UNIT IMAGE ID RMW ACTION >&2
BAD=()
for u in "${UNITS[@]}"; do
  unit_spec "$u"
  id=$(docker image inspect -f '{{.Id}}' "$U_IMAGE" 2>/dev/null | cut -c8-19) || id=''
  if [[ -z $id ]]; then
    rmw=missing
  elif verify_unit "$u"; then
    rmw=ok
  else
    rmw=MISMATCH
    BAD+=("$u has: ${VERIFY_GOT:-no rmw_zenoh}")
  fi
  printf '%-10s %-34s %-13s %-9s %s\n' "$u" "$U_IMAGE" "${id:--}" "$rmw" "${ACTION[$u]}" >&2
  if [[ $rmw == missing ]]; then BAD+=("$u: no image $U_IMAGE"); fi
done
(( ${#FAILED[@]} == 0 )) || die "failed: ${FAILED[*]}"
if (( ${#BAD[@]} )); then
  printf '[images] %s\n' "${BAD[@]//$'\n'/ }" >&2
  die "want rmw_zenoh_cpp=$PIN_RMW_ZENOH_CPP_VER + zenoh_cpp_vendor=$PIN_ZENOH_CPP_VENDOR_VER in every image"
fi
