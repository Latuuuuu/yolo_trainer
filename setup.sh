#!/usr/bin/env bash
# One-shot environment setup for yolo_trainer.
#   bash setup.sh              check the host, build the image, start the container
#   bash setup.sh --no-build   just (re)start the container
#   bash setup.sh --shell      also drop into the container when done
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=yolo_trainer
DO_BUILD=1
DO_SHELL=0

for arg in "$@"; do
  case "$arg" in
    --no-build) DO_BUILD=0 ;;
    --shell)    DO_SHELL=1 ;;
    -h|--help)  sed -n '2,5p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 1 ;;
  esac
done

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31m✗ %s\033[0m\n' "$1" >&2; shift; for l in "$@"; do printf '  %s\n' "$l" >&2; done; exit 1; }

step "1/5 Checking Docker"
command -v docker >/dev/null || fail "docker is not installed." \
  "Install Docker Engine: https://docs.docker.com/engine/install/"
docker compose version >/dev/null 2>&1 || fail "the 'docker compose' plugin is missing." \
  "Install docker-compose-plugin: https://docs.docker.com/compose/install/linux/"
docker info >/dev/null 2>&1 || fail "cannot talk to the Docker daemon." \
  "Start it:            sudo systemctl start docker" \
  "Run without sudo:    sudo usermod -aG docker \$USER   (then log out and back in)"
ok "docker $(docker version --format '{{.Server.Version}}') and compose plugin available"

step "2/5 Checking the GPU"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — no NVIDIA driver on this machine." \
  "This project needs an NVIDIA GPU. Install the driver first."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | while read -r line; do ok "GPU: $line"; done
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  ok "nvidia container runtime registered"
else
  fail "Docker cannot see the NVIDIA runtime." \
    "Install nvidia-container-toolkit, then:" \
    "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" \
    "Guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

step "3/5 Writing docker/.env (so container files belong to you)"
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" > "$REPO_ROOT/docker/.env"
ok "UID=$(id -u) GID=$(id -g)"
mkdir -p "$REPO_ROOT"/{configs,datasets,imgs,models,runs,weights}

step "4/5 Allowing the container to open windows (X11)"
if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null; then
  xhost +local:docker >/dev/null && ok "xhost +local:docker — 'predict.py --show' can open windows"
  warn "this resets on reboot; re-run setup.sh (or just xhost +local:docker) afterwards"
else
  warn "no DISPLAY or no xhost — windows will not open; use 'predict.py' without --show"
fi

step "5/5 Building and starting the container"
cd "$REPO_ROOT/docker"
if [ "$DO_BUILD" = 1 ]; then
  echo "  (first build downloads several GB and takes a while)"
  docker compose build
fi

# A container with this name may be left over from an older version of this
# repo (compose refuses to reuse the name across projects). It holds no data —
# the repository is mounted from the host — so recreating it is safe.
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  project=$(docker container inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$CONTAINER")
  if [ "$project" != "yolo_trainer" ]; then
    warn "removing a stale container named $CONTAINER (compose project: ${project:-none})"
    docker rm -f "$CONTAINER" >/dev/null
  fi
fi

docker compose up -d
docker exec "$CONTAINER" python -c "import torch; assert torch.cuda.is_available()" \
  && ok "container '$CONTAINER' is up and sees the GPU" \
  || fail "the container started but PyTorch cannot see the GPU." \
       "Check 'docker exec -it $CONTAINER nvidia-smi' and the toolkit install above."

cat <<MSG

Done. Next steps:

  docker exec -it $CONTAINER bash          # get a shell inside the container
  python train.py --config configs/example_detect.yaml
  python predict.py --weights runs/<run>/weights/best.pt --source imgs/<folder>

Stop it later with:  cd docker && docker compose down
MSG

if [ "$DO_SHELL" = 1 ]; then
  exec docker exec -it "$CONTAINER" bash
fi
