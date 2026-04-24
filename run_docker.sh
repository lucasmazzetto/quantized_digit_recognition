#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-quantized_digit_recognition:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-quantized_digit_recognition_dev}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"
DOCKER_IPC_MODE="${DOCKER_IPC_MODE:-host}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-}"
DOCKER_USE_GPU="${DOCKER_USE_GPU:-auto}"

DOCKER_USER_FLAGS=(
  -u "${HOST_UID}:${HOST_GID}"
  -e "HOME=/tmp"
  -e "USER=${HOST_USER}"
)

# Map host account database so uid/gid resolve to a username inside the container.
if [ -f /etc/passwd ]; then
  DOCKER_USER_FLAGS+=(-v /etc/passwd:/etc/passwd:ro)
fi

if [ -f /etc/group ]; then
  DOCKER_USER_FLAGS+=(-v /etc/group:/etc/group:ro)
fi

DOCKER_MEMORY_FLAGS=()
if [ -n "${DOCKER_IPC_MODE}" ]; then
  DOCKER_MEMORY_FLAGS+=(--ipc "${DOCKER_IPC_MODE}")
fi

# Use this only when not sharing host IPC namespace.
if [ -n "${DOCKER_SHM_SIZE}" ] && [ "${DOCKER_IPC_MODE}" != "host" ]; then
  DOCKER_MEMORY_FLAGS+=(--shm-size "${DOCKER_SHM_SIZE}")
fi

if [ "$#" -gt 0 ]; then
  CONTAINER_CMD=("$@")
  DOCKER_TTY_FLAGS=()
  if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_TTY_FLAGS=(-it)
  fi
else
  CONTAINER_CMD=("bash")
  DOCKER_TTY_FLAGS=(-it)
fi

DOCKER_RUN_COMMON_ARGS=(
  --rm
  "${DOCKER_TTY_FLAGS[@]}"
  --name "${CONTAINER_NAME}"
  "${DOCKER_MEMORY_FLAGS[@]}"
  "${DOCKER_USER_FLAGS[@]}"
  -v "${SCRIPT_DIR}:/app/neural_network"
  -w /app/neural_network
)

if [ "${DOCKER_USE_GPU}" = "0" ] || [ "${DOCKER_USE_GPU}" = "false" ]; then
  docker run "${DOCKER_RUN_COMMON_ARGS[@]}" \
    "${IMAGE_NAME}" \
    "${CONTAINER_CMD[@]}"
  exit 0
fi

if [ "${DOCKER_USE_GPU}" = "1" ] || [ "${DOCKER_USE_GPU}" = "true" ]; then
  docker run "${DOCKER_RUN_COMMON_ARGS[@]}" \
    --gpus all \
    "${IMAGE_NAME}" \
    "${CONTAINER_CMD[@]}"
  exit 0
fi

# Auto mode: try GPU first, then fall back to CPU-only if unavailable.
if docker run "${DOCKER_RUN_COMMON_ARGS[@]}" --gpus all "${IMAGE_NAME}" "${CONTAINER_CMD[@]}"; then
  exit 0
fi

echo "GPU launch failed. Falling back to CPU-only container..." >&2
docker run "${DOCKER_RUN_COMMON_ARGS[@]}" \
  "${IMAGE_NAME}" \
  "${CONTAINER_CMD[@]}"
