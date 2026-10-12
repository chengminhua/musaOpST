#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# Reuse a caller-selected local image. Does not alter existing containers.
set -euo pipefail
: "${MUSA_OPST_IMAGE:?Set MUSA_OPST_IMAGE to an existing local image ID or tag}"
: "${MUSA_OPST_GPU:?Set MUSA_OPST_GPU to the verified idle physical GPU index}"
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ $# == 0 ]]; then echo 'usage: tools/run_container.sh COMMAND [ARGS...]' >&2; exit 2; fi
mkdir -p "$workspace"/{.cache,.tmp}
mkdir -p "$workspace/.cache"/{torch_extensions,triton,tilelang,musa}
docker image inspect "$MUSA_OPST_IMAGE" >/dev/null
render_group=$(getent group render | cut -d: -f3)
group_args=()
if [[ -n "$render_group" ]]; then group_args=(--group-add "$render_group"); fi
exec docker run --rm -i --pull=never --runtime=mthreads --read-only --network=none --log-driver=none \
 --cap-drop=ALL --security-opt=no-new-privileges --user "$(id -u):$(id -g)" "${group_args[@]}" \
 --cpus=4 --memory=8g --shm-size=256m \
 --env "MTHREADS_VISIBLE_DEVICES=$MUSA_OPST_GPU" --env MUSA_VISIBLE_DEVICES=0 \
 --env MUSA_HOME=/usr/local/musa \
 --env PATH=/usr/local/musa/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
 --env "PYTHONPATH=$workspace:$workspace/.deps" --env PYTHONDONTWRITEBYTECODE=1 \
 --env "TMPDIR=$workspace/.tmp" --env "XDG_CACHE_HOME=$workspace/.cache" \
 --env "TORCH_EXTENSIONS_DIR=$workspace/.cache/torch_extensions" \
 --env "TRITON_CACHE_DIR=$workspace/.cache/triton" --env "TILELANG_CACHE_DIR=$workspace/.cache/tilelang" \
 --env "MUSA_CACHE_PATH=$workspace/.cache/musa" --env OMP_NUM_THREADS=2 --env MAX_JOBS=2 \
 --mount "type=bind,source=$workspace,target=$workspace" --workdir "$workspace" \
 --entrypoint /bin/bash "$MUSA_OPST_IMAGE" --noprofile --norc -c 'exec "$@"' _ "$@"
