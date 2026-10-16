#!/bin/bash
# Save the Arena image built on this instance to a PRIVATE registry, so later instances pull it in minutes
# instead of building for an hour. Run on the instance after a successful setup.
#
#   docker login <registry>                     # e.g. nvcr.io (user '$oauthtoken', NGC API key), or AWS ECR
#   ./push_arena_image.sh nvcr.io/<org>/sim2gate-arena:2cc9e97
#
# Keep the repository private: the image contains NVIDIA Isaac Sim, whose license terms govern redistribution.
# Then set ARENA_IMAGE to the same tag in setup.sh (plus REGISTRY/REGISTRY_USER/REGISTRY_TOKEN if needed).
set -euo pipefail
TARGET="${1:?usage: $0 <registry/repository:tag>}"
SOURCE="${SOURCE_IMAGE:-isaaclab_arena:latest}"
docker image inspect "${SOURCE}" >/dev/null
docker tag "${SOURCE}" "${TARGET}"
docker push "${TARGET}"
echo "Pushed ${TARGET}. Set ARENA_IMAGE=${TARGET} in setup.sh for the next deploy."
