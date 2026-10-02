#!/usr/bin/env bash
# Fetch ACL (Animation Compression Library, MIT) at the commit every reference number was measured with, into
# third_party/acl, with its rtm and sjson-cpp submodules. Then build the tools:
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build -j
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/acl"
ACL_COMMIT=3ee568542eca4428e1041908b4a644b98e7885bd
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/nfrechette/acl.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$ACL_COMMIT" || true
git -C "$DEST" checkout --quiet "$ACL_COMMIT"
git -C "$DEST" submodule update --init external/rtm external/sjson-cpp
echo "ACL at $(git -C "$DEST" rev-parse --short HEAD): rtm $(git -C "$DEST/external/rtm" rev-parse --short HEAD), sjson-cpp $(git -C "$DEST/external/sjson-cpp" rev-parse --short HEAD)"
