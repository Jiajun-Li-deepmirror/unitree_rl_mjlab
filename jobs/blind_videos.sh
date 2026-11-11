#!/bin/bash
# blind_videos.sh R PKG: a shim to jobs/accept_videos.sh blind R PKG (2026-11-11, the user: one acceptance-video script).
# Kept only because a blind chain started before the change (gp2 on t2, 11-11) calls this name at its delivery; remove
# it once that chain has delivered. New code calls jobs/accept_videos.sh.
LOG=${BL_LOG:-} exec bash "$(dirname "$0")/accept_videos.sh" blind "$1" "$2"
