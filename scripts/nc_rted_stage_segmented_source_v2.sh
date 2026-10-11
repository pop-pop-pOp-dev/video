#!/usr/bin/env bash
# Freeze an accepted corrected segmented source tree; this script never launches work.
set -euo pipefail

source_root= overlay_root= helper= helper_sha= controller= controller_sha= destination=
formal_sha= queue_script_sha= queue_sha= attestation_sha=
while (($#)); do
  case "$1" in
    --source-root) source_root=$2; shift 2;;
    --overlay-root) overlay_root=$2; shift 2;;
    --helper) helper=$2; shift 2;;
    --helper-sha256) helper_sha=$2; shift 2;;
    --controller) controller=$2; shift 2;;
    --controller-sha256) controller_sha=$2; shift 2;;
    --formal-sha256) formal_sha=$2; shift 2;;
    --queue-script-sha256) queue_script_sha=$2; shift 2;;
    --queue-sha256) queue_sha=$2; shift 2;;
    --attestation-sha256) attestation_sha=$2; shift 2;;
    --destination) destination=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
for value in "$source_root" "$overlay_root" "$helper" "$helper_sha" "$controller" "$controller_sha" "$formal_sha" "$queue_script_sha" "$queue_sha" "$attestation_sha" "$destination"; do
  [[ -n $value ]] || { echo "all staging arguments are required" >&2; exit 2; }
done
[[ -d $source_root && -d $overlay_root && -f $helper && -f $controller && ! -e $destination ]] || {
  echo "source, overlay, tool, or destination precondition differs" >&2; exit 1;
}
[[ $(sha256sum "$helper" | awk '{print $1}') == "$helper_sha" && $(sha256sum "$controller" | awk '{print $1}') == "$controller_sha" ]] || {
  echo "external tool bytes differ" >&2; exit 1;
}

declare -A expected=(
  [scripts/nc_rted_interleaved_formal.py]=$formal_sha
  [scripts/nc_rted_queue.py]=$queue_script_sha
  [src/nc_rted/queue.py]=$queue_sha
  [src/nc_rted/resource_attestation.py]=$attestation_sha
)
for relative in "${!expected[@]}"; do
  [[ -f $overlay_root/$relative ]] || { echo "missing overlay: $relative" >&2; exit 1; }
  [[ $(sha256sum "$overlay_root/$relative" | awk '{print $1}') == ${expected[$relative]} ]] || {
    echo "overlay digest differs: $relative" >&2; exit 1;
  }
done

parent=$(dirname "$destination")
name=$(basename "$destination")
[[ -d $parent ]] || { echo "destination parent is absent" >&2; exit 1; }
stage=$(mktemp -d "$parent/.${name}.stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp -a "$source_root/." "$stage/"
for relative in "${!expected[@]}"; do
  install -D -m 0644 "$overlay_root/$relative" "$stage/$relative"
done
install -D -m 0755 "$helper" "$stage/scripts/nc_rted_segmented_materialization.py"
install -D -m 0755 "$controller" "$stage/scripts/nc_rted_source39_segmented_formal_v1.py"
for relative in scripts/nc_rted_interleaved_formal.py scripts/nc_rted_queue.py src/nc_rted/queue.py src/nc_rted/resource_attestation.py; do
  printf '%s  %s\n' "${expected[$relative]}" "$relative"
done > "$stage/SEGMENTED_STAGE_SHA256SUMS"
printf '%s  %s\n' "$helper_sha" scripts/nc_rted_segmented_materialization.py >> "$stage/SEGMENTED_STAGE_SHA256SUMS"
printf '%s  %s\n' "$controller_sha" scripts/nc_rted_source39_segmented_formal_v1.py >> "$stage/SEGMENTED_STAGE_SHA256SUMS"
(cd "$stage" && sha256sum -c SEGMENTED_STAGE_SHA256SUMS)
mv "$stage" "$destination"
trap - EXIT
