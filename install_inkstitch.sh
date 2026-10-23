#!/usr/bin/env bash
set -euo pipefail

echo "Installing Inkscape and Ink/Stitch..."
sudo apt-get update
sudo apt-get install -y inkscape curl

case "$(uname -m)" in
  x86_64) asset_pattern='inkstitch_[0-9.]*_amd64.deb' ;;
  aarch64|arm64) asset_pattern='inkstitch_[0-9.]*_arm64.deb' ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

release_json="$(curl -fsSL https://api.github.com/repos/inkstitch/inkstitch/releases/latest)"
asset_url="$(printf '%s' "$release_json" \
  | sed -n "s|.*\"browser_download_url\": \"\\(https://[^\"]*${asset_pattern}\\)\".*|\\1|p" \
  | head -n 1)"

if [[ -z "$asset_url" ]]; then
  echo "Could not locate the latest Ink/Stitch Debian package for $(uname -m)." >&2
  echo "Download one manually from https://github.com/inkstitch/inkstitch/releases" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
curl -fL "$asset_url" -o "$tmp_dir/inkstitch.deb"
sudo apt-get install -y "$tmp_dir/inkstitch.deb"
echo "Ink/Stitch installed. Restart Inkscape, then use Extensions > Ink/Stitch > Simulator."
