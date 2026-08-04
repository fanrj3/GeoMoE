#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_root}/_site"

rm -rf -- "${output_dir}"
mkdir -p "${output_dir}/assets/icons" "${output_dir}/assets/map-zoom"

cp "${repo_root}/website/index.html" "${output_dir}/index.html"
cp "${repo_root}/website/styles.css" "${output_dir}/styles.css"
cp "${repo_root}/website/script.js" "${output_dir}/script.js"
cp "${repo_root}/assets/experiment.svg" "${output_dir}/assets/experiment.svg"
cp "${repo_root}/assets/pipeline.svg" "${output_dir}/assets/pipeline.svg"
cp "${repo_root}/assets/flag.svg" "${output_dir}/assets/flag.svg"
cp "${repo_root}/assets/vigormdataset.svg" "${output_dir}/assets/vigormdataset.svg"
cp "${repo_root}/assets/geographical.png" "${output_dir}/assets/geographical.png"
cp "${repo_root}/assets/icons/arxiv.svg" "${output_dir}/assets/icons/arxiv.svg"
cp "${repo_root}/assets/icons/github.svg" "${output_dir}/assets/icons/github.svg"
cp "${repo_root}/assets/icons/huggingface.svg" "${output_dir}/assets/icons/huggingface.svg"
cp "${repo_root}/assets/map-zoom/seattle-l0.webp" "${output_dir}/assets/map-zoom/seattle-l0.webp"
cp "${repo_root}/assets/map-zoom/seattle-l1.webp" "${output_dir}/assets/map-zoom/seattle-l1.webp"
cp "${repo_root}/assets/map-zoom/seattle-l2.webp" "${output_dir}/assets/map-zoom/seattle-l2.webp"
cp "${repo_root}/assets/map-zoom/seattle-l3.webp" "${output_dir}/assets/map-zoom/seattle-l3.webp"
touch "${output_dir}/.nojekyll"

echo "Built ${output_dir}"
