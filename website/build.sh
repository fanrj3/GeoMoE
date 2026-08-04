#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_root}/_site"

rm -rf -- "${output_dir}"
mkdir -p "${output_dir}/assets"

cp "${repo_root}/website/index.html" "${output_dir}/index.html"
cp "${repo_root}/website/styles.css" "${output_dir}/styles.css"
cp "${repo_root}/website/script.js" "${output_dir}/script.js"
cp "${repo_root}/assets/experiment.svg" "${output_dir}/assets/experiment.svg"
cp "${repo_root}/assets/pipeline.svg" "${output_dir}/assets/pipeline.svg"
cp "${repo_root}/assets/flag.svg" "${output_dir}/assets/flag.svg"
cp "${repo_root}/assets/vigormdataset.svg" "${output_dir}/assets/vigormdataset.svg"
touch "${output_dir}/.nojekyll"

echo "Built ${output_dir}"
