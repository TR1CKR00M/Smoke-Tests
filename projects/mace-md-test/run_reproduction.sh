#!/usr/bin/env bash
set -e  # 任何命令报错时立即退出脚本

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

#pip install -r requirements.txt

python tests/run_foundation_md.py --model medium-mpa-0 --temp 600 --steps 2000
python tests/run_foundation_md.py --model medium-0b3 --temp 600 --steps 2000
python tests/plot_temp_pressure.py

if [ ! -f "results/figures/multi_file_comparison.png" ]; then
    echo "错误：未生成预期对比图！"
    exit 1
fi

echo "=== 复现验证成功！ ==="