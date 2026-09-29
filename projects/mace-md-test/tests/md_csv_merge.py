#!/usr/bin/env python3
"""
merge_logs.py

将 data/raw 目录下分段生成的 MD 日志文件合并为一个完整连续的 CSV 文件。
自动提取步数排序，并修正时间轴偏移量（time_fs）。

用法：
    python tests/md_csv_merge.py --model medium-mpa-0 --temp 600
"""

import argparse
import glob
import os
import re

import pandas as pd


def merge_md_logs(
    data_dir: str = os.path.join("data", "raw"),
    model: str = "medium-mpa-0",
    temp: float = 600.0,
    out_csv: str = None,
):
  # 1. 检索符合条件的 log 文件
  pattern = os.path.join(data_dir, f"md_{model}_{int(temp)}K_*steps_log.csv")
  files = glob.glob(pattern)

  if not files:
    print(f"❌ 未在 {data_dir} 目录下找到匹配 pattern 的日志文件: {pattern}")
    return

  # 2. 提取文件名中的累计步数并排序
  file_info = []
  for f in files:
    match = re.search(r"(\d+)steps_log\.csv$", f)
    if match:
      steps = int(match.group(1))
      file_info.append((steps, f))

  # 按步数从小到大排序
  file_info.sort(key=lambda x: x[0])
  print(f"🔍 找到 {len(file_info)} 个分段日志文件，将按以下顺序合并：")
  for steps, filepath in file_info:
    print(f"  - [{steps} steps]: {filepath}")

  # 3. 逐个读取并修复时间轴后合并
  merged_dfs = []
  time_offset = 0.0  # 时间累积偏移量 (fs)

  for _i, (_steps, filepath) in enumerate(file_info):
    df = pd.read_csv(filepath)

    if df.empty:
      continue

    # 对时间列加上累积偏移量，保持时间轴连续
    df["time_fs"] = df["time_fs"] + time_offset

    # 更新下一个分段的时间偏移量（取当前分段最后一帧的时间）
    time_offset = df["time_fs"].iloc[-1]

    merged_dfs.append(df)

  # 4. 拼接所有 DataFrame
  final_df = pd.concat(merged_dfs, ignore_index=True)

  # 5. 保存合并后的 CSV
  if out_csv is None:
    total_steps = file_info[-1][0]
    out_csv = os.path.join(
        data_dir,
        f"md_{model}_{int(temp)}K_merged_total_{total_steps}steps_log.csv",
    )

  final_df.to_csv(out_csv, index=False)

  total_time_ps = final_df["time_fs"].iloc[-1] / 1000.0
  print(
      f"\n✅ 成功合并日志！\n- 总数据行数: {len(final_df)}\n- 总模拟时长:"
      f" {total_time_ps:.2f} ps\n- 输出文件: {out_csv}"
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="合并 data/raw 下分段运行的 MD 日志文件"
  )
  parser.add_argument(
      "--data-dir",
      default=os.path.join("results", "raw","md_restart"),
      help="日志所在的文件夹路径",
  )
  parser.add_argument("--model", default="medium-mpa-0", help="使用的 MACE 模型名")
  parser.add_argument("--temp", type=float, default=600.0, help="模拟温度")
  parser.add_argument(
      "--out", default=None, help="自定义合并后的输出文件名（可选）"
  )

  args = parser.parse_args()

  merge_md_logs(
      data_dir=args.data_dir,
      model=args.model,
      temp=args.temp,
      out_csv=args.out,
  )