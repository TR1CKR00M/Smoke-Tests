# step1_generate_defect_data.py
import numpy as np
from pathlib import Path
from ase.build import mx2
from ase.io import write
from mace.calculators import mace_mp

np.random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 1. 构建 4x4x1 MoS2 超胞 (48 原子) 并引入 2 个相邻 Cr 替位点缺陷
base_atoms = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(4, 4, 1), vacuum=10.0)
base_atoms.set_pbc([True, True, True])

# 将中心区域的两个相邻 Mo 原子替换为 Cr (制造 Cr-Cr 掺杂对活性中心)
mo_indices = [i for i, sym in enumerate(base_atoms.get_chemical_symbols()) if sym == 'Mo']
defect_sites = [mo_indices[5], mo_indices[6]]
symbols = base_atoms.get_chemical_symbols()
for idx in defect_sites:
    symbols[idx] = 'Cr'
base_atoms.set_chemical_symbols(symbols)

# 2. 加载官方 MACE-MP-0 Large (k=128, L_max=2) 作为高阶物理 Oracle
print("正在加载 MACE-MP-0 (large) 作为 Ground Truth 生成器...")
# MACE/PyTorch builds that expose MPS may still lack torch.mps.current_device,
# which some MACE versions call internally. Prefer CUDA when available and
# otherwise use CPU rather than selecting an unsupported MPS path.
import torch

compute_device = "cuda" if torch.cuda.is_available() else "cpu"
oracle_calc = mace_mp(model="large", device=compute_device, default_dtype="float32")


def generate_frames(n_frames, rattle_std, temp_label):
    frames = []
    for i in range(n_frames):
        atoms = base_atoms.copy()
        # 施加随机热位移
        atoms.rattle(stdev=rattle_std, seed=np.random.randint(1e6))
        # 对缺陷原子额外施加微小的局部畸变，模拟局域极化/迁移趋势
        pos = atoms.get_positions()
        pos[defect_sites] += np.random.normal(0, rattle_std * 1.3, size=(2, 3))
        atoms.set_positions(pos)

        atoms.calc = oracle_calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()

        # 存为 MACE 标准读取字段 REF_energy / REF_forces
        atoms.calc = None
        atoms.info["REF_energy"] = e
        atoms.Info_temp = temp_label
        atoms.arrays["REF_forces"] = f
        frames.append(atoms)
    return frames


print("正在生成 300K 训练/验证集与 1200K 测试集...")
train_300K = generate_frames(150, rattle_std=0.04, temp_label="300K")  # 150 帧低温训练
val_300K = generate_frames(30, rattle_std=0.04, temp_label="300K")  # 30 帧低温验证
test_300K = generate_frames(50, rattle_std=0.04, temp_label="300K")  # 50 帧同分布测试
test_1200K = generate_frames(50, rattle_std=0.12, temp_label="1200K")  # 50 帧高温 OOD 测试

DATA_DIR.mkdir(parents=True, exist_ok=True)
write(DATA_DIR / "defect_train_300K.xyz", train_300K)
write(DATA_DIR / "defect_val_300K.xyz", val_300K)
write(DATA_DIR / "defect_test_300K.xyz", test_300K)
write(DATA_DIR / "defect_test_1200K.xyz", test_1200K)
print("数据集生成完毕！")
