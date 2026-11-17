# diagnosis.py (精度自动对齐 + Mac 兼容终极版)
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

# 1. 设置全局默认精度为 float64，与 MACE 训练默认精度保持一致
torch.set_default_dtype(torch.float64)

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.data import AtomicData, config_from_atoms
from mace.tools import torch_geometric, AtomicNumberTable
from dscribe.descriptors import SOAP
from sklearn.ensemble import GradientBoostingRegressor
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# 2. 拦截 e3nn 内部未传 map_location 的 torch.jit.load，强制映射到 CPU
device = "cpu"
_orig_jit_load = torch.jit.load


def _cpu_jit_load(f, map_location=None, **kwargs):
    return _orig_jit_load(f, map_location=torch.device("cpu"), **kwargs)


torch.jit.load = _cpu_jit_load

train_atoms = read(DATA_DIR / "defect_train_300K.xyz", index=":")
test_1200K = read(DATA_DIR / "defect_test_1200K.xyz", index=":")

models = ["mace_k128_L0", "mace_k32_L0", "mace_k128_L2", "mace_k32_L2"]

# 预先计算 1200K 测试集的全局 SOAP 特征 (区分 l=0,1,2,3 角动量通道)
species = ["Cr", "Mo", "S"]
soap_desc = SOAP(species=species, r_cut=5.5, n_max=6, l_max=3, average="inner", periodic=True)
X_soap = soap_desc.create(test_1200K)

feat_names = []
for s1_i in range(len(species)):
    for s2_i in range(s1_i, len(species)):
        n_max_pairs = 6 * (6 + 1) // 2 if s1_i == s2_i else 6 * 6
        for n_pair in range(n_max_pairs):
            for l in range(4):
                feat_names.append(f"{species[s1_i]}-{species[s2_i]}_n{n_pair}_l{l}")
feat_names = feat_names[:X_soap.shape[1]]


def evaluate_and_gradcam(model_path, frames):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    # 3. 动态检测当前模型的权重精度 (float32 还是 float64)，确保输入数据 100% 匹配
    model_dtype = next(model.parameters()).dtype
    torch.set_default_dtype(model_dtype)

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max)

    e_errors, f_errors, selectivity_ratios = [], [], []

    activations = []

    def fwd_hook(module, inp, out):
        feat = out[0] if isinstance(out, tuple) else out
        activations.append(feat)
        feat.retain_grad()

    handle = model.interactions[-1].register_forward_hook(fwd_hook)

    for atoms in frames:
        activations.clear()
        config = config_from_atoms(atoms)
        data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
        batch = torch_geometric.Batch.from_data_list([data]).to(device)

        # 确保 Batch 内所有浮点张量与模型权重 dtype 完全一致
        batch_dict = batch.to_dict()
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch_dict[k] = v.to(dtype=model_dtype)
        out = model(batch_dict, compute_force=True, training=True)

        out = model(batch_dict, compute_force=True)
        pred_e = out["energy"]
        pred_f = out["forces"].detach().cpu().numpy()

        # 记录能量与力误差
        ref_e = atoms.info["REF_energy"]
        ref_f = atoms.arrays["REF_forces"]
        e_errors.append(abs(pred_e.item() - ref_e) / len(atoms) * 1000)  # meV/atom
        f_errors.append(np.sqrt(np.mean((pred_f - ref_f) ** 2)) * 1000)  # meV/A

        # 计算兼容高阶张量 (L=2) 的 Grad-CAM 原子激活度
        model.zero_grad()
        pred_e.backward(retain_graph=True)
        A = activations[0]  # shape: [N_atoms, N_irreps_channels]
        dA = A.grad  # shape: [N_atoms, N_irreps_channels]

        weights = dA.abs().mean(dim=0, keepdim=True)
        cam = (weights * A.abs()).sum(dim=-1).detach().cpu().numpy()

        # 区分缺陷中心 (Cr 及其 3.5 A 内第一配位壳层) vs 远端基质 (Bulk Mo/S)
        cr_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Cr"]
        i_list, j_list = neighbor_list("ij", atoms, cutoff=3.5)
        defect_zone = set(cr_idx)
        for i_a, j_a in zip(i_list, j_list):
            if i_a in cr_idx:
                defect_zone.add(j_a)
        bulk_zone = [i for i in range(len(atoms)) if i not in defect_zone]

        sel_ratio = (np.mean(cam[list(defect_zone)]) + 1e-8) / (np.mean(cam[bulk_zone]) + 1e-8)
        selectivity_ratios.append(sel_ratio)

    handle.remove()
    return np.array(e_errors), np.array(f_errors), np.mean(selectivity_ratios)


print(
    f"{'Model Name':<15} | {'300K F-RMSE':<12} | {'1200K F-RMSE':<12} | {'Overfit Ratio':<13} | {'Grad-CAM S_def':<14} | {'SHAP Cr(l>=2) Share'}")
print("-" * 95)

for m_name in models:
    m_path = MODEL_DIR / f"{m_name}_stagetwo.model"
    _, f_train, _ = evaluate_and_gradcam(m_path, train_atoms[:30])
    e_1200, f_1200, s_def = evaluate_and_gradcam(m_path, test_1200K)

    overfit_ratio = np.mean(f_1200) / np.mean(f_train)

    # Level 2: GBDT + TreeSHAP 拟合误差分布
    gbdt = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    gbdt.fit(X_soap, e_1200)
    explainer = shap.TreeExplainer(gbdt)
    shap_vals = np.abs(explainer.shap_values(X_soap)).mean(axis=0)

    cr_high_l_mask = [("Cr" in fn and ("_l2" in fn or "_l3" in fn)) for fn in feat_names]
    cr_high_l_share = shap_vals[cr_high_l_mask].sum() / (shap_vals.sum() + 1e-8) * 100

    print(
        f"{m_name:<15} | {np.mean(f_train):>8.1f} meV/A | {np.mean(f_1200):>8.1f} meV/A | {overfit_ratio:>11.2f}x | {s_def:>12.2f}x | {cr_high_l_share:>10.1f}%")
