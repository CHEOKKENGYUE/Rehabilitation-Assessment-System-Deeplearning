# 智能康复评估平台（本地精简版）

EEG / EMG / IMU 多模态融合康复评估系统：**FastAPI 后端 + React 前端**。深度学习模型 **CMK-AGN** 预测 **3 项**手部康复指标。

> 本版本为**纯本地运行**的精简版：只做 3 项深度学习预测，**不含** Barthel 指数（BI）预测、26 项数字生物标志物提取、以及大模型（LLM）康复报告生成。全流程可在 CPU 上运行，**无需 GPU、无需 MySQL、无需任何远程服务**。

## 3 项康复指标

| 任务键 | 临床量表 | 类型 |
|---|---|---|
| `FMA_UE` | FMA 手部评分 | 回归 0–20 |
| `hand_tone` | Hand MAS（Modified Ashworth 手部肌张力） | 6 级 `0/1/1+/2/3/4` |
| `hand_function` | Brunnstrom 分期（手） | 分期 1–6 |

## 目录结构

```
backend/          FastAPI 服务（main / inference / schemas / db）
frontend/         Vite + React + TS 单页前端
DL_model/         已训练的 .pth 模型（3 个任务各自最优 fold）
Deeplearning/     预处理 / 模型代码（被 backend 复用）
```

## 环境要求

- Python 3.10+、Node.js 18+
- 纯 CPU 即可运行（如 Mac）；有 GPU 时 PyTorch 会自动使用，但不是必需

## 启动后端

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动时会从 `../DL_model/*.pth` 加载 3 个 CMK-AGN 模型；日志应显示
`loaded 3 models: ['FMA_UE', 'hand_tone', 'hand_function']`。
`GET /api/health` 返回已加载的任务列表。首次启动会在 `backend/rehab.db`
自动建立 SQLite 库（已 gitignore）。

## 启动前端

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Vite 已配置代理 `/api → http://localhost:8000`，无需跨域配置。

## 使用流程

1. 登录（前端演示登录，输入任意账号即可进入）。
2. 填写患者基本信息（编号 / 姓名 / 性别 / 年龄 / 诊断 / 病程 / 偏瘫侧）。
3. 为每个 trial 上传一对 EEG CSV 和 EMG/IMU CSV（文件顺序需对应）。
4. 点击「开始评估」，前端通过 SSE 实时显示处理进度与 3 项预测结果。
5. 完成后可重新评估或查看患者档案。每次评估写入 `backend/rehab.db`。

## 主要 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/assess` | multipart 上传患者信息 + EEG/EMG 文件，返回 `session_id` |
| `GET` | `/api/assess/{session_id}/stream` | SSE 推送处理进度与预测结果 |
| `GET` | `/api/assess/{session_id}/result` | 断线重连后获取最终结果 |
| `GET` | `/api/patients`、`/api/assessments`、`/api/stats/summary` | 患者 / 评估记录 / 统计 |
| `GET` | `/api/health` | 健康检查 |
