# 物资计数系统（YOLO26 + BoT-SORT）

基于 YOLO 目标检测/OBB 模型与 BoT-SORT 多目标跟踪的物资计数系统。网页端上传图片、图片序列、视频或直播流，系统逐帧检测并跟踪目标，按类别统计去重后的数量，输出标注画面、JSON 与 CSV。

> 历史版本中「统一安全监测」页面已从网页移除。后端相关接口与算法脚本仍保留在仓库内，但默认不接入网页，详见文末「保留的安全监测后端」。

## 功能

- 输入：单张图片、图片序列（文件夹）、视频文件、WebRTC/WHEP 直播流。
- 模型：支持 OBB 旋转框模型与普通目标检测模型，类别表按所选模型的 `names` 自动生成。
- 计数：按「类别 + 轨迹 ID」去重，同一目标在连续帧中只计一次；直播断线重连进入新的连接代次，避免轨迹 ID 冲突。
- 准确度：可为目标类别填写实际数量，任务结束后给出准确度。
- 输出：标注图片或 H.264 标注视频、JSON、CSV，运行中可看实时预览与分类计数。
- 模拟直播：网页一键启动本地模拟直播流（WHEP），无需外部推流设备。

## 目录结构

```text
safety_warehouse_detect/
├── safety_tools/
│   ├── safety_web_server.py     # Web 后端与 HTTP 接口
│   ├── web/index.html           # 单页前端
│   ├── material_count_cli.py    # 物资计数命令行/算法入口
│   ├── mock_live_camera.py      # 模拟直播摄像头（WHEP 服务）
│   ├── model_info.py            # 子进程读取模型类别，避免服务端加载 torch
│   ├── model_config.py          # 默认模型名、推理尺寸、NMS 参数
│   ├── common.py                # 模型加载与类别校验
│   ├── live_stream.py           # WebRTC/WHEP 拉流与帧分发
│   └── 其他脚本                  # 保留的安全监测算法，未接入网页
├── ultralytics/                 # 本地 ultralytics 源码（AGPL-3.0）
├── yolo26s-best.pt              # 默认物资模型（10 类）
├── all_best.pt                  # 17 类 OBB 模型（仅安全监测后端使用）
├── outputs/safety_web/          # 默认输出目录（不入库）
├── Videos/                      # 测试视频（不入库）
├── requirements_app.txt         # 运行依赖（torch 单独安装）
└── docs/archive/                # 历史文档
```

## 环境要求

- Python 3.9–3.12（当前验证版本 3.12）。
- 可选 NVIDIA GPU；无 GPU 时用 `device=cpu`。
- 现代浏览器（Chrome / Edge）。
- 不需要 Node.js。

## 安装

```bash
# 1. 按机器安装 torch
#    NVIDIA（CUDA 12.6）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#    纯 CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 其余依赖
pip install -r requirements_app.txt
```

`ultralytics` 直接使用仓库内的源码目录，不需要另行 `pip install ultralytics`。

## 启动

```bash
python safety_tools/safety_web_server.py
```

打开 `http://127.0.0.1:7860`。

局域网内其他电脑访问：

```bat
set SAFETY_WEB_HOST=0.0.0.0
set SAFETY_WEB_PORT=7860
python safety_tools\safety_web_server.py
```

放行端口（管理员）：

```bat
netsh advfirewall firewall add rule name="safety-web" dir=in action=allow protocol=TCP localport=7860
```

然后访问 `http://<服务器IP>:7860`。

## 页面使用流程

1. 选择输入类型：单张图片 / 图片序列 / 视频文件 / 直播流。
2. 选择输入：上传文件，或在 `source_path` 填服务器本地路径；图片序列填文件夹路径。
3. 选择检测模型（默认 `yolo26s-best.pt`）：可上传、可浏览服务器文件；选定后自动列出可计数类别。
4. 勾选需要计数的类别，可填写实际数量用于准确度统计。
5. 设置输出目录、置信度、device 与跟踪参数。
6. 点击「开始计数」，运行中可看实时预览、分类计数与日志；「停止计数」可中断。
7. 完成后结果区显示统计、标注文件、JSON 和「下载 CSV」链接。

### 直播流

- 真实设备：在「直播地址」填入厂商提供的 WHEP/SRT 地址，先点「测试 Python 后端拉流」，画面正常后再开始计数。
- 模拟直播：点「启动模拟直播流」→ 在服务器文件浏览器里选一个本地视频 → 后端拉起模拟摄像头，地址自动填入 `http://127.0.0.1:8889/whep` → 点「测试 Python 后端拉流」。
- 更换模拟视频需先点「停止模拟直播流」，再重新启动。

## 接口

| 接口 | 说明 |
|---|---|
| `GET /api/config` | 默认模型、输出目录、默认设备、GPU 信息 |
| `GET /api/health` | 服务与直播连接状态 |
| `GET/POST /api/model-classes` | 读取（或上传后读取）模型 task 与类别 |
| `POST /api/detect/material-count` | 提交计数任务，返回 `job_id` |
| `POST /api/detect/material-count-live` | 仅直播流输入的计数任务 |
| `GET /api/jobs/{job_id}` | 任务状态、帧状态、最终结果（支持 `after_frame_id` 增量） |
| `GET /api/jobs/{job_id}/stream` | 实时预览 MJPEG |
| `POST /api/jobs/{job_id}/cancel` | 停止任务（`?release_stream=1` 同时释放直播连接） |
| `POST /api/live-stream/test` | 测试直播拉流 |
| `POST /api/mock-stream/start` / `stop` | 启动/停止模拟直播摄像头 |
| `GET /api/browse` | 服务器文件浏览器（`kind=file\|dir\|video`） |
| `GET /api/file` | 下载输出文件 |

`POST /api/detect/material-count` 主要字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `source_type` | string | `single_image` / `image_sequence` / `video` / `live_stream` |
| `source_file` | file | 上传的图片、视频或图片序列压缩包 |
| `source_path` | string | 服务器本地输入路径 |
| `stream_url` | string | 直播地址（`live_stream` 必填） |
| `sample_interval` | int | 抽帧间隔，默认 `5` |
| `model_file` / `model` | file / string | 模型上传或服务器路径，默认 `yolo26s-best.pt` |
| `target_classes` | string | 必填，换行或逗号分隔，按模型 `names` 校验 |
| `expected_counts` | string | 每行 `类别名称,实际数量` |
| `output_dir` | string | 输出目录，默认 `outputs/safety_web` |
| `conf` | float | 置信度，默认 `0.5` |
| `device` | string | `auto` / `0` / `cpu`，无 GPU 自动回退 `cpu` |
| `track_high_thresh` `track_low_thresh` `new_track_thresh` `track_buffer` `match_thresh` | float/int | BoT-SORT 参数 |
| `with_reid` `proximity_thresh` `appearance_thresh` | bool/float | ReID 参数 |

## 输出文件

默认目录 `outputs/safety_web/`：

| 文件 | 说明 |
|---|---|
| `material_count_<时间戳>.jpg` | 单图计数标注结果 |
| `material_count_<时间戳>.mp4` | 图片序列、视频、直播的标注视频 |
| `material_count_<时间戳>.json` | 完整结果（类别数量、准确度行、逐帧记录等） |
| `material_count_<时间戳>.csv` | 类别 / 检测数量 / 实际数量 / 准确度（UTF-8 BOM，Excel 可直接打开） |
| `material_count_preview_<job>.jpg` | 运行中实时预览帧 |
| `.model_info_cache.json` | 模型类别缓存，按「路径 + 修改时间 + 大小」失效 |

## 命令行

```bash
python safety_tools/material_count_cli.py \
  --source-type video --source Videos/video1.mp4 \
  --model yolo26s-best.pt \
  --target-classes "large cable" \
  --output-media outputs/count.mp4 --json outputs/count.json \
  --conf 0.5 --device 0 --sample-interval 5
```

## 模型

| 模型 | 类别数 | 用途 |
|---|---|---|
| `yolo26s-best.pt` | 10 | 默认物资模型：大小电缆、配电箱、变压器等 |
| `all_best.pt` | 17 | OBB 模型，含人员/叉车等安全类别，仅安全监测后端使用 |

物资计数支持 `task=obb` 与 `task=detect`，其他任务类型会被拒绝。

## 保留的安全监测后端

`/api/detect/safety/start`、`/api/detect/helmet`、`/api/detect/forklift-work`、`/api/detect/forklift-speed` 等接口及 `helmet_check.py`、`crane_safety_check.py` 等脚本仍在仓库中，网页上已无入口。如需恢复界面，重新接入 `index.html` 即可，后端无需改动。

## 许可证

仓库内的 `ultralytics/` 为 Ultralytics 源码副本，遵循 **AGPL-3.0**。对本项目进行分发或作为网络服务提供时，需遵守 AGPL 的源码开放要求。
