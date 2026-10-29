# YOLO26 Safety Tools

本目录包含 3 个独立的安全检测脚本，只使用已有 YOLO26 权重推理，不会重新训练模型。

WebUI 两个页面默认使用项目根目录的 17 类 OBB 模型 `all_best.pt`，也可以分别选择同一个模型文件。安全监测只需选择一份权重，即可覆盖全部七项功能。

两个页面及对应检测脚本统一使用 `imgsz=640`，覆盖权重保存的训练尺寸。图像等比缩放并补边，非正方形画面的张量不一定是 `640×640`。固定尺寸导出模型必须使用匹配的输入尺寸和后处理配置。

“统一安全监测”支持单选、多选或全选人员安全帽、叉车司机安全帽、叉车作业半径站人、叉车作业围蔽、叉车速度、吊臂下方人员检测和吊装作业围栏及安全监护检测。每个任务只加载一个模型，每个采样帧只检测、跟踪一次；吊臂复用同一份 OBB 结果，其他安全规则读取其外接矩形。各功能分别应用置信度阈值，并按模型 `names` 校验和筛选类别。异步接口为 `POST /api/detect/safety/start`；单张图片不支持叉车速度检测。

新增 `checks=lifting_operation`：每台 `crane` 下方同时检测到安全员和 `fence` 为 `safe`，缺少任一项为 `unsafe`；无起重机为 `unknown`。下方使用最简单的二维判断：目标检测框中心 Y 大于起重机检测框中心 Y，不限制水平距离、不检查围栏闭合。类别名同时接受 `safety officer` 和 `safety_officer`，使用页面统一 `conf`，每帧立即判定。结果位于 `results.lifting_operation`，含每台起重机的缺项和关联检测框；缺少安全员、缺少围栏分别产生 `NO_SAFETY_OFFICER_BELOW_CRANE`、`NO_FENCE_BELOW_CRANE` 告警，沿用按轨迹去重和隐患截图流程。多个起重机逐个判断，同一帧下方的防护目标可以复用。

首次推理前设置 `end2end=False`，使用 one-to-many 检测分支和 Ultralytics 内置 OBB NMS。项目自定义 NMS 已移除，页面不再提供 NMS 参数；后端统一配置 `iou=0.7`、`agnostic_nms=False`。围栏业务参数 `fence_iou_thres` 继续保留。切换分支会影响检测结果，应结合现场样本验证。

“运行日志”每个采样帧记录一份模型结果，采集于内置 NMS 之后、跟踪之前；包含类别、置信度、外接矩形和 OBB 四点坐标。无目标时显示“未检测到目标”。任务状态支持 `after_log_id` 增量读取 `detection_logs`，每个任务保留最近 200 条帧日志。

直播先由后台完成 WHEP 首帧接入，再加载模型。ICE/DTLS 传输建立与首帧解码分别计时；可在尚未收到 RTP 时根据 SDP 中的主视频 SSRC 请求关键帧。等待状态和 `stream_diagnostics` 会区分传输未连接、未收到 RTP 和收到 RTP 但无解码画面，并包含收包、丢包、解码无输出和 PLI 计数。无包时检查媒体服务器公告 IP、UDP 端口和推流状态；有包无画面时检查 H.264 IDR/SPS/PPS 和推流端 PLI 响应。仍无法恢复时有界超时并释放任务资源。

物资计数的文件输入与直播流输入也已合并到同一页面。选择直播流时显示直播原始画面和实时检测结果；选择单图、图片序列或视频时隐藏原始直播画面，只显示检测结果。

界面默认参数：统一安全监测的采样间隔为 `5` 帧、置信度为 `0.55`；物资计数的采样间隔为 `5` 帧、置信度为 `0.5`、设备为 `0`，跟踪高阈值 `0.3`、低阈值 `0.1`、新轨迹阈值 `0.4`、缓存 `50` 帧、匹配阈值 `0.25`，启用 ReID。页面、接口和命令行的默认采样间隔均为 `5`，仍可显式指定其他间隔。物资计数接口省略这些参数时使用相同默认值，设备不可用时仍按已有逻辑回退。

当前界面提交响应、任务状态、七个可选安全功能、物资计数以及所有嵌套 JSON 字段的类型和含义，统一以项目根目录 [README.md](../README.md#当前界面算法输出字段完整说明) 中的列表为准。

默认模型类别（业务映射以模型实际 `names` 为准）：

```text
0 person
1 helmet
2 head_no_helmet
3 forklift
4 fence
5 boom
6 large cable
7 4D distribution box
8 6D distribution box
9 200kV Transformer
10 400kV Transformer
11 400kV distribution box
12 500kV distribution box
13 630kV Transformer
14 630kV Transformer NX2
15 safety officer
16 crane
```

## 安装依赖

在仓库根目录或本目录安装：

```bash
pip install -r safety_tools/requirements.txt
```

如果直接使用当前 Ultralytics 源码仓库，建议从仓库根目录运行脚本，确保优先导入本地 `ultralytics` 包。

## 1. 安全帽检测

脚本：`helmet_check.py`

功能：

- 检测输入图像中的 `person`、`helmet`、`head_no_helmet`。
- 对每个 `person` 取上半部分头部区域。
- 头部区域与 `head_no_helmet` 明显重叠，判断为 `no_helmet`；否则，有匹配的 `helmet` 时判断为 `helmet`。
- 只检测到人员、没有匹配安全帽时，判断为 `no_helmet`。
- 只有 `head_no_helmet`、没有关联人员框时，也作为一条 `no_helmet` 人员记录计入隐患，框和轨迹 ID 使用该头部检测结果；通过 `detection_source=head_no_helmet` 标识。已关联人员的头部不会另计一人。
- 人员及叉车司机共用该安全帽判定。单张图片直接判定；视频、直播流和图片序列要求同一轨迹连续 5 个实际检测帧未戴帽，才确认 `status=no_helmet`、判定不安全并生成告警和证据图。前 4 帧为 `unknown`，原始未戴帽判断保留在 `raw_status`；戴帽、状态未知或在某个检测帧漏检都会重新计数，不同轨迹独立计数，无轨迹记录不能累计确认。只有安全帽或没有人员/未戴帽头部时，整体结果仍为 `unknown`。
- 五帧按实际算法检测次数计算，不按视频原始帧号差累计。例如每 15 帧采样一次，第 1、16、31、46、61 帧均未戴帽，才在第 61 帧确认。确认后仍按违规类型和轨迹 ID 去重，同一任务中已上报的轨迹不重复告警。

运行示例：

```bash
python safety_tools/helmet_check.py --weights all_best.pt --image test.jpg --output outputs/helmet_result.jpg --json outputs/helmet_result.json
```

不保存 JSON 文件，直接打印 JSON 到终端：

```bash
python safety_tools/helmet_check.py --weights all_best.pt --image test.jpg --output outputs/helmet_result.jpg
```

可选参数：

```bash
--conf 0.35
--no-helmet-overlap 0.2
```

JSON 输出包含：

- `total_persons`
- `helmet_worn_count`
- `no_helmet_count`
- `unknown_count`
- 每个人的 `box`、`status`、`confidence`、关联安全帽框、关联未戴帽头部框

## 2. 叉车作业检测

脚本：`forklift_work_check.py`

功能：

- 检测输入图像中的 `forklift`、`person`、`fence`。
- 同时检测 `helmet`、`head_no_helmet`，输出与安全帽脚本一致的人员佩戴统计和 `persons` 明细。
- 对每台叉车取检测框底部中心点作为地面点，并基于 `ground_depth_px` 生成更贴近地面的底部锚定作业区域 `work_zone`，适配无人机斜拍图像。
- 根据人员地面点和叉车地面点的归一化相对位置判断驾驶员或车上人员，并写入 `excluded_persons`，不触发人员靠近违规。
- 人员底部中心点落入矩形 `work_zone`、与叉车地面点距离小于动态阈值、或落入围栏多边形 `fence_based_work_zone`，任一条件满足即判定为叉车周围有人。
- 可选生成全局 `ground_point_work_area`：使用所有叉车和围栏检测框底边中点生成凸包区域，作为额外人员靠近判断条件。
- 可选生成启发式 `road_surface_mask`，用路面区域约束 `ground_point_work_area`，得到 `road_constrained_work_area`，减少区域覆盖到墙面、建筑或非路面区域。
- 可选估计叉车附近路面主方向，生成 `road_aligned_work_zone`，让叉车作业区沿道路走势倾斜绘制。
- 围栏框与 `work_zone` 的 IoU 大于阈值，或围栏中心点落入 `work_zone`，判定为叉车周围有围栏。
- 输出每台叉车的风险结果和违规项。

运行示例：

```bash
python safety_tools/forklift_work_check.py --weights all_best.pt --image test.jpg --output outputs/forklift_work_result.jpg --json outputs/forklift_work_result.json
```

不保存 JSON 文件，直接打印 JSON 到终端：

```bash
python safety_tools/forklift_work_check.py --weights all_best.pt --image test.jpg --output outputs/forklift_work_result.jpg
```

可选参数：

```bash
--conf 0.35
--no-helmet-overlap 0.2
--expand-scale 2.0
--left-scale 1.2
--right-scale 1.2
--front-scale 1.0
--back-scale 0.35
--height-depth-ratio 0.6
--width-depth-ratio 0.8
--fence-iou-thres 0.01
--driver-iou-thres 0.05
--driver-inside-ratio-thres 0.4
--driver-dx-norm-thres 0.45
--driver-min-dy-up-norm 0.18
--driver-max-dy-up-norm 0.85
--person-distance-width-scale 1.2
--person-distance-height-scale 0.8
--min-person-distance-px 180
--enable-fence-zone true
--fence-zone-padding 40
--enable-ground-point-work-area true
--ground-point-work-area-padding 80
--fallback-zone-radius 220
--draw-ground-point-work-area true
--enable-road-mask true
--draw-road-mask true
--road-sat-max 80
--road-value-min 25
--road-value-max 220
--road-morph-kernel 15
--road-min-area-ratio 0.03
--enable-road-aligned-zone true
--draw-road-aligned-zone true
--road-direction-roi-scale 3.0
--road-direction-min-pixels 500
--road-zone-length-scale 2.2
--road-zone-width-scale 1.2
--road-zone-back-scale 0.6
--road-zone-intersect-road-mask true
```

其中 `--driver-iou-thres` 和 `--driver-inside-ratio-thres` 仅为兼容旧命令保留，当前驾驶员排除使用人员地面点相对叉车地面点的 `dx_norm/dy_up_norm`。

其中 `--expand-scale` 仅为兼容旧命令保留，新逻辑不再使用检测框中心等比例扩大区域。

违规项：

- `PERSON_NEAR_FORKLIFT`
- `NO_FENCE_NEAR_FORKLIFT`

### 视频与图像序列监测

Web 主界面的“统一安全监测”支持：

- 单张图片
- 视频文件
- 图像序列文件夹
- 直播流

视频和图像序列通过 `safety_video_monitor.py` 逐帧处理，使用
`model.track(..., tracker="botsort.yaml", persist=True)` 为人员和叉车维护轨迹 ID。
任务运行期间，右侧预览区域显示当前标注帧，状态区域显示当前人数、佩戴/未佩戴安全帽人数、叉车数量和当前作业安全状态。证据图按照“违规类型 + 轨迹 ID”做任务级去重：同一人员在同一视频或直播连接中持续、间歇或再次被判为未戴安全帽都只保存一张；不同人员各自保存一张。重新启动任务后重新计数。安全监测不保存整段结果视频，只保存汇总 JSON 和隐患证据截图；任务也可以通过界面的“停止监测”按钮提前结束。

## 3. 叉车行驶像素速度检测

脚本：`forklift_speed_check.py`

功能：

- 使用 Ultralytics `model.track(..., tracker="botsort.yaml", persist=True)` 对视频中的叉车进行跟踪。
- 只检测和计算 `forklift` 类别。
- 使用叉车框底部中心点作为地面点。
- 通过相邻帧背景特征点估计相机/无人机全局运动，再计算叉车相对地面的像素速度。
- 速度单位为 `pixel/frame`，不会换算为真实速度。

运动补偿流程：

- 使用 ORB 提取相邻帧背景特征。
- 用叉车检测框生成 mask，过滤叉车内部特征点。
- 匹配背景特征点。
- 根据 `--motion-model` 估计 `affine` 或 `homography`。
- 将上一帧叉车地面点映射到当前帧坐标系。
- 当前地面点与补偿后的上一帧地面点距离即为当前像素速度。
- 背景特征不足或估计失败时，回退到普通图像坐标位移，并在 JSON 中标记 `motion_compensation_success=false`。

运行示例：

```bash
python safety_tools/forklift_speed_check.py --weights all_best.pt --video forklift.mp4 --output outputs/forklift_speed_result.mp4 --json outputs/forklift_speed_result.json
```

不保存 JSON 文件，直接打印 JSON 到终端：

```bash
python safety_tools/forklift_speed_check.py --weights all_best.pt --video forklift.mp4 --output outputs/forklift_speed_result.mp4
```

可选参数：

```bash
--conf 0.35
--speed-window 5
--min-bg-matches 20
--motion-model affine
```

JSON 输出包含：

- `video`
- `fps`
- 每帧的 `frame_id`
- `motion_compensation_success`
- 每台叉车的 `track_id`、`box`、`ground_point`、`current_speed_px_per_frame`、`smoothed_speed_px_per_frame`

## 4. 本地视频模拟直播流

脚本：`mock_live_camera.py`

没有真实摄像头或流媒体服务器时，可以让一段视频文件扮演摄像头：脚本按原始帧率解码、循环播放，并通过 WHEP/WebRTC 输出 H.264 或 VP8 视频轨。安全监测后台仍然走真实的 SDP/ICE/DTLS/RTP 拉流链路，与接入真实摄像头完全一致。

不依赖 ffmpeg 或 MediaMTX，只使用项目已有的 `PyAV`、`aiortc` 和 `aiohttp`。

启动模拟摄像头：

```bash
python safety_tools/mock_live_camera.py --video demo.mp4
```

默认地址：

```text
http://127.0.0.1:8889/whep
```

使用步骤：

1. 启动脚本后，在 WebUI 选择“直播流”，填入上面的地址。
2. 先点“测试 Python 后端拉流”，确认原始画面连续显示。
3. 再点“开始监测”或“开始计数”，即可用该模拟画面跑完整算法。

常用参数：

```bash
--host 0.0.0.0          # 允许局域网内其他电脑拉流
--port 8889             # 监听端口
--path whep             # WHEP 路径，最终地址为 http://host:port/<path>
--fps 25                # 覆盖视频原始帧率
--width 1280 --height 720
                        # 输出画面最大尺寸，等比缩放，默认不超过 720p
--no-resize             # 保持视频原始分辨率
--codec vp8             # 改用 VP8（默认 H.264）
--bitrate 2000000       # 目标码率，单位 bit/s
--no-loop               # 只播放一次（默认循环）
```

说明：

- 模拟器只服务本机或局域网，不启用 STUN；跨公网/NAT 场景需要改用真实流媒体服务器。
- 每个拉流连接独立解码和编码，连接建立后从视频开头播放，默认循环。
- H.264 使用 `libx264` 软编码，建议输出不超过 720p；视频过大时脚本会自动等比缩放。
- 输出日志和 `GET /healthz` 可用于查看当前会话数、已发送帧数和运行时长。
- 拉流端主动断开或停止检测时，模拟器会立即释放对应会话；如果拉流进程被强制结束、来不及发送 `DELETE`，看门狗会在 RTCP 接收者报告静默约 15 秒后回收该会话，避免继续满速编码。自定义拉流端需要按标准周期回送 RTCP RR，否则可能被判定为已断开。

## 输出说明

所有脚本都会自动创建输出目录。`--json` 是可选参数：传入时保存 JSON 文件，不传时直接打印 JSON 到终端。JSON 使用 `ensure_ascii=False`，中文路径和中文内容不会被转义。

安全监测输出标注图片、隐患证据截图和 JSON，不保存整段结果视频。物资计数的视频、图片序列和直播流模式会输出采用 H.264 压缩的 `.mp4`。

## 注意事项

- 检测效果依赖 `all_best.pt` 对固定 5 类的训练质量。
- 安全帽关联逻辑是基于检测框几何关系的启发式规则，拥挤遮挡场景可能需要调节 `--conf` 和 `--no-helmet-overlap`。
- 叉车作业区域是 2D 图像框扩大区域，不等价于真实世界距离。
- 视频速度为相对地面的像素速度。若画面背景纹理少、运动模糊严重、叉车占画面过大，运动补偿可能失败，此时会自动 fallback 到图像坐标位移。
