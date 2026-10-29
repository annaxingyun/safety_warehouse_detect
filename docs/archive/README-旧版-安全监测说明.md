# 安全监测与物资识别算法工程说明文档

本文档说明当前算法工程的功能、文件结构、本地部署方式、接口调用方式以及 WebUI 使用方法。工程当前以 `HTML + CSS + JavaScript` 实现网页界面，以 Python 后端负责模型推理、视频处理、文件管理和接口服务。

当前后端支持 Python 3.9～3.12，生产环境应使用独立虚拟环境并严格按照 `requirements.txt` 安装，避免复用包含不同 NumPy/Torch 二进制版本的 Anaconda base 环境。

# 新增功能

## 输入源能力

1. 所有算法支持视频或图像序列输入。
2. 直播监测统一由 Python 后端通过 WebRTC/WHEP 拉流、解码和共享帧；浏览器只显示后端输出的预览画面。

## 本次安全监测工程化升级

1. 按“违规类型 + 目标轨迹 ID”识别独立违规事项；同一轨迹的同一种违规在整次视频或直播连接中只返回一次，即使中间有漏检或短暂恢复也不重复保存证据图。新目标轨迹仍会单独返回。
2. 每个出现新增违规的告警帧保存一张标注图片；同一帧的多项违规共用该图片，`latest_frame_url` 指向最新图片，并通过 `annotated_updated` 标识是否出现新的标注图片。
3. 违规标注结果图片保存在接口入参 `output_dir` 指定的输出目录中。

## 业务逻辑解释

当前系统可以实现“按独立违规事项返回告警并保存证据”的核心逻辑

举例说明：例如A没有佩戴安全帽，那么现在需要返回安全告警，并通知喊话器喊话（只播报一次），然后还要本地保存一张图片作为证据存档，过一会B进来了，没有带安全帽，此时也需要喊话器播报一次，并且保存一张图片作为证据，然后再过一会C叉车作业半径站人了，喊话器需要播报一次，同时保存一张图片作为证据，喊话器根据当前算法返回的字段播放安全预警录音，同时本地也要保存图片证据（包含违规发生的时间和监测结果标注）

| 需求                       | 当前状态                                         |
| -------------------------- | ------------------------------------------------ |
| A未戴安全帽只告警一次      | 已支持，按照违规类型和A的轨迹ID去重              |
| B进入后未戴安全帽再次告警  | 已支持，B具有新的轨迹ID，会产生新的违规事件      |
| 叉车作业半径有人再次告警   | 已支持，违规类型不同，会产生新的违规事件         |
| 每个新增违规批次保存证据图 | 已支持，同一帧的多项违规共用一张图片，图片保存在`output_dir`，文件名包含时间戳 |
| 图片包含YOLO原始检测框     | 已支持，不绘制文字、状态或规则辅助区域           |
| 返回违规发生时间           | 已支持，通过`hazard_timestamp`返回               |

建议喊话器控制程序轮询：

```
GET /api/jobs/{job_id}?after_frame_id={last_frame_id}
```

当`result_updated=true`时，读取：

```
{
  "code": "NO_HELMET",
  "message": "检测到人员未佩戴安全帽",
  "incident_id": "唯一违规事件ID",
  "incident_episode": 1,
  "track_id": 12,
  "hazard_timestamp": "2026-07-20T10:30:15.123456",
  "hazard_image_path": "output/helmet_20260720_103015_123456_frame_1800.jpg"
}
```

喊话器程序应当：

1. 使用`code`选择录音，例如`NO_HELMET`播放“请佩戴安全帽”。
2. 使用`incident_id`作为幂等键，确保同一个事件只播放一次。
3. 播放成功后再更新`last_frame_id`。
4. 如果一次检测帧同时出现多项新违规，需要遍历`hazards`逐项处理，不能只处理第一项。

因此，对于A、B和C场景，系统会生成3个独立违规事件、保存3张证据图片，也具备驱动3次喊话的字段基础。

## 系统的总体运行流程

现在系统的核心运行流程是：持续检测每一帧，但只对“新发生的独立违规事项”生成告警、保存证据并对外上报。

### 1. 创建安全监测任务

调用：

```
POST /api/detect/safety/start
```

系统根据入参确定：

- 输入类型：单张图片、图片序列、视频或直播流。
- 启用的安全算法。
- 模型权重、置信度、检测间隔和运行设备。
- 证据图片输出目录`output_dir`。

接口创建异步任务并返回`job_id`。

### 2. 获取并检测画面

系统持续从视频或直播流读取画面，然后执行：

1. YOLO目标检测。
2. 目标跟踪，为人员、叉车分配`track_id`。
3. 根据启用的算法进行安全分析。

目前可以识别的主要违规包括：

- `NO_HELMET`：人员未佩戴安全帽。
- `DRIVER_NO_HELMET`：叉车司机未佩戴安全帽。
- `PERSON_NEAR_FORKLIFT`：叉车作业半径内有人。
- `NO_FENCE_NEAR_FORKLIFT`：叉车作业区域未设置围蔽。
- `FORKLIFT_OVERSPEED`：叉车超速。

### 3. 判断是否为新的违规事项

系统按照以下组合识别一个独立违规：

```
违规类型 + 人员/叉车轨迹ID + 违规发生批次
```

具体规则如下：

- A持续未戴安全帽：只产生一次告警。
- B进入画面且未戴安全帽：B的轨迹ID不同，产生新的告警。
- 同一人员同时发生不同违规：每种违规分别产生告警。
- 同一轨迹的违规消失后再次出现：本次任务内不重复告警和保存图片；重新启动任务后可再次告警。
- 叉车作业半径站人：按照叉车轨迹ID和人员轨迹ID组合去重。

例如：

```
A未戴安全帽 → 新违规，告警一次；后续漏检或再次检测到A仍不重复保存
A持续未戴安全帽 → 重复状态，不告警
B未戴安全帽 → 新轨迹，新告警
C进入叉车作业半径 → 新违规类型，新告警
```

去重逻辑位于[common.py (line 38)](D:/codeproject/safety_warehouse_detect/safety_tools/common.py:38)。

### 4. 保存违规证据图片

检测到新违规后，系统会：

- 在证据画面中仅绘制YOLO原始检测框，不绘制类别、置信度、状态或告警文字。
- 每个包含新增违规的告警帧保存一张JPG图片，同一帧的多项违规共用该图片。
- 图片保存在请求指定的`output_dir`。
- 文件名包含算法、违规类型、时间戳和帧号。
- 返回证据图片路径和违规发生时间。

如果同一帧同时出现3项新违规，只保存1张图片，但仍返回3项独立的违规明细：

```
{
  "hazard_count": 3,
  "hazard_image_paths": [
    "本帧证据图片"
  ],
  "hazards": [
    {},
    {},
    {}
  ]
}
```

分别返回。相关逻辑位于[safety_video_monitor.py (line 190)](D:/codeproject/safety_warehouse_detect/safety_tools/safety_video_monitor.py:190)。

### 5. 实时画面与违规告警分开处理

系统维护两套数据：

- `frame_status`：当前最新画面的检测状态，每个检测帧都会更新，供界面展示。
- `frame_result`：只包含尚未被客户端读取的新违规事件。

因此：

- 安全帧正常显示，但不保存图片。
- 已经上报过的持续违规正常显示，但不重复保存、上报。
- 只有新违规才生成`frame_result`和证据图片。
- `annotated_stream_url`持续提供实时标注视频流。

### 6. 客户端轮询告警

客户端通过下面的接口获取新违规：

```
GET /api/jobs/{job_id}?after_frame_id={last_frame_id}&after_annotated_frame_id={last_annotated_frame_id}
```

关键返回字段：

- `result_updated=true`：存在尚未读取的新违规。
- `frame_result`：本次返回的违规事件。
- `result_frame_id`：违规事件游标。
- `annotated_updated=true`：产生了新的证据图片。
- `latest_frame_url`：最近一张新增违规证据图片。
- `hazard_image_paths`：本次所有新增违规的证据图片。
- `annotated_stream_url`：实时标注画面。

服务器会保留产生的违规事件，客户端按照`result_frame_id`逐个读取，避免因为轮询速度较慢而只拿到最后一次告警。相关逻辑位于[safety_web_server.py (line 658)](D:/codeproject/safety_warehouse_detect/safety_tools/safety_web_server.py:658)。

### 7. 对接喊话器时的流程

当前系统尚未直接控制喊话器，但已经提供喊话所需字段。外部喊话器控制程序应当：

```
轮询任务接口
  ↓
result_updated=true
  ↓
遍历frame_result.hazard_event.hazards
  ↓
使用incident_id防止重复播放
  ↓
根据code选择预警录音
  ↓
触发喊话器
  ↓
播放成功后更新last_frame_id
```

所以现阶段系统已经完成了“检测、跟踪、违规去重、事件返回、证据保存和实时展示”，还需要增加喊话器通信和播放确认模块，才能形成完整的现场闭环。



### 直播流部署前检查

1. 在运行后端的目标电脑上确认直播地址可访问，而不只是确认浏览器所在电脑可以访问。
2. 启动 Web 服务后，先在 WebUI 选择“直播流”，填写地址并点击“测试直播”。原画面能够连续显示后，再点击“开始检测”。
3. 如果其他电脑需要通过局域网访问 WebUI，应将服务监听地址设置为 `0.0.0.0`，并放行 Web 服务端口，默认端口为 `7860`。浏览器访问 `http://后端电脑IP:7860`。
4. 测试预览和算法检测复用同一个 Python WebRTC 接收器，启动算法不会再额外建立同源直播连接；直播地址只需要能够被后端电脑访问。
5. 当前直播输入固定按照 WebRTC/WHEP 处理，优先协商 H.264，并允许 VP8 作为备用；不会把 WebRTC 信令地址交给 OpenCV/FFmpeg 当作普通媒体 URL 打开。
6. 如果在无桌面的 Linux 服务器上安装 OpenCV 后出现 `libGL.so.1` 等图形库错误，可以只保留一个 OpenCV 包，并根据服务器环境将完整版替换为相同版本的 `opencv-contrib-python-headless`；不要让完整版和 headless 版同时存在。

因此，在其他电脑部署直播流功能时，应通过 `requirements.txt` 安装 `aiortc`、PyAV 和 `certifi` 等依赖，并检查模型文件、后端到流媒体服务器的 HTTP 信令网络以及 ICE/UDP 端口连通性。

本地开发或演示时，如果没有真实摄像头或流媒体服务器，可以先运行 `python safety_tools/mock_live_camera.py --video demo.mp4` 启动本地 WHEP 模拟直播源，再把 `http://127.0.0.1:8889/whep` 填入 WebUI 的直播地址。详见 [safety_tools/README.md](safety_tools/README.md#4-本地视频模拟直播流)。

### 直播监测可靠性与运行参数

直播接收采用分级恢复：短时丢包先使用 NACK/RTX，连续丢包请求关键帧，传输失败或长时间无帧时自动重建 WHEP 会话。预览与算法共享连接，重连期间保留最后一帧；连接代次变化后算法会重置跟踪、速度和运动估计状态，避免使用断流前的时序数据。

直播首帧接入完成后才加载算法模型。视频轨协商、ICE/DTLS 建连与关键帧解码分别显示状态并使用独立超时预算；首次 PLI 可从 SDP 获取主视频 SSRC。任务状态的 `stream_diagnostics` 提供 `connection_state`、`ice_state`、`rtp_packets_received`、`rtp_packets_lost`、`decode_errors`、`decode_empty_units` 和 `keyframe_requests`（诊断采样后可用）。如果 `raw_frame_id=null`，无 RTP 时检查公告 IP、媒体 UDP 端口及推流；收到 RTP 但无画面时检查 IDR/SPS/PPS 和推流端 PLI 响应。

服务提供只读健康检查接口 `GET /api/health`，返回任务数量、活动工作线程以及共享直播接收器的连接、恢复和重连状态，不返回直播地址。

以下环境变量用于控制长期运行时的资源边界：

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `SAFETY_MAX_ALGORITHM_JOBS` | `1` | 同时执行的算法任务数；其余任务排队，防止模型/GPU内存被重复任务耗尽。 |
| `SAFETY_MAX_RETAINED_EVENTS` | `256` | 每个任务在内存中保留的待查询告警事件上限。 |
| `SAFETY_MAX_RESULT_EVENTS` | `5000` | 长时直播最终结果中保留的最近告警事件上限；总数仍由 `hazard_event_count` 记录，完整事件写入 JSONL 日志。 |
| `SAFETY_MAX_RETAINED_JOBS` | `100` | 内存中保留的已结束任务数量上限。 |
| `SAFETY_JOB_RETENTION_SECONDS` | `21600` | 已结束任务在内存中的最长保留时间，默认6小时。 |
| `SAFETY_CA_BUNDLE` | 空 | HTTPS WHEP 使用的自定义 CA 证书包路径；未设置时使用 `certifi`。 |

停止服务时应使用 `Ctrl+C`，后端会先通知运行任务退出、停止共享直播接收器、关闭编码器并回收工作线程。JSON结果和视频结果都采用“临时文件完成后原子替换”的方式发布，异常中断不会把半写入文件作为正式结果返回。

每个安全监测任务还会在最终 JSON 同目录生成 `*.events.jsonl` 告警事件日志。每次新增告警都会立即追加并刷新到磁盘，因此即使直播任务尚未结束或进程意外退出，已经确认的告警元数据仍可恢复；每行都是一个完整、独立的 JSON 对象。

## 当前界面算法输出字段完整说明

本节按照当前 WebUI 的两个页面组织输出字段：

1. “统一安全监测”页面：人员安全帽、叉车司机安全帽、叉车作业半径站人、叉车作业围蔽、叉车速度、吊臂下方人员检测和吊装作业围栏及安全监护检测可以单选、多选或全选。
2. “物资计数系统”页面：支持单张图片、图片序列、视频和直播流输入。

当前界面默认值：统一安全监测采样间隔为 `5` 帧、置信度为 `0.55`；物资计数采样间隔 `5` 帧、置信度 `0.5`、设备 `0`、跟踪高阈值 `0.3`、低阈值 `0.1`、新轨迹阈值 `0.4`、缓存 `50` 帧、匹配阈值 `0.25`，启用 ReID。页面、接口和命令行的默认采样间隔均为 `5`，仍可显式指定其他间隔。默认模型均位于项目根目录：安全监测与物资计数均使用 `all_best.pt`。

两个页面默认使用同一份 17 类 OBB 权重 `all_best.pt`，统一显式传入 `imgsz=640`。安全监测每个采样帧只检测、跟踪一次，七项功能共用结果；吊臂使用旋转框，其他安全规则使用外接矩形。类别按模型 `names` 解析。图像等比缩放并补边，实际张量不一定为 `640×640`；固定尺寸导出模型须使用匹配配置。

统一安全监测只返回用户已勾选功能的详细结果。未勾选功能不会出现在 `frame_result.results` 和最终 `result.json.results` 中。模型在首次推理前设置 `end2end=False`，使用 Ultralytics 内置 OBB NMS；后端统一配置 `iou=0.7`、`agnostic_nms=False`。自定义 NMS 和页面 NMS 参数已移除，`fence_iou_thres` 业务参数继续保留。每帧检测日志采集于内置 NMS 之后、跟踪之前。WebUI 对异步任务每 1 秒查询一次最新结果，查询接口为：

```http
GET /api/jobs/{job_id}?after_frame_id={last_result_frame_id}&after_annotated_frame_id={last_annotated_frame_id}&after_log_id={last_log_id}
```

接口兼容规则：

- `GET /api/jobs/{job_id}`：保持原任务状态查询语义，不启用增量事件消费。任务完成后，`result`字段为最终汇总结果；任务运行中可通过`frame_status`查看当前最新帧状态。
- 携带`after_frame_id`或`after_annotated_frame_id`：启用增量告警查询，`frame_result`表示游标之后尚未消费的下一条告警事件，`result_updated`和`annotated_updated`分别表示告警事件和证据图片是否有更新。
- `after_log_id`：独立的 YOLO 检测日志游标，首次为 `0`，后续使用已读取的最大 `log_id`。省略时返回缓冲区中的全部帧日志。日志采集于置信度及类别过滤、内置 NMS 之后，跟踪器与业务规则处理之前；每个实际推理帧记录类别 ID/名称、置信度、像素坐标和数量，无目标时也记录。每个任务保留最近 200 条帧日志，界面保留约 20 万字符，未读取的旧日志被覆盖时会提示数量。

说明：

- 后端算法按照页面设置的 `sample_interval` 持续抽帧检测，WebUI 每 1 秒读取一次当前最新检测结果。该规则适用于图片序列、视频和直播流；单张图片只产生一帧结果。
- `frame_status` 是仅用于界面展示的当前检测帧状态，每个检测帧都会更新，但不会触发本地图片保存。
- `frame_result` 按“违规类型 + 人员/叉车轨迹 ID”做任务级去重。同一轨迹在本次任务中只生成一次事件和一张证据图，即使中间漏检、短暂安全或离开后重新出现也不重复；新的轨迹 ID 会生成新事件。
- 每个出现新增违规的告警帧都在 `output_dir` 中保存一张带微秒时间戳和帧号的 JPG；同一帧的多个新违规实例共用该图片，并通过 `hazard_image_paths` 返回唯一图片路径。
- `latest_frame_url` 指向最近一个新增违规实例的图片。调用方把 `annotated_frame_id` 作为下一次的 `after_annotated_frame_id`，通过 `annotated_updated` 判断是否产生了新图片。
- 安全帧及持续存在的已上报违规只更新内存中的 `frame_status` 和实时 MJPEG 画面，不重复保存图片。物资计数任务仍按原逻辑保存结果图片或视频。
- 安全结果画面只绘制 NMS 和跟踪后的 YOLO 检测框，不绘制类别、置信度、轨迹ID、速度、状态、告警文字、统计文字或作业区等规则辅助图形。
- 安全帽规则：只检测到人员且没有匹配安全帽时，原始状态为 `no_helmet`；未关联人员的 `head_no_helmet` 也计为一条未佩戴记录。已关联的人员和头部只计一次。单独的头部记录以头部框和头部轨迹 ID 表示，`detection_source=head_no_helmet`；人员记录为 `detection_source=person`。视频、直播流和图片序列中，同一轨迹须连续 5 个实际检测帧均为未佩戴，才将 `status` 设为 `no_helmet`、判定 `unsafe` 并产生告警及证据图；前 4 帧为 `unknown`，不计入 `no_helmet_count`。戴帽、状态未知或人员在某个检测帧漏检会中断连续计数；不同轨迹独立计数。无有效轨迹的未戴帽记录保持 `unknown`，原始判断保留在 `raw_status`。该规则同时适用于普通人员和叉车司机；抽帧间隔不会代替检测次数，例如间隔 15 帧采样时，第 1、16、31、46、61 帧均未戴帽才在第 61 帧确认。单张图片仍按当前图像直接判定。只有安全帽或没有检测目标时仍为 `unknown`。
- 坐标框统一采用 `[x1, y1, x2, y2]`，坐标单位为输入画面的像素。
- `array<number>` 表示数值数组，`array<object>` 表示对象数组；类型后标注“或 `null`”表示该字段可能没有有效结果。
- JSON 中的 `integer` 对应整数，`number` 对应整数或小数，`boolean` 对应 `true` 或 `false`，`object<string, integer>` 表示以动态类别名称为键、整数为值的对象。

### 异步任务公共返回字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | `boolean` | 状态查询是否成功。任务存在并成功返回时为 `true`。 |
| `job_id` | `string` | 当前检测任务的唯一标识。 |
| `status` | `string` | 任务状态：`running` 表示运行中，`done` 表示正常完成，`canceled` 表示用户停止，`error` 表示运行失败。 |
| `algorithm` | `string` | 算法名称。当前统一安全页面为 `safety_combined`，物资页面为 `material_count`；兼容接口还可能返回 `helmet`、`forklift_work` 或 `forklift_speed`。 |
| `selected_checks` | `array<string>`，可选 | 统一安全监测任务实际勾选的功能代码；物资计数和旧单项接口通常不返回。 |
| `progress` | `integer` | 任务进度百分比。直播没有固定总帧数，运行过程中通常为 `0`。 |
| `progress_text` | `string` | 当前运行状态说明，例如当前检测帧、累计数量或停止状态。 |
| `command` | `string` | 任务对应的命令行说明。直播地址会被替换为“直播地址已隐藏”，避免泄露带用户名或密码的地址。 |
| `raw_frame_id` | `integer`，可选 | 当前最新原始输入帧编号。该编号可能大于 `frame_id`，因为并非所有源帧都会执行检测。 |
| `raw_frame_status` | `object`，可选 | 原始输入帧的状态信息。字段见下表。 |
| `frame_id` | `integer` | 当前最新完成算法检测的源帧编号。 |
| `result_frame_id` | `integer`，可选 | 最近一次可上报隐患事件的源帧编号；没有隐患事件时不返回。调用方应把它作为下一次查询的 `after_frame_id`。 |
| `annotated_frame_id` | `integer`，可选 | 最近一次写入违规标注图片的源帧编号；尚未检测到违规时不返回。 |
| `annotated_updated` | `boolean` | `annotated_frame_id` 是否大于查询参数 `after_annotated_frame_id`。安全帧不会令其变为 `true`。 |
| `annotated_timestamp` | `string`，可选 | 最近一次违规标注图片写入时间，使用带时区的 ISO 8601 格式。 |
| `annotated_image_path` | `string`，可选 | 最新违规标注图片在服务器上的本地绝对路径，位于请求指定的 `output_dir`。 |
| `latest_frame_url` | `string`，可选 | 最近一个新增违规实例的标注图片访问地址；尚未产生违规事件时不返回。 |
| `annotated_stream_url` | `string` | 实时标注 MJPEG 画面地址，通常为 `/api/jobs/{job_id}/stream/annotated`；安全帧也会持续更新该画面。 |
| `result_timestamp` | `string` 或 `null` | 当前 `frame_result` 生成时间，使用带时区的 ISO 8601 格式。尚无检测结果时为 `null` 或不返回。 |
| `result_updated` | `boolean` | 是否存在比 `after_frame_id` 更新的可上报隐患事件。 |
| `frame_result` | `object` 或 `null` | 本次新增的隐患事件帧结果；无新增事件时为 `null`。 |
| `frame_status` | `object` 或 `null` | 当前最新检测帧状态，供 WebUI 持续展示；安全帧和已抑制的重复隐患帧也会更新。 |
| `detection_logs` | `array<object>` | `after_log_id` 之后尚未读取的 YOLO 帧日志；每项含递增的 `log_id`（整数）、`frame_id`（整数）、`timestamp`（带时区的时间字符串）、`message`（日志文本）。任务完成后仍可读取保留的日志。 |
| `detection_logs_dropped` | `integer` | 因缓冲区容量限制被覆盖、且晚于调用方游标的旧帧日志数量；正常跟上读取时为 `0`。 |
| `result` | `object` 或 `null` | 任务停止或完成后的最终汇总响应；任务运行中通常为 `null` 或不返回。 |
| `output_media_url` | `string`，可选 | 任务结束后生成的结果图片或视频地址。任务运行中可能不返回。 |
| `cancel_requested` | `boolean` | 用户点击停止后出现，表示后端已收到停止请求；正常运行且未请求停止时通常不返回。 |
| `error` | `string` | `status=error` 时的错误原因；没有错误时可能不返回。 |

隐患事件公共字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `has_hazard` | `boolean` | 当前帧是否存在安全隐患。 |
| `is_first_hazard` | `boolean` | 是否为当前视频或直播任务首次上报的隐患。 |
| `hazard_timestamp` | `string` | 隐患发生时间，带时区和微秒精度。 |
| `hazard_image_path` | `string` | 本地隐患证据图绝对路径。文件名包含时间戳和帧号。 |
| `hazard_image_paths` | `array<string>` | 当前事件的本地证据图路径；同一告警帧只包含一个唯一图片路径，多项违规共用该图片。 |
| `hazard_image_url` | `string`，Web API 可选 | 证据图下载地址。 |
| `incident_key` | `string` | 由违规代码、目标类型和轨迹 ID 组成的去重键。 |
| `incident_episode` | `integer` | 当前任务中该去重键的上报批次；任务级去重后通常为 `1`，新任务重新计数。 |
| `incident_id` | `string` | 当前违规实例批次的唯一业务标识。 |
| `hazard_count` | `integer` | 当前事件中的隐患对象数量。 |
| `hazards` | `array<object>` | 隐患明细；每项包含 `code`、`check`、`message`、`count`，可选 `details`。 |
| `hazard_event` | `object` | 上述事件字段的统一嵌套对象，供业务系统直接消费。 |

`raw_frame_status` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `fps` | `number` | 从直播源读取到的帧率；直播源没有提供有效帧率时默认使用 `25.0`。 |
| `generation` | `integer` | 直播连接代次，初次连接为 `1`，断线重连成功后递增。 |

公共返回示例：

```json
{
  "ok": true,
  "job_id": "a1b2c3",
  "status": "running",
  "algorithm": "safety_combined",
  "selected_checks": ["person_helmet", "driver_helmet"],
  "frame_id": 125,
  "result_timestamp": "2026-07-14T14:30:12+08:00",
  "result_updated": true,
  "frame_result": {}
}
```

### 统一安全监测页面的输出层级

当前页面提交接口为：

```http
POST /api/detect/safety/start
Content-Type: multipart/form-data
```

#### `/api/detect/safety/start` 入参

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `checks` | `string`，可重复 | 是 | 无 | 启用的安全监测功能。每个功能单独提交一个同名字段，可选值见下表；至少提交一个。 |
| `source_type` | `string` | 否 | `single_image` | 输入类型：`single_image`、`image_sequence`、`video` 或 `live_stream`。 |
| `source_file` | `file`，可重复 | 条件必填 | 无 | 上传输入文件。单图和视频提交一个文件；图片序列提交多个同名文件。若使用服务器本地 `source_path`，可以不上传。直播流不使用该字段。 |
| `source_path` | `string` | 条件必填 | 空 | 服务器本地输入路径，是 `source_file` 的替代方式。单图/视频填写文件路径，图片序列填写文件夹路径。路径必须能被后端进程访问。 |
| `stream_url` | `string` | 条件必填 | 空 | `source_type=live_stream` 时必填；当前固定接收 WebRTC/WHEP 的 HTTP(S) 信令地址，也支持 `whep+https://` 或 `whep+http://` 显式前缀。RTSP/RTMP 地址不会进入此链路。 |
| `weights_file` | `file` | 否 | 无 | 上传安全检测模型权重；提交后优先于 `weights`。 |
| `weights` | `string` | 否 | 服务端默认权重 | 服务器本地模型权重路径，是 `weights_file` 的替代方式。 |
| `output_dir` | `string` | 否 | `outputs/safety_web` | 服务器本地输出目录。隐患证据图和最终 JSON 保存在该目录。 |
| `expected_hazards` | `integer` | 否 | `0` | 人工统计的隐患数量，用于计算监测准确度。 |
| `conf` | `number` | 否 | `0.4` | YOLO 检测置信度阈值。 |
| `device` | `string` | 否 | 自动选择 | 推理设备，例如 `auto`、`cpu`、`0` 或 `cuda:0`；GPU 不可用时后端会回退到 CPU。 |
| `sample_interval` | `integer` | 否 | `5` | 视频、直播流或图片序列的检测采样间隔，最小为 `1`。 |
| `nearby_mode` | `string` | 否 | `circle` | 叉车附近站人判定方式：`original`、`circle` 或 `both`。仅叉车作业相关检查使用。 |
| `camera_mode` | `string` | 否 | `oblique` | 叉车算法统一拍摄模式。`oblique` 为斜拍：作业半径沿用地面点和透视椭圆，速度按叉车长×高自标定；`overhead` 为俯拍：作业半径以检测框中心绘制正圆，速度按叉车长×宽自标定。 |
| `forklift_radius_scale` | `number` | 否 | `1.2` | 叉车圆形作业半径相对叉车尺寸的缩放系数。 |
| `min_forklift_radius_px` | `number` | 否 | `180` | 圆形作业区最小半径，单位像素。 |
| `max_forklift_radius_px` | `number` | 否 | `450` | 圆形作业区最大半径，单位像素。 |
| `fence_iou_thres` | `number` | 否 | `0.01` | 围栏与叉车作业区判定为相邻时使用的最小 IoU。 |
| `draw_circle_work_zone` | `boolean` | 否 | `true` | 兼容参数；安全结果画面现在仅绘制原始检测框，因此该参数不再改变输出画面。 |
| `speed_limit` | `number` | 否 | `10.0` | `5 km/h` 对应的像素速度阈值，单位 `px/frame`。仅叉车速度检测使用。 |
| `movement_threshold` | `number` | 否 | `1.0` | 判断叉车是否运动的像素速度阈值，单位 `px/frame`。 |
| `speed_window` | `integer` | 否 | `5` | 叉车速度平滑窗口长度。 |
| `min_bg_matches` | `integer` | 否 | `20` | 相机运动补偿所需的最少背景特征匹配数。 |
| `motion_model` | `string` | 否 | `affine` | 相机运动补偿模型，当前算法支持 `affine` 或 `homography`。 |
| `boom_conf` | `number` | 否 | `0.40` | `boom` OBB 的置信度阈值。 |
| `crane_person_conf` | `number` | 否 | `0.40` | 吊臂模型中 `person` 的置信度阈值。 |
| `crane_line_margin` | `number` | 否 | `5` | 人框四角到吊臂下侧长边的最小有符号距离，单位像素。 |
| `crane_bottom_margin` | `number` | 否 | `5` | 人员脚底低于吊臂 OBB 最低点所需的额外距离，单位像素。 |
| `crane_range_extend_ratio` | `number` | 否 | `0.05` | 吊臂主轴两端允许扩展的长度比例。 |
| `crane_enter_window` | `integer` | 否 | `5` | 多帧进入危险状态的统计窗口，以实际处理帧计。 |
| `crane_enter_hits` | `integer` | 否 | `3` | 统计窗口内进入危险状态所需的危险帧数。 |
| `crane_exit_safe_frames` | `integer` | 否 | `5` | 解除危险状态所需的连续安全处理帧数。 |
| `crane_track_ttl_frames` | `integer` | 否 | `30` | boom-person 时序状态的最大保留处理帧数。 |

`checks` 可选值：

| 值 | 功能 |
|---|---|
| `person_helmet` | 人员安全帽检测 |
| `driver_helmet` | 叉车司机安全帽检测 |
| `forklift_proximity` | 叉车作业半径站人检测 |
| `forklift_fence` | 叉车作业围蔽检测 |
| `forklift_speed` | 叉车速度检测 |
| `crane_under_boom` | 吊臂下方人员检测 |
| `lifting_operation` | 吊装作业围栏及安全监护检测 |

条件与校验规则：

- `checks` 至少提交一个；需要同时启用多个功能时，重复提交 `checks` 字段，不能使用逗号拼接成一个值。
- `source_type=live_stream` 时必须提交 `stream_url`。
- 其他输入类型必须在 `source_file` 和 `source_path` 中至少提供一种。
- `source_type=single_image` 时不能选择 `forklift_speed`；速度检测需要视频、直播流或连续图片序列。
- `image_sequence` 使用接口上传时，应以多个 `source_file` part 提交图片，并保留各文件的相对目录文件名；也可以直接通过 `source_path` 指向服务器上的图片文件夹。

单图请求示例：

```powershell
curl.exe -X POST "http://127.0.0.1:7860/api/detect/safety/start" `
  -F "checks=person_helmet" `
  -F "checks=forklift_proximity" `
  -F "source_type=single_image" `
  -F "source_file=@D:\data\warehouse.jpg" `
  -F "weights=D:\models\all_best.pt" `
  -F "output_dir=D:\outputs\safety" `
  -F "conf=0.3"
```

直播流请求示例：

```powershell
curl.exe -X POST "http://127.0.0.1:7860/api/detect/safety/start" `
  -F "checks=person_helmet" `
  -F "checks=forklift_speed" `
  -F "source_type=live_stream" `
  -F "stream_url=http://192.168.1.20:8090/index/api/whep?app=live&stream=camera1" `
  -F "weights=D:\models\all_best.pt" `
  -F "sample_interval=5" `
  -F "speed_limit=10.0"
```

接口会异步创建任务。提交成功后先返回：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | `boolean` | 是否成功创建任务。 |
| `job_id` | `string` | 异步任务唯一标识，后续用于查询状态、预览画面和停止任务。 |
| `command` | `string` | 后端生成的任务命令说明；直播地址会被隐藏。 |
| `selected_checks` | `array<string>` | 本次勾选的功能代码，按照页面固定顺序排列。 |

`selected_checks[]` 与页面选项的对应关系：

| 页面选项 | 字段值 | 详细结果位置 |
|---|---|---|
| 人员安全帽检测 | `person_helmet` | `frame_result.results.person_helmet` |
| 叉车司机安全帽检测 | `driver_helmet` | `frame_result.results.driver_helmet` |
| 叉车作业半径站人检测 | `forklift_proximity` | `frame_result.results.forklift_proximity` |
| 叉车作业围蔽检测 | `forklift_fence` | `frame_result.results.forklift_fence` |
| 叉车速度检测 | `forklift_speed` | `frame_result.results.forklift_speed` |
| 吊臂下方人员检测 | `crane_under_boom` | `frame_result.results.crane_under_boom` |
| 吊装作业围栏及安全监护检测 | `lifting_operation` | `frame_result.results.lifting_operation` |

例如只勾选“人员安全帽检测”和“叉车司机安全帽检测”时，结果结构为：

```json
{
  "selected_checks": ["person_helmet", "driver_helmet"],
  "results": {
    "person_helmet": {},
    "driver_helmet": {}
  },
  "result": "safe"
}
```

此时不会返回 `forklift_proximity`、`forklift_fence`、`forklift_speed`、`crane_under_boom` 和 `lifting_operation`，结果图也不会绘制这些功能的内容。

#### 统一安全监测 `frame_result` 外层字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_id` | `integer` | 当前完成检测的源帧编号。 |
| `image` | `string` | 当前帧名称；图片输入通常为文件名，视频/直播通常为 `frame_N` 或 `stream_frame_N`。 |
| `source_type` | `string` | 输入类型：`single_image`、`image_sequence`、`video` 或 `live_stream`。 |
| `sample_interval` | `integer` | 采样间隔，表示相邻检测帧至少跨过的源帧数。 |
| `source_frame_delta` | `integer` | 当前检测帧与上一个实际检测帧之间跨过的源帧数。速度计算使用该实际值。 |
| `selected_checks` | `array<string>` | 当前实际启用的功能代码。 |
| `results` | `object` | 按功能代码保存的详细结果对象，只包含已勾选功能。 |
| `result` | `string` | 当前帧综合结果：任一已选功能为 `unsafe` 时为 `unsafe`；全部为 `safe` 时为 `safe`；其他情况为 `unknown`。 |
| `total_persons` | `integer`，可选 | 勾选人员安全帽检测时返回的人员数，等于 `results.person_helmet.total_persons`；同时启用叉车作业相关功能时会排除已判定的司机。 |
| `helmet_worn_count` | `integer`，可选 | 勾选人员安全帽检测时返回的戴帽人数。 |
| `no_helmet_count` | `integer`，可选 | 勾选人员安全帽检测时返回的未戴帽人数。 |
| `driver_count` | `integer`，可选 | 勾选司机安全帽检测时返回的司机人数。 |
| `driver_no_helmet_count` | `integer`，可选 | 勾选司机安全帽检测时返回的未戴帽司机人数。 |
| `forklift_count` | `integer` | 当前帧相关已选功能中检测到的最大叉车数量；未检测到叉车时为 `0`。 |
| `current_speed_px_per_frame` | `number` 或 `null`，可选 | 勾选叉车速度时，当前最快叉车的瞬时像素速度。 |
| `current_speed_kmh` | `number` 或 `null`，可选 | 当前最快叉车瞬时速度的 km/h 换算值。 |
| `smoothed_speed_px_per_frame` | `number` 或 `null`，可选 | 当前最快叉车的平滑像素速度。 |
| `smoothed_speed_kmh` | `number` 或 `null`，可选 | 当前最快叉车平滑速度的 km/h 换算值。 |
| `speed_limit_kmh` | `number`，可选 | 速度展示使用的参考限速，当前为 `5.0 km/h`。 |
| `overspeed_frame_count` | `integer`，可选 | 从任务开始到当前帧的超速采样帧数。 |
| `speed_status` | `string`，可选 | 当前速度状态：`unknown`、`ok` 或 `overspeed`。 |
| `crane_boom_count` | `integer`，可选 | 吊臂 OBB 模型检测到的吊臂数量。 |
| `crane_person_count` | `integer`，可选 | 吊臂 OBB 模型检测到的人员数量。 |
| `crane_danger_person_count` | `integer`，可选 | 当前帧经时序确认位于吊臂下方的人员数量。 |

#### 最终 `result` 响应外层字段

任务结束后，`GET /api/jobs/{job_id}` 的 `result` 是页面展示和下载使用的最终响应：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | `boolean` | 最终响应是否成功生成。 |
| `finished_at` | `string` | 算法结束时间，带本地时区的 ISO 8601 时间。 |
| `command` | `string` | 本次任务的命令说明。 |
| `summary` | `string` | 页面“结果摘要”显示文本，包含综合状态、处理帧数、各已选功能状态、隐患数和准确度。 |
| `statistics` | `string` | 页面统计文本，当前与 `summary` 内容相同。 |
| `accuracy` | `number` | 按人工隐患数与检测隐患数计算的百分比准确度，范围 `0～100`。 |
| `detected_hazards` | `integer` | 所有已勾选功能的隐患数之和。 |
| `hazards_by_check` | `object<string, integer>` | 各已选功能的隐患数；键为功能代码，未勾选功能不返回。 |
| `expected_hazards` | `integer` | 页面填写的人工隐患数量。 |
| `json` | `object` | 最终算法 JSON。详细字段见下表。 |
| `json_text` | `string` | `json` 的格式化字符串，用于页面直接展示。 |
| `output_media_url` | `string` | 标注结果图片或视频的访问地址。 |
| `output_json_url` | `string` | 最终 JSON 文件的访问地址。 |
| `logs` | `string` | 任务日志；无日志时为空字符串。 |

`hazards_by_check` 中各功能隐患数的计算含义：

| 键 | 类型 | 隐患数含义 |
|---|---|---|
| `person_helmet` | `integer` | 未戴安全帽的普通人员数量，即 `results.person_helmet.no_helmet_count`。 |
| `driver_helmet` | `integer` | 未戴安全帽的司机数量，即 `results.driver_helmet.no_helmet_count`。 |
| `forklift_proximity` | `integer` | 进入叉车作业区的人员记录数，即 `results.forklift_proximity.nearby_person_count`。 |
| `forklift_fence` | `integer` | 当前帧未检测到围栏时，判定为未围蔽的叉车数量。 |
| `forklift_speed` | `integer` | 代表帧存在超速时为 `1`，否则为 `0`。 |
| `crane_under_boom` | `integer` | 经几何规则和多帧防抖确认位于吊臂下方的人员数量。 |
| `lifting_operation` | `integer` | 各起重机缺少安全监护人员、缺少围栏的项数之和；同一台起重机两项都缺少时计为 2。 |

`accuracy` 的计算公式为 `100 × (1 - |检测隐患数 - 人工隐患数| / max(检测隐患数, 人工隐患数, 1))`，结果最低为 `0`。

最终 `result.json` 以任务期间最严重的违规帧作为详细结果；若没有违规帧，则使用最后一个有效检测帧，并增加以下任务汇总字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `source` | `string` | 输入文件、目录、视频或直播源标识。 |
| `source_type` | `string` | 输入类型。 |
| `sample_interval` | `integer` | 本次任务的采样间隔。 |
| `processed_frames` | `integer` | 实际完成算法检测的帧数。 |
| `unsafe_frame_count` | `integer` | 综合结果为 `unsafe` 的采样帧数量。 |
| `stopped_by_user` | `boolean` | 是否由用户点击停止而结束。 |
| `result` | `string` | 整个任务的综合结果：`safe`、`unsafe` 或 `unknown`。只要出现过违规帧，最终为 `unsafe`。 |
| `finished_at` | `string` | 算法处理结束时间，带本地时区的 ISO 8601 时间。 |
| `selected_checks` | `array<string>` | 本次实际启用的功能代码。 |
| `results` | `object` | 最终代表帧中各已选功能的详细结果。 |

### 选项一：人员安全帽检测 `results.person_helmet`

| 字段 | 类型 | 含义 |
|---|---|---|
| `result` | `string` | 当前帧结果：存在未戴帽人员为 `unsafe`；全部戴帽为 `safe`；无人或存在未知状态且无未戴帽人员时为 `unknown`。 |
| `total_persons` | `integer` | 当前帧纳入人员安全帽检查的人数。同时启用司机安全帽、作业半径站人或作业围蔽功能时，会排除已判定的司机；只勾选人员安全帽时统计当前帧全部人员。 |
| `helmet_worn_count` | `integer` | 判定为佩戴安全帽的人数。 |
| `no_helmet_count` | `integer` | 判定为未佩戴安全帽的人数。 |
| `unknown_count` | `integer` | 安全帽状态无法确定的人数。 |
| `persons` | `array<object>` | 纳入本功能统计的人员详情列表。 |

`persons[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `person_id` | `integer` | 人员标识；有轨迹时使用 `track_id`，否则使用当前帧顺序编号。 |
| `track_id` | `integer` 或 `null` | BoT-SORT 人员轨迹 ID；未建立轨迹时为 `null`。 |
| `box` | `array<integer>` | 人员框 `[x1, y1, x2, y2]`。 |
| `status` | `string` | 安全帽确认状态：`helmet`、`no_helmet` 或 `unknown`。连续输入中，同一轨迹连续 5 个检测帧未戴帽才为 `no_helmet`，确认前为 `unknown`；单张图片直接判定。 |
| `raw_status` | `string`，可选 | 当前帧未经连续 5 帧确认的原始安全帽状态；可用于区分待确认的未戴帽与其他未知状态。 |
| `confidence` | `number` | 用于安全帽状态判定的置信度，通常为 `0～1`。 |
| `associated_helmet_box` | `array<integer>` 或 `null` | 与该人员头部区域匹配的安全帽框。 |
| `associated_no_helmet_box` | `array<integer>` 或 `null` | 与该人员头部区域匹配的未戴帽头部框。 |
| `detection_source` | `string` | `person` 表示由人员框判定；`head_no_helmet` 表示无关联人员的未戴帽头部，此时 `box` 和 `track_id` 使用头部检测结果，并计入 `total_persons`、`no_helmet_count`。 |

### 选项二：叉车司机安全帽检测 `results.driver_helmet`

司机判定顺序为：人员框完全包含在叉车框内时优先判定为司机；不满足完全包含时，再使用人员与叉车的重叠率和相对位置条件判定。

| 字段 | 类型 | 含义 |
|---|---|---|
| `result` | `string` | 司机安全帽结果：存在未戴帽司机为 `unsafe`；全部戴帽为 `safe`；无司机或存在未知状态时为 `unknown`。 |
| `driver_count` | `integer` | 当前帧判定为叉车司机或车上人员的数量。 |
| `helmet_worn_count` | `integer` | 戴安全帽司机数量。 |
| `no_helmet_count` | `integer` | 未戴安全帽司机数量。 |
| `unknown_count` | `integer` | 安全帽状态未知的司机数量。 |
| `drivers` | `array<object>` | 司机或车上人员详情。 |

`drivers[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `person_id` | `integer` | 人员标识。 |
| `track_id` | `integer` 或 `null` | 人员轨迹 ID。 |
| `box` | `array<integer>` | 司机人员框。 |
| `ground_point` | `array<number>` | 人员框底边中心地面点 `[x, y]`。 |
| `dx_norm` | `number` | 人员与叉车中心的归一化水平偏移。 |
| `dy_up_norm` | `number` | 人员相对叉车向上的归一化垂直偏移。 |
| `person_forklift_iou` | `number` | 人员框与叉车框的 IoU。 |
| `person_inside_forklift_ratio` | `number` | 人员框落在叉车框内的面积占人员框面积的比例。 |
| `person_box_fully_inside_forklift` | `boolean` | 人员框是否完全被叉车框包围。该字段为 `true` 时直接优先认定为司机。 |
| `reason` | `string` | 司机认定原因：`person_box_fully_inside_forklift` 或 `driver_or_on_forklift`。 |
| `status` | `string` | 司机安全帽确认状态：`helmet`、`no_helmet` 或 `unknown`。连续输入中，同一轨迹连续 5 个检测帧未戴帽才为 `no_helmet`，确认前为 `unknown`；单张图片直接判定。 |

### 选项三：叉车作业半径站人检测 `results.forklift_proximity`

| 字段 | 类型 | 含义 |
|---|---|---|
| `result` | `string` | 检测到叉车且有人进入作业区为 `unsafe`；有叉车且无人进入为 `safe`；未检测到叉车为 `unknown`。 |
| `forklift_count` | `integer` | 当前帧叉车数量。 |
| `has_person_nearby` | `boolean` | 是否存在非司机人员进入任一叉车作业区。 |
| `nearby_person_count` | `integer` | 各叉车附近人员记录数量之和。同一人员同时进入多辆叉车作业区时可能出现多条记录。 |
| `forklifts` | `array<object>` | 每辆叉车的站人检测结果。 |

`forklifts[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `forklift_id` | `integer` | 叉车标识；有轨迹时使用轨迹 ID。 |
| `forklift_box` | `array<integer>` | 叉车检测框。 |
| `has_person_nearby` | `boolean` | 当前叉车作业区是否有非司机人员。 |
| `nearby_persons` | `array<object>` | 当前叉车附近人员列表，完整字段见后文“附近人员 `nearby_persons[]` 字段”。 |
| `work_zone` | `array<integer>` | 当前叉车矩形作业区。 |
| `circle_work_zone` | `object` | 当前叉车圆形或椭圆形作业区，完整字段见后文“`circle_work_zone` 字段”。 |

### 选项四：叉车作业围蔽检测 `results.forklift_fence`

视频、直播流和连续图片序列会先根据叉车轨迹判断运动状态。只有确认静止的叉车才进行围蔽检查；运动中的叉车以及速度暂时未知的叉车会跳过围蔽检查，不产生缺少围蔽告警。默认运动阈值为 `1.0 px/frame`，可通过 `movement_threshold` 调整。单张图片无法计算运动状态，因此仍直接进行围蔽检查。

| 字段 | 类型 | 含义 |
|---|---|---|
| `result` | `string` | 存在可检查的静止叉车且当前帧检测到至少一个围栏为 `safe`；静止叉车存在但未检测到围栏为 `unsafe`；没有可检查的静止叉车为 `unknown`。 |
| `forklift_count` | `integer` | 当前帧叉车数量。 |
| `fence_count` | `integer` | 当前帧检测到的围栏或围蔽总数。 |
| `all_forklifts_fenced` | `boolean` | 是否检测到叉车且当前帧至少检测到一个围栏。 |
| `checked_forklift_count` | `integer` | 当前帧实际执行围蔽检查的静止叉车数量。 |
| `moving_forklift_count` | `integer` | 因处于运动状态而跳过围蔽检查的叉车数量。 |
| `movement_unknown_forklift_count` | `integer` | 因轨迹速度暂时未知而跳过围蔽检查的叉车数量。 |
| `movement_threshold_px_per_frame` | `number` | 叉车运动状态判定阈值。 |
| `forklifts` | `array<object>` | 每辆叉车的围蔽检测结果。 |

`forklifts[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `forklift_id` | `integer` | 叉车标识。 |
| `forklift_box` | `array<integer>` | 叉车检测框。 |
| `has_fence_nearby` | `boolean` | 兼容字段；表示当前帧是否检测到至少一个围栏，不再要求围栏位于叉车作业区附近。 |
| `nearby_fences` | `array<object>` | 附近围蔽列表，子字段为 `fence_id`（`integer`，围蔽编号）、`box`（`array<integer>`，围蔽框）和 `iou_with_work_zone`（`number`，围蔽框与作业区 IoU）。 |
| `current_speed_px_per_frame` | `number` 或 `null` | 用于决定是否执行围蔽检查的当前叉车速度。 |
| `is_moving` | `boolean` 或 `null` | 是否正在运动；速度未知时为 `null`。 |
| `fence_check_applicable` | `boolean` | 当前叉车是否实际执行围蔽检查。 |
| `fence_check_skip_reason` | `string` 或 `null` | 跳过原因：`forklift_moving` 或 `movement_unknown`。 |

### 选项五：叉车速度检测 `results.forklift_speed`

| 字段 | 类型 | 含义 |
|---|---|---|
| `result` | `string` | 当前帧速度结果：`safe`、`unsafe` 或 `unknown`。 |
| `speed_status` | `string` | 当前速度状态：`unknown`、`ok` 或 `overspeed`。 |
| `forklift_count` | `integer` | 当前帧成功跟踪的叉车数量。 |
| `current_speed_px_per_frame` | `number` 或 `null` | 当前最快叉车归一化到每个源帧的瞬时像素速度。 |
| `smoothed_speed_px_per_frame` | `number` 或 `null` | 当前最快叉车在速度窗口内的平均像素速度。 |
| `current_speed_kmh` | `number` 或 `null` | 当前最快叉车的瞬时 km/h 换算值。 |
| `smoothed_speed_kmh` | `number` 或 `null` | 当前最快叉车的平滑 km/h 换算值。 |
| `speed_limit_px_per_frame` | `number` | 超速判断使用的像素速度阈值。 |
| `speed_limit_kmh` | `number` | 比例换算使用的参考限速，当前为 `5.0 km/h`。 |
| `max_speed_px_per_frame` | `number` 或 `null` | 从任务开始到当前帧出现过的最大像素速度。 |
| `max_speed_kmh` | `number` 或 `null` | 最大像素速度对应的 km/h 换算值。 |
| `overspeed_frame_count` | `integer` | 从任务开始到当前帧出现超速的采样帧数量。 |
| `motion_compensation_success` | `boolean` | 当前帧是否成功补偿相机背景运动。 |
| `source_frame_delta` | `integer` | 当前检测帧与上一个检测帧之间实际跨过的源帧数。 |
| `forklifts` | `array<object>` | 当前帧每辆叉车的速度详情，完整字段见后文“速度结果 `forklifts[]` 字段”。 |

### 选项六：吊臂下方人员检测 `results.crane_under_boom`

| 字段 | 类型 | 含义 |
|---|---|---|
| `result` | `string` | `unsafe` 表示至少一个 boom-person 组合已通过时序确认；没有有效吊臂几何时为 `unknown`，其余为 `safe`。 |
| `boom_count` | `integer` | 当前帧置信度过滤后的吊臂数量。 |
| `valid_boom_count` | `integer` | 能稳定建立下侧长边和主轴的吊臂数量；接近竖直等不稳定几何不计入。 |
| `person_count` | `integer` | 当前帧吊臂模型检测到的人员数量。 |
| `raw_danger_pair_count` | `integer` | 当前单帧同时满足下侧半平面、主轴范围和脚底最低点条件的组合数量。 |
| `danger_pair_count` | `integer` | 当前帧已通过多帧防抖确认的危险组合数量。 |
| `active_danger_pair_count` | `integer` | 状态机中仍处于危险状态的组合数量，短时漏检期间可能大于当前帧 `danger_pair_count`。 |
| `danger_person_count` | `integer` | 当前帧危险组合中去重后的危险人员数量。 |
| `temporal_applied` | `boolean` | 视频、直播和图片序列为 `true`；单张图片直接使用单帧结果，为 `false`。 |
| `booms` | `array<object>` | 吊臂 OBB、下侧长边、主轴、最低点及几何有效性。 |
| `persons` | `array<object>` | 人员水平外接框、置信度和轨迹 ID。 |
| `pairs` | `array<object>` | 每个 boom-person 组合的 `fully_below`、`in_boom_range`、`foot_below_lowest`、`raw_danger` 和 `confirmed_danger` 等判定明细。 |

### 选项七：吊装作业围栏及安全监护检测 `results.lifting_operation`

使用共享检测结果和统一 `conf`，兼容 `safety officer` / `safety_officer` 类别名。逐台检查 `crane`：画面中安全员和围栏的框中心 Y 大于该起重机框中心 Y，即视为在其下方；两者都有为合规，缺少任一项为不合规。该简单规则不限制水平距离、不检查围栏闭合、不估计吊装状态，多个起重机可以复用同一帧下方的安全员或围栏。单图、图片序列、视频和直播均逐帧立即判定。

| 字段 | 类型 | 含义 |
|---|---|---|
| `result` | `string` | 没有起重机时 `unknown`；所有起重机合规为 `safe`，任一不合规为 `unsafe`。 |
| `crane_count` | `integer` | 起重机数量。 |
| `safety_officer_count` / `fence_count` | `integer` | 当前帧置信度过滤后安全员 / 围栏数量，包含未满足下方条件的目标。 |
| `compliant_crane_count` / `unsafe_crane_count` | `integer` | 合规 / 不合规起重机数量。 |
| `missing_safety_officer_count` / `missing_fence_count` | `integer` | 下方缺少安全员 / 围栏的起重机数量。 |
| `cranes` | `array<object>` | 每台起重机的判定明细，字段见下表。 |

`cranes[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `crane_id` / `track_id` | `integer|string` / `integer|null` | 起重机目标 ID / 跟踪 ID；无轨迹时目标 ID 使用帧内 `crane_序号`。 |
| `box` / `confidence` / `center_y` | `array<number>` / `number` / `number` | 起重机 XYXY 外接框、置信度和中心 Y。 |
| `has_safety_officer` / `has_fence` | `boolean` | 下方是否有安全员 / 围栏。 |
| `safety_officers` / `fences` | `array<object>` | 满足下方条件的关联检测，包含 `box`、`confidence`、`class_id`、`class_name`、`track_id`，OBB 结果另含 `points`。 |
| `result` | `string` | 当前起重机 `safe` 或 `unsafe`。 |
| `violations` | `array<string>` | 缺少安全员为 `NO_SAFETY_OFFICER_BELOW_CRANE`，缺少围栏为 `NO_FENCE_BELOW_CRANE`，合规为空数组。 |

缺项沿用隐患事件、截图和按目标轨迹去重流程。接口在原有 `POST /api/detect/safety/start` 中增加 `checks=lifting_operation` 即可启用；也可以使用命令行：

```bash
python safety_tools/safety_video_monitor.py --checks lifting_operation --source-type single_image --source test.jpg --weights all_best.pt --conf 0.55 --output outputs/lifting.jpg --json outputs/lifting.json
```

### 物资计数页面输出组织

物资计数不放在统一安全监测的 `results` 中，它由“物资计数系统”页面单独提交。运行中的实时结果位于 `frame_result`，任务结束后的计数汇总位于 `result.json`。其字段在后文“物资计数 `frame_result`”和“物资计数最终 `result` 字段”中逐项列出。

### 旧单项接口兼容输出

以下“安全帽检测 `frame_result`”“叉车作业安全检测 `frame_result`”和独立“叉车速度检测 `frame_result`”表格用于说明仍然保留的旧单项接口。当前统一安全监测页面应优先按照上面的 `frame_result.results.<功能代码>` 读取结果。

### 安全帽检测 `frame_result`

| 字段 | 类型 | 含义 |
|---|---|---|
| `image` | `string` | 当前检测帧名称，例如 `stream_frame_125`。该字段沿用原单图算法的 `image` 字段。 |
| `frame_id` | `integer` | 当前完成检测的源帧编号。 |
| `source_type` | `string` | 输入源类型，直播模式固定为 `live_stream`。 |
| `sample_interval` | `integer` | 采样间隔。`1` 表示每个可处理源帧都检测，`N` 表示至少跨过 N 个源帧再检测。 |
| `total_persons` | `integer` | 当前帧检测到的人员总数。 |
| `person_count` | `integer` | 当前帧人员数量，与 `total_persons` 相同，作为兼容字段保留。 |
| `helmet_worn_count` | `integer` | 当前帧判断为佩戴安全帽的人员数量。 |
| `no_helmet_count` | `integer` | 当前帧判断为未佩戴安全帽的人员数量。 |
| `unknown_count` | `integer` | 当前帧安全帽状态无法确定的人员数量。 |
| `persons` | `array<object>` | 当前帧人员检测、跟踪和安全帽判断结果。子字段见下表。 |
| `result` | `string` | 当前帧安全状态：`safe`、`unsafe` 或 `unknown`。存在未戴安全帽人员时为 `unsafe`。 |

`persons[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `person_id` | `integer` | 人员标识。有有效轨迹时使用 `track_id`，否则使用当前帧人员顺序编号。 |
| `track_id` | `integer` 或 `null` | BoT-SORT 人员轨迹 ID；未成功建立轨迹时为 `null`。 |
| `box` | `array<integer>` | 人员检测框 `[x1, y1, x2, y2]`。 |
| `status` | `string` | 安全帽确认状态：`helmet`、`no_helmet` 或 `unknown`。连续输入中，同一轨迹连续 5 个检测帧未戴帽才为 `no_helmet`，确认前为 `unknown`；单张图片直接判定。 |
| `raw_status` | `string` | 未经过连续 5 帧确认的当前帧原始安全帽状态，包括无有效轨迹的检测记录。 |
| `confidence` | `number` | 用于判定该人员安全帽状态的置信度，范围通常为 `0～1`。 |
| `associated_helmet_box` | `array<integer>` 或 `null` | 与人员头部区域关联的安全帽框。没有匹配安全帽时为 `null`。 |
| `associated_no_helmet_box` | `array<integer>` 或 `null` | 与人员头部区域关联的未戴帽头部框。没有匹配结果时为 `null`。 |
| `detection_source` | `string` | `person` 或 `head_no_helmet`；后者表示只有未戴帽头部，此时框和轨迹 ID 来自头部检测，仍计为一条未佩戴记录。 |

### 叉车作业安全检测 `frame_result`

| 字段 | 类型 | 含义 |
|---|---|---|
| `image` | `string` | 当前检测帧名称，例如 `stream_frame_125`。 |
| `frame_id` | `integer` | 当前完成检测的源帧编号。 |
| `source_type` | `string` | 输入源类型，直播模式固定为 `live_stream`。 |
| `sample_interval` | `integer` | 当前视频文件或直播流的检测采样间隔。 |
| `total_persons` | `integer` | 当前帧人员总数。 |
| `person_count` | `integer` | 当前帧人员数量，与 `total_persons` 相同。 |
| `helmet_worn_count` | `integer` | 当前帧佩戴安全帽人数。 |
| `no_helmet_count` | `integer` | 当前帧未佩戴安全帽人数。 |
| `unknown_count` | `integer` | 当前帧安全帽状态未知人数。 |
| `persons` | `array<object>` | 当前帧全部人员及其安全帽状态。字段与安全帽检测的 `persons[]` 相同，并增加 `helmet_status`。 |
| `has_forklift` | `boolean` | 当前帧是否检测到叉车。 |
| `forklift_count` | `integer` | 当前帧叉车数量。 |
| `fence_count` | `integer` | 当前帧检测到的围栏或围蔽数量。 |
| `has_fence` | `boolean` | 当前帧是否检测到至少一个围栏。 |
| `has_fence_nearby` | `boolean` | 当前帧是否同时检测到叉车和至少一个围栏。 |
| `has_person_nearby` | `boolean` | 是否有非驾驶员人员进入任一叉车作业区。 |
| `excluded_person_count` | `integer` | 被识别为驾驶员或车上人员、因而不计入周边站人违规的人数。 |
| `forklift_is_safe` | `boolean` | 叉车作业区的人员和围栏规则是否安全。没有检测到叉车时为 `false`。 |
| `helmet_is_safe` | `boolean` | 当前帧全部人员是否不存在未戴安全帽情况。 |
| `is_safe` | `boolean` | 当前帧综合安全状态。只有叉车规则和安全帽规则都安全时为 `true`。 |
| `result` | `string` | 当前帧综合结果：`safe` 或 `unsafe`。 |
| `violations` | `array<string>` | 当前帧去重后的违规代码。可能包括 `PERSON_NEAR_FORKLIFT`、`NO_FENCE_NEAR_FORKLIFT`。 |
| `nearby_persons` | `array<object>` | 所有叉车附近的非驾驶员人员汇总列表。字段见“附近人员字段”。 |
| `excluded_persons` | `array<object>` | 驾驶员或车上人员汇总列表。字段见“排除人员字段”。 |
| `nearby_mode` | `string` | 周边站人判断模式：`circle`、`original` 或 `both`。 |
| `compliance_summary` | `object` | 安全帽、司机安全帽、围栏和作业半径规则的汇总。子字段见下表。 |
| `ground_point_work_area` | `object` | 根据叉车和围栏地面点生成的全局作业区域。子字段见下表。 |
| `road_surface_mask` | `object` | 道路区域自动提取状态。子字段见下表。 |
| `road_constrained_work_area` | `object` | 地面点作业区与道路区域相交后的限制区域。子字段见下表。 |
| `forklifts` | `array<object>` | 每辆叉车的作业区、附近人员、围栏和违规详情。子字段见下表。 |

叉车作业 `persons[]` 在安全帽人员字段基础上增加：

| 字段 | 类型 | 含义 |
|---|---|---|
| `helmet_status` | `string` | 人员安全帽状态，值为 `helmet`、`no_helmet` 或 `unknown`，与该人员的 `status` 对应。 |

`compliance_summary` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `helmet_result` | `string` | 全部人员安全帽结果：`safe`、`unsafe` 或 `unknown`。 |
| `driver_helmet_result` | `string` | 驾驶员或车上人员安全帽结果：`safe`、`unsafe` 或 `unknown`。 |
| `fence_result` | `string` | 围栏规则结果：`safe` 或 `unsafe`。 |
| `work_radius_result` | `string` | 作业半径站人规则结果：`safe` 或 `unsafe`。 |
| `overall_result` | `string` | 上述规则综合结果：`safe` 或 `unsafe`。 |
| `helmet_no_helmet_count` | `integer` | 当前帧全部人员中未戴安全帽数量。 |
| `helmet_unknown_count` | `integer` | 当前帧全部人员中安全帽状态未知数量。 |
| `driver_no_helmet_count` | `integer` | 驾驶员或车上人员中未戴安全帽数量。 |
| `driver_unknown_count` | `integer` | 驾驶员或车上人员中安全帽状态未知数量。 |

`forklifts[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `forklift_id` | `integer` | 叉车标识。有有效跟踪结果时使用 BoT-SORT 轨迹 ID。 |
| `forklift_box` | `array<integer>` | 叉车检测框。 |
| `forklift_ground_point` | `array<number>` | 叉车检测框底部中心地面点 `[x, y]`。 |
| `circle_work_zone` | `object` | 叉车圆形或椭圆形作业区域，字段见下表。 |
| `ground_depth_px` | `number` | 根据叉车尺寸计算的地面作业深度，单位为像素。 |
| `near_threshold_px` | `number` | 地面点距离规则使用的人员附近阈值，单位为像素。 |
| `work_zone` | `array<integer>` | 叉车矩形作业区 `[x1, y1, x2, y2]`。 |
| `fence_based_work_zone` | `array<array<integer>>` 或 `null` | 根据附近围栏生成的多边形作业区；无法生成时为 `null`。 |
| `road_aligned_work_zone` | `object` | 沿道路方向生成并经过道路/围栏裁剪的作业区。字段见下表。 |
| `detection_rules_used` | `array<string>` | 当前叉车附近人员判断实际启用的规则名称。 |
| `excluded_persons` | `array<object>` | 当前叉车驾驶员或车上人员列表。 |
| `has_person_nearby` | `boolean` | 当前叉车作业区是否有非驾驶员人员。 |
| `nearby_persons` | `array<object>` | 当前叉车附近人员列表。 |
| `has_fence_nearby` | `boolean` | 兼容字段；表示当前帧是否检测到至少一个围栏。 |
| `nearby_fences` | `array<object>` | 兼容字段；保存当前帧检测到的全部围栏。 |
| `violations` | `array<string>` | 当前叉车违规代码列表。 |
| `is_safe` | `boolean` | 当前叉车作业是否安全。 |
| `message` | `string` | 当前叉车安全判定的英文说明。 |

`circle_work_zone` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `enabled` | `boolean` | 是否启用圆形或椭圆形作业区。 |
| `center` | `array<number>` | 作业区中心点。 |
| `radius_px` | `number` | 作业区最大半径。 |
| `radius_x_px` | `number` | 椭圆长轴方向半径。 |
| `radius_y_px` | `number` | 椭圆短轴方向半径。 |
| `radius_scale` | `number` | 基于矩形作业区计算椭圆半径的缩放系数。 |
| `min_radius_px` | `number` | 配置的最小作业半径。 |
| `max_radius_px` | `number` | 配置的最大作业半径。 |
| `shape` | `string` | 作业区形状：`circle` 或 `ellipse`。 |
| `rotation_angle_deg` | `number` | 椭圆相对图像横轴的旋转角度，单位为度。 |
| `road_direction_confidence` | `number` | 道路方向估计置信度。 |
| `scale_source` | `string` | 半径尺寸来源，当前为 `work_zone`。 |
| `polygon` | `array<array<integer>>` | 用于绘制和点包含判断的作业区多边形顶点。 |

`road_aligned_work_zone` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `enabled` | `boolean` | 是否启用道路方向作业区。 |
| `success` | `boolean` | 是否成功生成有效作业区。 |
| `source` | `string` 或 `null` | 道路方向或基础区域的来源说明。 |
| `road_direction_angle` | `number` 或 `null` | 道路方向角度，单位为度。 |
| `direction_confidence` | `number` | 道路方向估计置信度。 |
| `base_rect_work_zone` | `array<integer>` 或 `null` | 生成道路方向作业区使用的基础矩形。 |
| `polygon` | `array<array<integer>>` 或 `null` | 作业区主多边形。 |
| `rotated_base_polygon` | `array<array<integer>>` 或 `null` | 按道路方向旋转后的基础多边形。 |
| `polygons` | `array<array<array<integer>>>` | 作业区可能包含的多个外轮廓。 |
| `fence_clip_polygon` | `array<array<integer>>` 或 `null` | 围栏裁剪使用的多边形。 |
| `fence_barrier_polygons` | `array<array<array<integer>>>` | 围栏障碍区域多边形列表。 |
| `fence_barrier_boxes` | `array<array<integer>>` | 用于围栏裁剪的围栏框列表。 |
| `used_fence_clip` | `boolean` | 是否实际使用围栏裁剪。 |
| `used_road_mask_clip` | `boolean` | 是否实际使用道路掩膜裁剪。 |
| `used_road_mask_intersection` | `boolean` | 是否与道路掩膜执行过相交运算。 |
| `fence_clipped_work_zone` | `object` 或 `null` | 围栏裁剪后作业区的生成状态和几何信息，字段见下表。 |
| `base_area` | `integer` | 裁剪前基础区域面积，单位为像素平方。 |
| `aligned_area` | `integer` | 最终道路方向作业区面积，单位为像素平方。 |
| `aligned_area_ratio` | `number` | 最终区域面积相对于基础区域面积的比例。 |

`fence_clipped_work_zone` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `enabled` | `boolean` | 是否启用围栏障碍裁剪。 |
| `success` | `boolean` | 是否使用围栏障碍成功得到有效裁剪区域。 |
| `used_road_mask` | `boolean` | 裁剪过程中是否同时使用道路掩膜。 |
| `used_fence_barrier` | `boolean` | 是否实际使用围栏障碍区域进行裁剪。 |
| `kept_component_contains_forklift` | `boolean` | 最终保留的连通区域是否包含当前叉车地面点。 |
| `area_ratio` | `number` | 围栏裁剪后区域面积相对于裁剪前道路方向区域面积的比例。 |

附近人员 `nearby_persons[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `person_id` | `integer` | 人员标识。 |
| `track_id` | `integer` 或 `null` | BoT-SORT 人员轨迹 ID。 |
| `box` | `array<integer>` | 人员检测框。 |
| `ground_point` | `array<number>` | 人员框底部中心地面点。 |
| `foot_points` | `array<array<number>>` | 人员脚部区域的多个候选点。 |
| `closest_foot_point` | `array<number>` 或 `null` | 与叉车地面点距离最近的脚部点。 |
| `distance_to_forklift_px` | `number` | 人员与叉车的地面像素距离。 |
| `near_threshold_px` | `number` | 地面点距离规则使用的阈值。 |
| `dx_norm` | `number` | 人员相对叉车的归一化水平偏移。 |
| `dy_up_norm` | `number` | 人员相对叉车向上方向的归一化垂直偏移。 |
| `nearby_mode` | `string` | 当前使用的附近判断模式。 |
| `matched_by_original_rules` | `boolean` | 是否命中原矩形、距离或区域规则。 |
| `matched_by_circle_rule` | `boolean` | 是否命中圆形或椭圆形作业区规则。 |
| `in_rect_work_zone` | `boolean` | 人员是否位于当前叉车的基础矩形作业区。 |
| `near_by_distance` | `boolean` | 人员与叉车的地面点距离是否小于 `near_threshold_px`。 |
| `in_fence_based_work_zone` | `boolean` | 人员是否位于由叉车和附近围栏共同生成的作业区。 |
| `in_circle_work_zone` | `boolean` | 人员是否位于圆形或椭圆形作业区。 |
| `in_ground_point_work_area` | `boolean` | 人员是否位于全局地面点作业区。 |
| `in_road_constrained_work_area` | `boolean` | 人员是否位于道路限制作业区。 |
| `in_road_aligned_forklift_work_zone` | `boolean` | 人员是否位于当前叉车道路方向作业区。 |
| `in_fence_clipped_forklift_work_zone` | `boolean` | 人员是否位于围栏裁剪后的叉车作业区。 |
| `reason` | `string` 或 `array<string>` | 命中单条规则时为一个字符串，命中多条规则时为字符串数组。 |
| `reasons` | `array<string>` | 命中的全部附近人员判断规则。 |
| `helmet_status` | `string` | 该人员安全帽状态。 |

排除人员 `excluded_persons[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `person_id` | `integer` | 人员标识。 |
| `track_id` | `integer` 或 `null` | BoT-SORT 人员轨迹 ID。 |
| `box` | `array<integer>` | 人员检测框。 |
| `ground_point` | `array<number>` | 人员地面点。 |
| `dx_norm` | `number` | 人员相对叉车的归一化水平偏移。 |
| `dy_up_norm` | `number` | 人员相对叉车向上方向的归一化垂直偏移。 |
| `person_forklift_iou` | `number` | 人员框与叉车框的 IoU。 |
| `person_inside_forklift_ratio` | `number` | 人员框落入叉车框的面积比例。 |
| `reason` | `string` | 排除原因，当前为 `driver_or_on_forklift`。 |
| `helmet_status` | `string` | 驾驶员或车上人员安全帽状态。 |

附近围栏 `nearby_fences[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `fence_id` | `integer` | 当前帧围栏顺序编号。 |
| `box` | `array<integer>` | 围栏检测框。 |
| `iou_with_work_zone` | `number` | 围栏框与叉车矩形作业区的 IoU。 |

`ground_point_work_area` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `enabled` | `boolean` | 是否启用全局地面点作业区。 |
| `type` | `string` 或 `null` | 作业区生成类型；未生成时为 `null`。 |
| `padding_px` | `integer` | 作业区多边形向外扩展像素数。 |
| `fallback_zone_radius` | `integer` | 无法生成凸包时使用的回退半径。 |
| `polygon` | `array<array<integer>>` 或 `null` | 作业区多边形。 |
| `ground_points` | `object` | 参与生成区域的叉车和围栏地面点，包含 `forklifts` 与 `fences` 数组。 |

`ground_points.forklifts[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `forklift_id` | `integer` | 当前帧叉车顺序编号。 |
| `box` | `array<integer>` | 叉车检测框。 |
| `ground_point` | `array<number>` | 叉车框底部中心地面点。 |

`ground_points.fences[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `fence_id` | `integer` | 当前帧围栏顺序编号。 |
| `box` | `array<integer>` | 围栏检测框。 |
| `ground_point` | `array<number>` | 围栏框底部中心地面点。 |

`road_surface_mask` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `enabled` | `boolean` | 是否启用道路区域提取。 |
| `success` | `boolean` | 是否成功生成有效道路掩膜。 |
| `method` | `string` | 道路提取方法，当前通常为 `auto_hsv_connected_component`。 |
| `road_area_ratio` | `number` | 道路掩膜面积占整幅图像的比例。 |
| `used_connected_component` | `boolean` | 是否使用连通区域筛选。 |
| `contains_forklift_ground_point` | `boolean` | 选中的道路连通区域是否包含叉车地面点。 |
| `warning` | `string` 或 `null` | 道路提取失败、面积异常或回退选择的原因。 |

`road_constrained_work_area` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `enabled` | `boolean` | 是否启用道路限制作业区。 |
| `type` | `string` | 区域类型，当前为 `ground_point_work_area_intersect_road_mask`。 |
| `success` | `boolean` | 地面点作业区与道路掩膜是否成功得到有效交集。 |
| `person_count` | `integer` | 当前区域中的人员数量。 |
| `polygons` | `array<array<array<integer>>>` | 道路限制区域的多边形轮廓。 |

### 叉车速度检测 `frame_result`

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_id` | `integer` | 当前完成速度检测的源帧编号。 |
| `source_type` | `string` | 输入源类型，直播模式固定为 `live_stream`。 |
| `processed_frame_count` | `integer` | 当前任务已经完成速度检测的采样帧数量。 |
| `sample_interval` | `integer` | 页面设置的直播采样间隔。 |
| `source_frame_delta` | `integer` | 当前检测帧与上一个实际完成检测帧之间跨过的源帧数。速度换算使用该实际值，而不是固定使用 `sample_interval`。 |
| `current_speed_px_per_frame` | `number` 或 `null` | 当前最快叉车的像素速度，已除以 `source_frame_delta`；首次出现或无法关联轨迹时为 `null`。 |
| `smoothed_speed_px_per_frame` | `number` 或 `null` | 当前最快叉车在速度窗口内的平均像素速度。 |
| `current_speed_kmh` | `number` 或 `null` | 根据像素速度阈值比例换算的当前速度，单位为 km/h。 |
| `smoothed_speed_kmh` | `number` 或 `null` | 平滑速度换算值，单位为 km/h。 |
| `speed_limit_px_per_frame` | `number` | 超速判断使用的像素速度阈值。 |
| `speed_limit_kmh` | `number` | 用于比例换算和展示的参考限速，当前为 `5.0 km/h`。 |
| `overspeed` | `boolean` | 当前帧是否存在超速叉车。 |
| `speed_status` | `string` | 当前帧速度状态：`unknown`、`ok` 或 `overspeed`。 |
| `track_id` | `integer` 或 `null` | 当前帧最快叉车的 BoT-SORT 轨迹 ID。没有可计算速度的叉车时为 `null`。 |
| `forklift_count` | `integer` | 当前帧跟踪到的叉车数量。 |
| `max_speed_px_per_frame` | `number` 或 `null` | 从任务开始到当前帧出现过的最大像素速度。 |
| `max_speed_kmh` | `number` 或 `null` | 最大像素速度对应的 km/h 换算值。 |
| `overspeed_frame_count` | `integer` | 从任务开始到当前帧，检测到超速的采样帧数量。 |
| `total_frames` | `integer` | 输入总帧数。直播没有固定总帧数，因此为 `0`。 |
| `motion_compensation_success` | `boolean` | 当前帧是否成功使用背景特征估计并补偿相机运动。 |
| `forklifts` | `array<object>` | 当前帧每辆叉车的速度详情。子字段见下表。 |
| `result` | `string` | 当前帧速度安全结果：`safe`、`unsafe` 或 `unknown`。 |

速度结果 `forklifts[]` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `track_id` | `integer` | 叉车 BoT-SORT 轨迹 ID。 |
| `box` | `array<integer>` | 叉车检测框。 |
| `ground_point` | `array<number>` | 叉车框底部中心地面点。 |
| `current_speed_px_per_frame` | `number` 或 `null` | 当前叉车归一化到单个源帧的像素速度。 |
| `sampled_displacement_px` | `number` 或 `null` | 当前检测帧与上次检测帧之间的总像素位移。 |
| `source_frame_delta` | `integer` | 两次检测之间实际跨过的源帧数。 |
| `smoothed_speed_px_per_frame` | `number` 或 `null` | 当前叉车速度窗口内的平均像素速度。 |
| `current_speed_kmh` | `number` 或 `null` | 当前叉车速度的 km/h 换算值。 |
| `smoothed_speed_kmh` | `number` 或 `null` | 当前叉车平滑速度的 km/h 换算值。 |
| `speed_limit_px_per_frame` | `number` | 像素速度阈值。 |
| `speed_limit_kmh` | `number` | 参考限速。 |
| `speed_conversion` | `object` | 速度换算参数，包含 `reference_speed_kmh`、`reference_speed_px_per_frame` 和 `formula`。 |
| `overspeed` | `boolean` | 当前叉车是否超速。 |
| `speed_status` | `string` | 当前叉车速度状态：`unknown`、`ok` 或 `overspeed`。 |
| `motion_compensation_success` | `boolean` | 当前叉车速度是否使用了成功的背景运动补偿。 |
| `speed_source` | `string` | 速度来源：`motion_compensated_ground_pixel_speed`、`image_displacement_fallback` 或 `unavailable_no_previous_track`。 |

`speed_conversion` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `reference_speed_kmh` | `number` | 参考实际速度，当前为 `5.0 km/h`。 |
| `reference_speed_px_per_frame` | `number` | 与参考实际速度对应的像素速度，即页面配置的像素速度阈值。 |
| `formula` | `string` | 当前采用的速度换算公式文本。 |

### 物资计数 `frame_result`

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_id` | `integer` | 当前完成物资检测与跟踪的源帧编号。 |
| `name` | `string` | 输入任务名称，直播模式固定为 `live_stream`。 |
| `source_type` | `string` | 输入源类型，直播模式固定为 `live_stream`。 |
| `sample_interval` | `integer` | 当前直播检测采样间隔。 |
| `total_frames` | `integer` | 输入总帧数。直播没有固定总帧数，因此为 `0`。 |
| `total` | `integer` | 从任务开始到当前帧累计去重后的物资总数，沿用原物资计数返回字段。 |
| `total_count` | `integer` | 与 `total` 相同的实时计数兼容字段。 |
| `classes` | `object<string, integer>` | 按模型类别名称统计的累计去重数量，例如 `{"4D distribution box": 5}`。 |
| `stopped_by_user` | `boolean` | 当前任务是否已收到用户停止请求。运行过程中为 `false`。 |

物资计数使用“类别 ID + BoT-SORT 轨迹 ID + 直播连接代次”作为去重键。同一轨迹在连续帧中只计数一次；直播断线重连后会进入新的连接代次，避免新连接中的轨迹 ID 与旧连接冲突。

### 每 1 秒返回与结果摘要显示规则

安全监测 WebUI 分别保存上一次收到的隐患事件帧号和违规标注图片帧号：

```text
第一次查询：after_frame_id=-1，after_annotated_frame_id=-1
第 125 帧首次违规：result_frame_id=125，result_updated=true，annotated_frame_id=125，annotated_updated=true
下一次查询：after_frame_id=125，after_annotated_frame_id=125
```

如果后续处理的是安全帧，没有新的隐患事件或违规标注图，则：

```json
{
  "frame_id": 129,
  "result_frame_id": 125,
  "result_updated": false,
  "frame_result": null,
  "annotated_frame_id": 125,
  "annotated_updated": false,
  "latest_frame_url": "/api/file?path=...NO_HELMET_hazard_..._frame_125.jpg"
}
```

如果第 130 帧仍是同一轨迹的持续违规，不重复上报或保存图片；如果第 150 帧出现新的违规类型或新的目标轨迹，则：

```json
{
  "frame_id": 150,
  "result_frame_id": 150,
  "result_updated": true,
  "frame_result": {
    "frame_id": 150,
    "hazard_count": 1
  },
  "annotated_frame_id": 150,
  "annotated_updated": true,
  "latest_frame_url": "/api/file?path=...FORKLIFT_OVERSPEED_hazard_..._frame_150.jpg"
}
```

`frame_status` 和 `annotated_stream_url` 始终用于当前实时画面展示，不受隐患事件去重和违规标注图游标影响。

当任务停止后，“结果摘要”会切换为任务最终汇总结果和最终 JSON 文件内容。





## 1. 功能介绍

本工程面向仓库安全生产场景，基于 YOLO/Ultralytics 检测、BoT-SORT 跟踪、OpenCV 视频处理和若干业务规则，实现安全监测与物资抽样计数。

### 1.1 20人员安全帽检测

入口脚本：`safety_tools/helmet_check.py`

功能说明：

- 检测图像中的人员、安全帽、未戴帽头部。
- 对每个人员框提取上半身头部区域。
- 根据安全帽框中心点是否落入头部区域，判断是否佩戴安全帽。
- 根据 `head_no_helmet` 与头部区域的重叠关系，判断未佩戴安全帽。
- 输出每个人的检测框、状态、置信度和统计信息。

默认类别约定：

```text
0 person
1 helmet
2 head_no_helmet
3 forklift
4 fence
```

输出结果包括：

- 总人数
- 佩戴安全帽人数
- 未佩戴安全帽人数
- 状态未知人数
- 标注后的图片
- JSON 检测详情

### 1.2 21叉车司机佩戴安全帽检测

入口脚本：`safety_tools/forklift_work_check.py`

功能说明：

- 在叉车作业安全检测中同时识别 `person`、`helmet`、`head_no_helmet`、`forklift`、`fence`。
- 根据人员地面点与叉车地面点的相对位置，判断驾驶员或车上人员。
- 对驾驶员或车上人员进行安全帽状态关联。
- 驾驶员或车上人员会被加入 `excluded_persons`，不参与“叉车周边站人”违规统计。
- 仍会在 JSON 结果中保留其人员框、安全帽状态和排除原因。

说明：该功能不是独立接口，而是包含在“叉车作业安全检测”接口中。

### 1.3 22叉车作业半径站人检测

入口脚本：`safety_tools/forklift_work_check.py`

功能说明：

- 检测叉车和人员。
- 以叉车框底部中心点作为地面点。
- 根据叉车尺寸生成作业半径或作业区域。
- 支持 `circle`、`original`、`both` 三种附近人员判断模式。
- 判断人员底部中心点是否落入叉车作业半径或作业区。
- 驾驶员或车上人员会被排除，不触发“周边站人”违规。

主要违规项：

```text
PERSON_NEAR_FORKLIFT
```

### 1.4 23叉车作业是否有围蔽检测

入口脚本：`safety_tools/forklift_work_check.py`

功能说明：

- 检测图像中的围栏或围蔽目标 `fence`。
- 同一帧只要同时检测到叉车和至少一个围栏，就判定叉车已有围蔽。
- 检测到叉车但当前帧未检测到任何围栏时，输出违规。

主要违规项：

```text
NO_FENCE_NEAR_FORKLIFT
```

### 1.5 24叉车行驶速度检测

入口脚本：`safety_tools/forklift_speed_check.py`

功能说明：

- 输入视频，检测并跟踪叉车。
- 使用 Ultralytics `model.track` 和 BoT-SORT 进行目标跟踪。
- 以叉车检测框底部中心点作为地面点。
- 通过 ORB 特征点估计相邻帧背景运动，用于补偿相机或无人机运动。
- 计算叉车相对地面的像素速度，单位为 `pixel/frame`。
- 支持设置超速阈值。
- 支持 WebUI 实时预览、进度轮询和停止检测。

输出结果包括：

- 标注后的视频
- 每帧叉车速度
- 平滑速度
- 是否超速
- 超速帧数量
- 最大速度
- JSON 检测详情

### 1.6 物资抽样计数算法

入口脚本：`safety_tools/material_count_cli.py`

功能说明：

- 支持单张图片、图片序列、视频文件三种输入类型。
- 默认模型为 OBB 权重 `all_best.pt`，结果使用旋转四边形标注；也支持普通目标检测模型（task=`detect`），例如 `yolo26s-best.pt`，此时使用轴对齐矩形标注。其他 task（`segment`、`pose`、`classify`）会被拒绝。
- 可计数类别由所选模型自身的 `names` 决定：WebUI 选中或上传模型后自动列出类别，不再固定为九类；服务端按同一份类别表校验提交的类别。
- 使用默认模型时，界面提供 `large cable`、`4D distribution box`、`6D distribution box`、`200kV Transformer`、`400kV Transformer`、`400kV distribution box`、`500kV distribution box`、`630kV Transformer`、`630kV Transformer NX2` 九个候选类别。
- WebUI 可勾选本次需要计数的目标类别；未勾选类别不会进入跟踪、计数、标注画面或返回字段。
- 单张图片模式使用检测结果直接计数。
- 图片序列和视频模式使用 BoT-SORT 跟踪计数，避免同一目标在多帧中重复计数。
- 支持 ReID、跟踪阈值、缓存帧数、匹配阈值等参数配置。
- 支持 WebUI 实时计数预览、进度轮询和停止计数。
- 支持输入抽样物资实际数量，输出检测数量和准确度对比。

输出结果包括：

- 累计物资数量
- 各类别数量
- 抽样准确度
- 标注后的图片或视频
- JSON 计数详情

## 2. 项目文件工程架构

当前工程根目录示例：

```text
safety_warehouse_detect/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ mkdocs.yml
├─ all_best.pt
├─ all_best.pt
├─ yolo26n.pt
├─ yolo26n-cls.pt
├─ safety_tools/
├─ ultralytics/
├─ outputs/
└─ runs/
```

### 2.1 根目录文件

| 文件或目录 | 作用 |
|---|---|
| `README.md` | 当前项目说明文档。 |
| `requirements.txt` | 主依赖文件，包含 OpenCV、NumPy、ONNX Runtime、Ultralytics 等依赖。 |
| `pyproject.toml` | 本地 Ultralytics 包配置文件，定义包元数据、依赖和工具配置。 |
| `mkdocs.yml` | Ultralytics 文档站点配置，当前算法部署通常不需要修改。 |
| `all_best.pt` | **安全检测默认权重，通常用于人员、安全帽、叉车、围蔽检测**。 |
| `all_best.pt` | **17 类统一 OBB 模型，安全监测与物资计数均默认使用此权重**。 |
| `yolo26n.pt` | YOLO26 通用检测权重 |
| `yolo26n-cls.pt` | YOLO26 分类权重。 |
| `outputs/` | WebUI 和算法运行输出目录，保存上传文件、结果图片、结果视频和 JSON。 |
| `runs/` | 运行时中间文件目录，例如 Web 物资计数生成的 BoT-SORT 配置。 |
| `safety_tools/` | 业务算法、Web 服务和前端页面目录。 |
| `ultralytics/` | 本地 Ultralytics 源码，包含 YOLO、跟踪、训练、推理等基础能力。 |

### 2.2 `safety_tools/` 目录

```text
safety_tools/
├─ common.py
├─ helmet_check.py
├─ forklift_work_check.py
├─ forklift_speed_check.py
├─ material_count_cli.py
├─ safety_web_server.py
├─ requirements.txt
├─ README.md
├─ web/
│  └─ index.html
└─ .ultralytics_config/
```

| 文件或目录 | 作用 |
|---|---|
| `common.py` | 公共工具函数，包含模型加载、检测结果解析、框计算、IoU、绘制、JSON 保存等。 |
| `helmet_check.py` | 人员安全帽检测脚本，支持命令行调用和 Web 后端调用。 |
| `forklift_work_check.py` | 叉车作业安全检测脚本，包含司机安全帽、作业半径站人、围蔽检测等规则。 |
| `forklift_speed_check.py` | 叉车速度检测脚本，处理视频、跟踪叉车、补偿背景运动并计算像素速度。 |
| `material_count_cli.py` | 物资数量识别脚本，支持单图、图片序列、视频计数和实时预览回调。 |
| `safety_web_server.py` | Web 后端服务入口，提供静态页面、上传处理、算法调用、任务轮询、实时预览流和文件访问接口。 |
| `requirements.txt` | 安全工具的最小依赖说明。 |
| `README.md` | 安全工具子目录原始说明。 |
| `web/index.html` | WebUI 前端页面，使用 HTML、CSS、JavaScript 实现。 |
| `.ultralytics_config/` | Ultralytics 运行配置目录。 |

### 2.3 `safety_tools/web/` 目录

| 文件 | 作用 |
|---|---|
| `index.html` | 单页 WebUI，包含“统一安全监测”和统一的物资计数页面；物资页面可切换文件或直播流输入。 |

前端页面内置：

- 页面布局和样式 CSS。
- 文件选择、模型选择、输出目录选择逻辑。
- 表单提交逻辑。
- 异步任务轮询逻辑。
- 叉车速度和物资计数实时预览流显示逻辑。
- 输出图片、输出视频、JSON 链接展示逻辑。

### 2.4 `ultralytics/` 目录

该目录是本地 Ultralytics 源码，用于提供 YOLO 检测、跟踪和模型基础能力。

主要子目录：

| 目录 | 作用 |
|---|---|
| `ultralytics/cfg/` | 默认配置、数据集配置、模型结构配置和跟踪器配置。 |
| `ultralytics/cfg/trackers/` | BoT-SORT、ByteTrack 等跟踪器 YAML 配置。 |
| `ultralytics/data/` | 数据集加载、增强、转换、划分等工具。 |
| `ultralytics/engine/` | 训练、验证、预测、导出、结果对象等核心引擎。 |
| `ultralytics/models/` | YOLO、YOLOE、RT-DETR、SAM 等模型实现。 |
| `ultralytics/nn/` | 网络结构、模块、任务解析、后端封装。 |
| `ultralytics/trackers/` | 多目标跟踪实现，包括 BoT-SORT、ByteTrack、Kalman Filter、匹配逻辑。 |
| `ultralytics/utils/` | 日志、绘图、检查、下载、NMS、指标、Torch 工具等通用能力。 |
| `ultralytics/solutions/` | Ultralytics 提供的通用视觉应用示例和解决方案。 |

### 2.5 `outputs/` 和 `runs/`

| 目录 | 作用 |
|---|---|
| `outputs/safety_web/` | WebUI 默认输出目录，保存上传文件、检测图片、检测视频、JSON 和实时预览帧。 |
| `outputs/safety_web/uploads/` | WebUI 上传的图片、视频、模型或图片序列临时保存目录。 |
| `runs/material_count_web/` | 物资计数运行时生成的 BoT-SORT 配置文件目录。 |

## 3. 本地部署和运行流程

### 3.1 环境要求

推荐环境：

- 操作系统：Windows 10/11 或 Linux。
- Python：建议 Python 3.9 到 3.12。
- GPU：可选。若使用 NVIDIA GPU，需要安装匹配 CUDA 的 PyTorch。
- 浏览器：Chrome、Edge 或其他现代浏览器。
- 必要 Python 包：`torch`、`torchvision`、`opencv-python` 或 `opencv-contrib-python`、`numpy`、`ultralytics` 等。

说明：

- 本项目 WebUI 是 `HTML + CSS + JavaScript`，不需要 Node.js 作为运行环境。
- Python 后端负责模型推理和文件处理。
- 若出现 NumPy 与 PyTorch ABI 不兼容提示，请安装与当前 PyTorch 匹配的 NumPy 版本，常见解决方式是使用 `numpy<2` 或安装支持 NumPy 2.x 的 PyTorch 版本。

### 3.2 创建 Python 环境

使用 Conda：

```powershell
conda create -n safety_warehouse python=3.10 -y
conda activate safety_warehouse
```

或使用 venv：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3.3 安装 PyTorch

CPU 环境示例：

```powershell
pip install torch torchvision torchaudio
```

CUDA 环境请根据本机 CUDA 版本安装对应 PyTorch。示例：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果已经安装了 CUDA/GPU 版本 PyTorch，也可以直接使用 CPU 推理，不需要重新安装 CPU 版本 PyTorch。运行时将 WebUI 或命令行中的 `device` 设置为 `cpu` 即可。

### 3.4 安装项目依赖

在项目根目录运行：

```powershell
cd D:\codeproject\safety_warehouse_detect
pip install -r requirements.txt
```

如果只运行安全工具的最小功能，也可以参考：

```powershell
pip install -r safety_tools\requirements.txt
```

### 3.5 启动 Web 服务

在项目根目录运行：

```powershell
python safety_tools\safety_web_server.py
```

默认服务地址：

```text
http://127.0.0.1:7860
```

可通过环境变量修改监听地址和端口：

```powershell
$env:SAFETY_WEB_HOST="0.0.0.0"
$env:SAFETY_WEB_PORT="7860"
python safety_tools\safety_web_server.py
```

### 3.6 程序运行流程

WebUI 运行链路：

1. 用户浏览器访问 `http://127.0.0.1:7860`。
2. 后端 `safety_web_server.py` 返回 `safety_tools/web/index.html`。
3. 用户在页面中选择输入图片、视频、图片序列、模型和输出目录。
4. 前端使用 `FormData` 提交到对应 `/api/...` 接口。
5. 后端保存上传文件，确定模型路径和输出路径。
6. 后端调用对应算法脚本或后台任务函数。
7. 算法生成标注图片、标注视频和 JSON。
8. 对于视频速度检测和物资计数，后端同时通过 `/api/jobs/{job_id}/stream` 推送实时预览 JPEG 流。
9. 前端轮询 `/api/jobs/{job_id}` 更新进度和统计。
10. 检测完成后，前端显示结果摘要、统计、输出文件和 JSON 链接。

### 3.7 CPU/GPU 运行设置

所有当前算法都支持在没有 GPU 的电脑上使用 CPU 运行。设备参数统一使用 `device`：

| 取值 | 含义 |
|---|---|
| `auto` | 由 Web 后端自动选择设备：检测到 CUDA GPU 时使用 `0`，未检测到 GPU 时使用 `cpu`。 |
| `0` | 使用第 0 块 CUDA GPU。 |
| `1` | 使用第 1 块 CUDA GPU，依此类推。 |
| `cpu` | 强制使用 CPU 推理，适用于无 NVIDIA GPU、未安装 CUDA 或现场电脑只允许 CPU 运行的情况。 |

WebUI 设备选择规则：

1. 打开 WebUI 时，后端会自动检测当前电脑是否存在可用 CUDA GPU。
2. 如果检测到 GPU，`device` 默认填充为 `0`。
3. 如果未检测到 GPU，`device` 默认填充为 `cpu`。
4. 即使用户在无 GPU 电脑上误填 `0`、`cuda` 或 `cuda:0`，后端也会自动改为 `cpu` 调用算法，避免 CUDA 设备不可用导致任务失败。

命令行使用 CPU 的方法：

```powershell
python safety_tools\helmet_check.py --weights all_best.pt --image test.jpg --output outputs\helmet.jpg --json outputs\helmet.json --conf 0.3 --device cpu
python safety_tools\forklift_work_check.py --weights all_best.pt --image test.jpg --output outputs\work.jpg --json outputs\work.json --conf 0.3 --device cpu
python safety_tools\forklift_speed_check.py --weights all_best.pt --video forklift.mp4 --output outputs\speed.mp4 --json outputs\speed.json --conf 0.3 --device cpu
python safety_tools\material_count_cli.py --source-type single_image --source test.jpg --model all_best.pt --output-media outputs\count.jpg --json outputs\count.json --conf 0.3 --device cpu
```

注意：

- CPU 不需要 CUDA，但需要安装 PyTorch CPU 版本或可在 CPU 上运行的 PyTorch 环境。
- 已安装 CUDA/GPU 版本 PyTorch 的环境同样可以跑 CPU，只需设置 `device=cpu`，不需要重新安装 CPU 版 PyTorch。
- CPU 运行速度明显慢于 GPU，视频检测和图片序列计数耗时会更长。
- 命令行直接运行算法脚本时同样会解析设备；无 GPU 电脑即使误传 `--device 0`、`--device cuda` 或 `--device cuda:0`，脚本也会自动回退到 `cpu`。为了日志和排查更直观，现场部署时仍建议无 GPU 电脑显式传入 `--device cpu`。

## 4. 算法功能调用接口

### 4.1 公共接口

#### 获取默认配置

```text
GET /api/config
```

返回默认模型路径和输出目录：

```json
{
  "default_weights": "D:\\codeproject\\safety_warehouse_detect\\all_best.pt",
  "default_material_model": "D:\\codeproject\\safety_warehouse_detect\\all_best.pt",
  "material_class_candidates": [
    "4D distribution box",
    "6D distribution box",
    "200kV Transformer",
    "400kV Transformer",
    "400kV distribution box",
    "500kV distribution box",
    "630kV Transformer",
    "630kV Transformer NX2"
  ],
  "default_output_dir": "D:\\codeproject\\safety_warehouse_detect\\outputs\\safety_web",
  "default_device": "0",
  "has_gpu": true,
  "gpu_count": 1,
  "gpu_detection_error": ""
}
```

#### 查询模型类别

```text
GET /api/model-classes?path=<urlencoded model path>
POST /api/model-classes          # multipart/form-data，字段 model_file
```

返回模型的 task 和类别名称，供 WebUI 在选中或上传物资计数模型后动态生成类别勾选表：

```json
{
  "ok": true,
  "path": "D:\\codeproject\\safety_warehouse_detect\\yolo26s-best.pt",
  "task": "detect",
  "names": ["small cable", "large cable", "large distribution box"],
  "supported": true,
  "error": ""
}
```

`supported` 表示该 task 是否可用于物资计数，目前 `obb` 和 `detect` 为 `true`。POST 形式会把上传的模型保存到服务端并返回其真实路径，前端随后直接提交该路径，避免重复上传。结果按“路径 + 修改时间 + 文件大小”缓存。

#### 访问输出文件

```text
GET /api/file?path=<urlencoded absolute path>
```

用于访问输出图片、输出视频、JSON 文件和实时预览帧。视频文件支持浏览器 Range 请求，便于回放和拖动进度条。

#### 选择输出目录

```text
GET /api/select-path?kind=dir&initial=<path>
```

在服务端本机打开系统目录选择对话框，返回用户选择的真实目录路径。

#### 查询异步任务状态

```text
GET /api/jobs/{job_id}
```

用于查询叉车速度检测和物资计数任务状态。

常见状态：

```text
running
done
error
canceled
```

#### 实时预览流

```text
GET /api/jobs/{job_id}/stream
```

返回类型：

```text
multipart/x-mixed-replace; boundary=frame
```

前端通过 `<img src="/api/jobs/{job_id}/stream">` 显示实时检测或计数画面。

#### 停止异步任务

```text
POST /api/jobs/{job_id}/cancel
```

用于停止叉车速度检测或物资计数任务。

### 4.2 20人员安全帽检测接口

对应功能：

- 人员安全帽检测

接口：

```text
POST /api/detect/helmet
```

请求类型：

```text
multipart/form-data
```

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | file | 上传输入图片，可选。 |
| `image_path` | string | 服务端本地图片路径；未上传图片时使用。 |
| `weights_file` | file | 上传模型文件，可选，优先级高于 `weights`。 |
| `weights` | string | 服务端本地权重路径。 |
| `output_dir` | string | 输出目录。 |
| `expected_hazards` | int | 人工输入隐患数量，用于计算监测准确度。 |
| `conf` | float | 检测置信度，WebUI 默认 `0.4`。 |
| `device` | string | 推理设备；Web 后端支持 `auto`、`0`、`1`、`cuda`、`cuda:0`、`cpu`，无 GPU 时会自动回退到 `cpu`。 |

返回内容：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 接口是否调用成功。 |
| `finished_at` | string | 检测算法运行结束时间，ISO 8601 格式，例如 `2026-07-03T14:30:25+08:00`。 |
| `command` | string | 后端实际调用的命令行。 |
| `summary` | string | 面向界面显示的结果摘要，包含人员统计和准确度。 |
| `statistics` | string | 面向界面显示的统计文本。 |
| `accuracy` | float | 监测准确度百分比，计算方式为 `1 - abs(检测隐患数量 - 人工隐患数量) / max(检测隐患数量, 人工隐患数量, 1)`。 |
| `detected_hazards` | int | 算法检测出的隐患数量，安全帽检测中为未戴安全帽人数。 |
| `expected_hazards` | int | 用户输入的人工隐患数量。 |
| `json` | object | 算法原始 JSON 结果。 |
| `json_text` | string | 格式化后的 JSON 字符串，供页面展示。 |
| `output_media_url` | string | 标注结果图片访问 URL。 |
| `output_json_url` | string | JSON 文件访问 URL。 |
| `logs` | string | 算法标准输出和错误输出。 |

`json` 字段结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `image` | string | 输入图片文件名。 |
| `total_persons` | int | 检测到的人员总数。 |
| `helmet_worn_count` | int | 佩戴安全帽人数。 |
| `no_helmet_count` | int | 未佩戴安全帽人数。 |
| `unknown_count` | int | 安全帽状态未知人数。 |
| `persons` | array<object> | 每个人员的检测结果列表。 |
| `finished_at` | string | 检测算法运行结束时间。 |

`persons` 元素结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `person_id` | int | 人员序号，从 1 开始。 |
| `box` | array<int> | 人员框 `[x1, y1, x2, y2]`。 |
| `status` | string | 安全帽状态：`helmet`、`no_helmet` 或 `unknown`。 |
| `confidence` | float | 用于判断该人员状态的置信度。 |
| `associated_helmet_box` | array<int> 或 null | 关联到的安全帽框。 |
| `associated_no_helmet_box` | array<int> 或 null | 关联到的未戴帽头部框。 |

### 4.3 叉车作业安全检测接口

对应功能：

- 21叉车司机佩戴安全帽检测
- 22叉车作业半径站人检测
- 23叉车作业是否有围蔽检测

接口：

```text
POST /api/detect/forklift-work
```

请求类型：

```text
multipart/form-data
```

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | file | 上传输入图片，可选。 |
| `image_path` | string | 服务端本地图片路径；未上传图片时使用。 |
| `weights_file` | file | 上传模型文件，可选，优先级高于 `weights`。 |
| `weights` | string | 服务端本地权重路径。 |
| `output_dir` | string | 输出目录。 |
| `expected_hazards` | int | 人工输入隐患数量。 |
| `nearby_mode` | string | 周边人员判断模式，可选 `circle`、`original`、`both`。 |
| `conf` | float | 检测置信度，WebUI 默认 `0.4`。 |
| `fence_iou_thres` | float | 围蔽与作业区 IoU 阈值，默认 `0.01`。 |
| `device` | string | 推理设备；Web 后端支持 `auto`、`0`、`1`、`cuda`、`cuda:0`、`cpu`，无 GPU 时会自动回退到 `cpu`。 |
| `forklift_radius_scale` | float | 圆形作业区半径缩放系数。 |
| `min_forklift_radius_px` | float | 最小作业半径，单位像素。 |
| `max_forklift_radius_px` | float | 最大作业半径，单位像素。 |
| `draw_circle_work_zone` | bool | 是否绘制圆形作业区。 |

返回内容：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 接口是否调用成功。 |
| `finished_at` | string | 检测算法运行结束时间，ISO 8601 格式。 |
| `command` | string | 后端实际调用的命令行。 |
| `summary` | string | 叉车作业安全检测摘要，包含作业结果、附近人员、违规项和准确度。 |
| `statistics` | string | 统计文本，包含人员、叉车、围蔽和安全状态。 |
| `accuracy` | float | 监测准确度百分比。 |
| `detected_hazards` | int | 检测出的隐患数量，由附近站人、未戴安全帽、无围蔽等规则汇总。 |
| `expected_hazards` | int | 用户输入的人工隐患数量。 |
| `json` | object | 算法原始 JSON 结果。 |
| `json_text` | string | 格式化后的 JSON 字符串。 |
| `output_media_url` | string | 标注结果图片访问 URL。 |
| `output_json_url` | string | JSON 文件访问 URL。 |
| `logs` | string | 算法标准输出和错误输出。 |

`json` 字段结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `image` | string | 输入图片文件名。 |
| `total_persons` / `person_count` | int | 检测到的人员数量。 |
| `helmet_worn_count` | int | 佩戴安全帽人数。 |
| `no_helmet_count` | int | 未佩戴安全帽人数。 |
| `unknown_count` | int | 安全帽状态未知人数。 |
| `persons` | array<object> | 人员检测与安全帽状态列表。 |
| `has_forklift` | bool | 是否检测到叉车。 |
| `forklift_count` | int | 检测到的叉车数量。 |
| `fence_count` | int | 检测到的围蔽数量。 |
| `has_fence` | bool | 当前图像中是否存在围蔽。 |
| `has_fence_nearby` | bool | 兼容字段；表示当前帧是否检测到至少一个围栏。 |
| `has_person_nearby` | bool | 是否有人员进入叉车作业半径。 |
| `excluded_person_count` | int | 被判断为司机或排除人员的数量。 |
| `forklift_is_safe` | bool | 叉车作业半径和围蔽规则是否安全。 |
| `helmet_is_safe` | bool | 人员安全帽规则是否安全。 |
| `is_safe` | bool | 综合安全结果。 |
| `result` | string | 综合结果：`safe` 或 `unsafe`。 |
| `violations` | array<string> | 违规项列表，例如 `PERSON_NEAR_FORKLIFT`、`NO_FENCE_NEAR_FORKLIFT`、`NO_HELMET`。 |
| `nearby_persons` | array<object> | 进入叉车作业半径的人员列表。 |
| `excluded_persons` | array<object> | 被排除的司机或关联人员列表。 |
| `nearby_mode` | string | 周边人员判断模式。 |
| `compliance_summary` | object | 司机安全帽、围蔽、作业半径等规则汇总。 |
| `ground_point_work_area` | object 或 null | 基于地面点生成的作业区信息。 |
| `road_surface_mask` | object 或 null | 道路区域估计信息。 |
| `road_constrained_work_area` | object 或 null | 道路约束后的作业区信息。 |
| `forklifts` | array<object> | 每辆叉车的检测框、作业区、附近人员、安全状态和违规项。 |
| `finished_at` | string | 检测算法运行结束时间。 |

### 4.4 24叉车行驶速度检测接口

对应功能：

- 叉车行驶速度检测

WebUI 使用异步接口：

```text
POST /api/detect/forklift-speed/start
```

请求类型：

```text
multipart/form-data
```

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | file | 上传输入视频，可选。 |
| `video_path` | string | 服务端本地视频路径；未上传视频时使用。 |
| `weights_file` | file | 上传模型文件，可选，优先级高于 `weights`。 |
| `weights` | string | 服务端本地权重路径。 |
| `output_dir` | string | 输出目录。 |
| `expected_hazards` | int | 人工输入隐患数量。 |
| `conf` | float | 检测置信度，WebUI 默认 `0.4`。 |
| `speed_limit` | float | 超速阈值，单位 `pixel/frame`。该值表示实际速度 `5 km/h` 在当前视频标定下对应的像素速度。 |
| `speed_window` | int | 速度平滑窗口。 |
| `min_bg_matches` | int | 背景运动估计最少匹配点数量。 |
| `motion_model` | string | 背景运动模型，可选 `affine` 或 `homography`。 |
| `device` | string | 推理设备；Web 后端支持 `auto`、`0`、`1`、`cuda`、`cuda:0`、`cpu`，无 GPU 时会自动回退到 `cpu`。 |

提交成功后返回：

```json
{
  "ok": true,
  "job_id": "xxxx",
  "command": "..."
}
```

随后使用：

```text
GET /api/jobs/{job_id}
GET /api/jobs/{job_id}/stream
POST /api/jobs/{job_id}/cancel
```

说明：

- `/stream` 用于实时预览检测画面。
- `/jobs/{job_id}` 用于查询进度、速度、超速帧数和最终结果；任务完成后 `result` 字段中包含完整返回。
- `/cancel` 用于停止检测。
- 实际速度换算公式：`speed_kmh = speed_px_per_frame / speed_limit_px_per_frame * 5.0`。例如 `speed_limit=10.0` 时，`10 px/frame` 对应 `5 km/h`，`20 px/frame` 对应 `10 km/h`。

`GET /api/jobs/{job_id}` 返回字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 任务是否有效。 |
| `job_id` | string | 异步任务 ID。 |
| `status` | string | 任务状态：`running`、`done`、`error`、`canceled`。 |
| `progress` | int | 进度百分比，0 到 100。 |
| `progress_text` | string | 当前进度文本。 |
| `frame_id` | int | 当前处理帧号。 |
| `frame_status` | object | 当前帧速度状态。 |
| `command` | string | 后端实际执行命令。 |
| `result` | object | 任务完成后的最终结果，结构同同步接口返回。 |
| `output_media_url` | string | 输出视频访问 URL。 |
| `error` | string | 失败时的错误信息。 |

速度检测最终 `result` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 接口是否调用成功。 |
| `finished_at` | string | 检测算法运行结束时间，ISO 8601 格式。 |
| `command` | string | 后端实际调用的命令行。 |
| `summary` | string | 速度检测摘要和准确度。 |
| `statistics` | string | 速度统计文本。 |
| `accuracy` | float | 监测准确度百分比。 |
| `detected_hazards` | int | 超速隐患数量；出现超速时为 1，否则为 0。 |
| `expected_hazards` | int | 用户输入的人工隐患数量。 |
| `json` | object | 算法原始 JSON 结果。 |
| `json_text` | string | 格式化后的 JSON 字符串。 |
| `output_media_url` | string | 标注结果视频访问 URL。 |
| `output_json_url` | string | JSON 文件访问 URL。 |
| `logs` | string | 算法日志。 |

速度检测 `json` 字段结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `video` | string | 输入视频文件名。 |
| `fps` | float | 输入视频帧率。 |
| `speed_limit_px_per_frame` | float | 超速阈值，单位 `pixel/frame`。 |
| `speed_limit_kmh` | float | 实际限速，固定为 `5.0 km/h`。 |
| `summary` | object | 视频级速度汇总。 |
| `frames` | array<object> | 每帧叉车速度结果。 |
| `finished_at` | string | 检测算法运行结束时间。 |

`summary` 字段结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `video` | string | 输入视频文件名。 |
| `speed_limit_px_per_frame` | float | 超速阈值。 |
| `speed_limit_kmh` | float | 实际限速，固定为 `5.0 km/h`。 |
| `speed_conversion` | object | 像素速度到真实速度的换算配置和公式。 |
| `total_frames` | int | 实际处理帧数。 |
| `overspeed_frame_count` | int | 出现超速的帧数。 |
| `overspeed_track_ids` | array<int> | 出现超速的叉车轨迹 ID。 |
| `max_speed_px_per_frame` | float 或 null | 最大检测速度。 |
| `max_speed_kmh` | float 或 null | 最大检测速度换算后的真实速度，单位 `km/h`。 |
| `result` | string | 速度检测结果：`safe`、`unsafe` 或 `unknown`。 |
| `stopped_by_user` | bool | 是否被用户中断。 |

`frames` 元素结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_id` | int | 帧号，从 0 开始。 |
| `motion_compensation_success` | bool | 当前帧背景运动补偿是否成功。 |
| `forklifts` | array<object> | 当前帧叉车速度列表。 |

`forklifts` 元素结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `track_id` | int | 叉车跟踪 ID。 |
| `box` | array<int> | 叉车框 `[x1, y1, x2, y2]`。 |
| `ground_point` | array<float> | 叉车底部中心点坐标。 |
| `current_speed_px_per_frame` | float 或 null | 当前帧速度。 |
| `smoothed_speed_px_per_frame` | float 或 null | 平滑后的速度。 |
| `current_speed_kmh` | float 或 null | 当前帧速度换算后的真实速度，单位 `km/h`。 |
| `smoothed_speed_kmh` | float 或 null | 平滑速度换算后的真实速度，单位 `km/h`。 |
| `speed_limit_px_per_frame` | float | 超速阈值。 |
| `speed_limit_kmh` | float | 实际限速，固定为 `5.0 km/h`。 |
| `speed_conversion` | object | 换算说明，包含 `reference_speed_kmh=5.0`、`reference_speed_px_per_frame` 和公式。 |
| `overspeed` | bool | 是否超速。 |
| `speed_status` | string | 速度状态：`ok`、`overspeed` 或 `unknown`。 |
| `motion_compensation_success` | bool | 当前速度是否基于成功的运动补偿。 |
| `speed_source` | string | 速度来源说明。 |

保留同步接口：

```text
POST /api/detect/forklift-speed
```

同步接口会等待整个视频处理完成后返回，一般不建议 WebUI 使用。

### 4.5 物资抽样计数算法接口

对应功能：

- 物资数量识别

接口：

```text
POST /api/detect/material-count
```

请求类型：

```text
multipart/form-data
```

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `source_type` | string | 输入类型，可选 `single_image`、`image_sequence`、`video`、`live_stream`。 |
| `source_file` | file | 上传单张图片、视频或图片序列文件夹。 |
| `source_path` | string | 服务端本地输入路径；未上传文件时使用。 |
| `stream_url` | string | 直播流地址；`source_type=live_stream` 时必填。 |
| `sample_interval` | int | 视频文件或直播流的抽帧检测间隔，默认 `5`，最小为 `1`；`N` 表示每 `N` 个源帧送检一帧。 |
| `model_file` | file | 上传物资计数模型，可选，优先级高于 `model`。 |
| `model` | string | 服务端本地物资计数模型路径，默认 `all_best.pt`；模型 task 必须为 `obb` 或 `detect`。 |
| `output_dir` | string | 输出目录。 |
| `conf` | float | 物资计数检测置信度，默认 `0.5`。 |
| `device` | string | 推理设备；Web 后端支持 `auto`、`0`、`1`、`cuda`、`cuda:0`、`cpu`，无 GPU 时会自动回退到 `cpu`。 |
| `track_high_thresh` | float | BoT-SORT 高阈值，默认 `0.3`。 |
| `track_low_thresh` | float | BoT-SORT 低阈值，默认 `0.1`。 |
| `new_track_thresh` | float | 新轨迹阈值，默认 `0.4`。 |
| `track_buffer` | int | 轨迹缓存帧数，默认 `50`。 |
| `match_thresh` | float | 匹配阈值，默认 `0.25`。 |
| `with_reid` | bool | 是否启用 ReID，默认启用。 |
| `proximity_thresh` | float | ReID 空间接近阈值，默认 `0.6`，范围 `0～1`；越大，允许外观关联的空间条件越严格。 |
| `appearance_thresh` | float | ReID 外观相似度阈值，默认 `0.85`，范围 `0～1`；越大，外观关联条件越严格。 |
| `target_classes` | string | 必填。需要检测和计数的类别名称，使用换行或逗号分隔；按所选模型的 `names` 校验，默认模型下即 `/api/config.material_class_candidates` 返回的九个候选项。未选择的类别不会跟踪、绘制或返回。 |
| `expected_counts` | string | 抽样物资实际数量，每行格式为 `类别名称,实际数量`。 |

提交成功后返回：

```json
{
  "ok": true,
  "job_id": "xxxx",
  "command": "..."
}
```

随后使用：

```text
GET /api/jobs/{job_id}
GET /api/jobs/{job_id}/stream
POST /api/jobs/{job_id}/cancel
```

说明：

- 单张图片模式会输出标注图片。
- 图片序列、视频和直播流模式会输出 H.264 标注视频。默认使用 `libx264`、`CRF 26`；数值越大文件越小、画质越低。
- `/stream` 用于实时预览计数画面。
- `/jobs/{job_id}` 用于查询进度、累计计数、类别计数和最终结果；任务完成后 `result` 字段中包含完整返回。
- `/cancel` 用于停止计数。

#### 物资计数时间戳定义

物资计数算法会在算法任务真正结束时写入时间戳字段：

| 字段 | 类型 | 格式 | 定义 |
|---|---|---|---|
| `finished_at` | string | ISO 8601 本地时区时间，精确到秒，例如 `2026-07-03T09:43:08+08:00` | 算法完成计数、写出结果媒体并生成返回 JSON 的时间。正常完成和用户停止任务时都会写入。 |
| `created_at` | string | 同 `finished_at` | 兼容旧版本字段，目前值与 `finished_at` 完全相同。新开发建议使用 `finished_at`。 |

说明：

- `finished_at` 不是任务提交时间，也不是前端收到结果的时间，而是后端计数算法运行结束并构造结果 JSON 的时间。
- 对于 `single_image`，表示单张图片推理和标注图片写出完成的时间。
- 对于 `image_sequence` 和 `video`，表示所有已处理帧计数完成、输出视频释放完成后的时间。
- 如果用户点击“停止计数”，`stopped_by_user=true`，`finished_at` 表示中断处理完成并生成当前结果的时间。

#### 物资计数异步提交返回字段

调用 `POST /api/detect/material-count` 后，接口会先返回异步任务信息：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 是否成功创建计数任务。 |
| `job_id` | string | 后端生成的任务 ID，用于查询进度、实时预览和停止任务。 |
| `command` | string | 后端为该任务构造的命令行参数文本，便于排查模型、输入、输出、阈值和设备配置。 |

#### 物资计数任务状态字段

调用 `GET /api/jobs/{job_id}` 查询任务状态时返回：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 任务对象是否有效。 |
| `job_id` | string | 任务 ID。 |
| `status` | string | 任务状态：`running`、`done`、`error`、`canceled`。 |
| `progress` | int | 进度百分比，范围 0 到 100；总帧数未知时可能停留在 0。 |
| `progress_text` | string | 当前进度文本，例如当前帧、累计计数或停止中。 |
| `command` | string | 后端实际调用命令。 |
| `algorithm` | string | 任务算法名称，物资计数固定为 `material_count`。 |
| `frame_id` | int | 当前已处理帧号；单图任务一般为 1。 |
| `frame_result` | object | 当前最新帧实时状态，包含当前累计数量和类别计数。 |
| `frame_status` | object | 与 `frame_result` 内容相同的兼容字段。 |
| `output_media_url` | string | 任务完成后输出图片或视频的访问 URL。 |
| `result` | object | 任务完成或被停止后的最终返回结果；结构见下表。 |
| `error` | string | 任务失败时的错误信息。 |

`frame_status` 字段结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_id` | int | 当前帧序号，从 1 开始。 |
| `name` | string | 当前输入任务名称；直播模式为 `live_stream`。 |
| `source_type` | string | 输入类型。 |
| `sample_interval` | int | 本次任务的抽帧检测间隔。 |
| `target_classes` | array<string> | 本次任务勾选的目标类别。 |
| `total_frames` | int | 总帧数；视频无法读取总帧数时可能为 0。 |
| `total` | int | 当前累计去重计数。 |
| `total_count` | int | 当前累计计数。 |
| `classes` | object<string, int> | 当前累计类别计数，例如 `{ "box": 3, "pallet": 2 }`。 |
| `stopped_by_user` | bool | 当前任务是否已收到停止信号。 |

#### 物资计数最终 `result` 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 接口是否调用成功。 |
| `finished_at` | string | 计数算法运行结束时间，ISO 8601 本地时区格式。 |
| `command` | string | 后端实际调用的命令行。 |
| `summary` | string | 计数摘要，包含累计计数、类别计数和准确度。 |
| `statistics` | string | 计数统计文本。 |
| `json` | object | 算法原始 JSON 结果。 |
| `json_text` | string | 格式化后的 JSON 字符串。 |
| `output_media_url` | string | 标注图片或标注视频访问 URL。 |
| `output_json_url` | string | JSON 文件访问 URL。 |
| `logs` | string | 算法日志。 |

#### 物资计数算法 `json` 字段结构

| 字段 | 类型 | 含义 |
|---|---|---|
| `name` | string | 输入源名称。单图/视频取文件名主干，图片序列取文件夹名，直播模式为 `live_stream`。 |
| `source_type` | string | 输入类型：`single_image`、`image_sequence`、`video` 或 `live_stream`。 |
| `sample_interval` | int | 本次任务的抽帧检测间隔。 |
| `target_classes` | array<string> | 本次任务实际检测的类别名称；名称统一为模型中的类别名称。 |
| `total` | int | 累计识别/计数总数量。单图模式为检测框数量；图片序列/视频模式为去重后的累计轨迹数量。 |
| `classes` | object<string, int> | 只包含勾选类别的计数；勾选但未检出时数量为 `0`。 |
| `output_media_path` | string | 输出标注图片或视频的服务端本地绝对/相对路径。 |
| `stopped_by_user` | bool | 是否被用户中断。`true` 表示用户点击了停止计数，返回的是已处理部分的当前累计结果。 |
| `finished_at` | string | 算法运行结束时间，ISO 8601 本地时区格式。 |
| `created_at` | string | 兼容旧字段，当前与 `finished_at` 相同。 |
| `expected_counts` | object<string, int> | 用户输入的抽样实际数量，键为类别名称，值为实际数量。未填写时为空对象 `{}`。 |
| `accuracy_rows` | array<object> | 各类别检测数量、实际数量和准确度明细。 |

`accuracy_rows` 元素结构：

| 字段 | 类型 | 含义 |
|---|---|---|
| `class_name` | string | 类别名称。 |
| `detected` | int | 算法检测/计数数量。 |
| `actual` | int 或 null | 用户输入的实际数量；未输入时为 `null`。 |
| `accuracy` | string | 准确度文本，例如 `95.00%`、`-` 或 `无法计算`。 |

## 5. WebUI 运行和控件说明

### 5.1 打开 WebUI

启动服务：

```powershell
python safety_tools\safety_web_server.py
```

浏览器访问：

```text
http://127.0.0.1:7860
```

页面顶部包含两个功能页：

- 统一安全监测（支持人员安全帽、司机安全帽、作业半径站人、作业围蔽、叉车速度多选及全选）
- 物资计数系统（支持单图、图片序列、视频和直播流）

### 5.2 通用控件说明

| 控件 | 含义 |
|---|---|
| 输入图片 / 输入视频 / 输入文件 | 选择本机图片、视频或文件夹上传到后端。 |
| 本地路径输入框 | 可以手动填写服务端机器上的已有文件路径。 |
| 权重文件 / 检测模型 | 模型权重路径，也可以点击“选择模型”上传模型文件。 |
| 选择模型 | 打开浏览器文件选择框，选择 `.pt`、`.onnx`、`.engine` 等模型。 |
| 输出目录 | 检测结果保存目录。 |
| 选择目录 | 调用后端本机系统目录选择框，选择真实输出目录。 |
| 置信度 / conf | 检测阈值，值越高，检测越严格；默认 `0.4`。 |
| 运行状态 | 显示当前任务状态，例如空闲、检测中、完成、失败、停止中。 |
| 检测统计结果 / 类别计数与准确度 | 展示算法统计文本。 |
| 结果摘要 | 展示摘要和 JSON 详情。 |
| 运行日志 | 展示接口调用命令、错误、任务日志以及跟踪前的 YOLO 逐帧检测类别、置信度和坐标。 |
| 打开输出文件 | 打开标注后的图片或视频。 |
| 打开 JSON | 打开检测或计数结果 JSON 文件。 |

### 5.3 统一安全监测页面

控件说明：

| 控件 | 含义 |
|---|---|
| 检测所有功能 | 一次勾选或取消全部五项安全监测功能。 |
| 人员安全帽检测 | 输出普通人员戴帽、未戴帽和未知状态。 |
| 叉车司机安全帽检测 | 输出司机识别结果及司机安全帽状态。 |
| 叉车作业半径站人检测 | 输出进入叉车作业区的非司机人员。 |
| 叉车作业围蔽检测 | 输出同一帧中是否同时检测到叉车和围栏。 |
| 叉车速度检测 | 输出叉车像素速度、km/h 换算值和超速状态。 |
| 输入类型 | 选择单张图片、图像序列、视频文件或直播流。单张图片不能启用叉车速度。 |
| 输入源 / 直播地址 | 上传文件、选择文件夹、填写服务端本地路径或填写直播地址。 |
| 测试直播 | 在正式检测前验证后端能否连接并读取直播画面。 |
| 采样间隔 | 每隔多少个源帧执行一次检测，最小为 `1`。 |
| 权重文件 | 统一安全检测模型路径，默认 `all_best.pt`。 |
| 输出目录 | 标注图片、标注视频和 JSON 保存目录。 |
| 人工隐患总数 | 人工统计隐患数量，用于与所选功能检测隐患总数计算准确度。 |
| 检测置信度 | YOLO 检测阈值；内置 NMS 由后端统一配置，页面不提供 NMS 调节项。 |
| nearby-mode | 周边人员判断模式：`circle`、`original` 或 `both`。 |
| fence iou | 仅用于生成可选的围栏形状人员接近区域，不再用于围蔽有无判定。 |
| radius scale / min radius / max radius | 叉车圆形或椭圆形作业区的尺寸参数。 |
| 绘制作业区 | 兼容选项；安全结果画面固定仅绘制YOLO原始检测框。 |
| 速度阈值 | `5 km/h` 对应的像素速度阈值，单位 `px/frame`。 |
| speed-window | 速度平滑窗口。 |
| min-bg-matches | 背景运动补偿最少特征匹配点数。 |
| motion-model | 背景运动模型，`affine` 或 `homography`。 |
| device | 推理设备，WebUI 会自动检测 GPU；无 GPU 时自动使用 `cpu`，也可手动填写 `0`、`1` 或 `cpu`。 |
| 开始统一监测 | 按当前勾选功能启动异步检测任务。 |
| 停止监测 | 停止当前检测并返回已经处理部分的结果。 |

使用流程：

1. 打开“统一安全监测”页，勾选一个、多个或全部检测功能。
2. 选择输入类型并上传输入源；直播模式填写地址后先点击“测试直播”。
3. 确认权重、输出目录、人工隐患总数、置信度和设备。
4. 如果勾选叉车作业功能，设置作业半径和围蔽参数；如果勾选叉车速度，设置速度和运动补偿参数。
5. 点击“开始统一监测”。
6. 实时统计、最终 JSON 和隐患数只包含已勾选功能；结果画面保存YOLO原始检测框且不带文字。
7. 任务完成后查看结果摘要、输出媒体和 JSON；也可以点击“停止监测”提前结束。

### 5.4 物资计数系统页面

控件说明：

| 控件 | 含义 |
|---|---|
| 输入类型 | 选择 `单张图片`、`图片序列`、`视频文件` 或 `直播流`。 |
| 输入文件 | 根据输入类型选择图片、文件夹或视频。 |
| 输入路径 | 可填写服务端本地图片、文件夹或视频路径。 |
| 直播地址 / 测试直播 | 直播模式下填写流地址，并在计数前测试连接和原始画面。 |
| 采样间隔 | 每隔多少个源帧执行一次物资检测与跟踪，默认 `5`。 |
| 检测模型 | 物资计数模型路径，默认使用项目根目录下的 `all_best.pt`；支持 OBB 与普通目标检测模型，可计数类别随模型自动更新。 |
| 输出目录 | 标注图片、标注视频和 JSON 保存目录。 |
| 置信度 | 物资计数检测置信度，默认 `0.5`。 |
| device | 物资计数默认设备为 `0`；无 GPU 时后端自动回退到 `cpu`，也可手动填写其他设备。 |
| 高阈值 | BoT-SORT 高置信检测阈值。 |
| 低阈值 | BoT-SORT 低置信检测阈值。 |
| 新轨迹阈值 | 新目标轨迹初始化阈值。 |
| 缓存帧数 | 轨迹丢失后保留的帧数。 |
| 匹配阈值 | 跟踪匹配阈值，默认 `0.25`。 |
| 启用 ReID | 是否启用 ReID 外观特征辅助跟踪。 |
| 计数目标类别 | 表格按所选模型的 `names` 自动生成：默认 `all_best.pt` 显示 `large cable` 及八类配电箱、变压器共九类，换成其他模型后显示该模型自己的类别。勾选需要检测的类别，并可在同一行填写实际数量用于准确度计算。 |
| 开始计数 | 启动物资计数任务。 |
| 停止计数 | 停止当前物资计数任务。 |

输入类型说明：

| 输入类型 | 选择行为 | 输出 |
|---|---|---|
| 单张图片 | 弹出图片选择框 | 标注图片和 JSON |
| 图片序列 | 弹出文件夹选择框 | 实时预览、标注视频和 JSON |
| 视频文件 | 弹出视频选择框 | 实时预览、标注视频和 JSON |
| 直播流 | 填写直播地址并测试连接 | 直播原始画面、实时计数结果、标注视频和 JSON |

使用流程：

1. 打开“物资计数系统”页。
2. 选择输入类型。
3. 选择图片、文件夹或视频，也可以填写服务端本地路径；直播模式填写地址并先测试直播。
4. 选择物资识别模型。
5. 设置输出目录、置信度、设备和跟踪参数；WebUI 会在无 GPU 电脑上自动使用 `cpu`。
6. 在“计数目标类别”表格中至少勾选一个类别；如需准确度统计，在该类别后填写实际数量。
7. 点击“开始计数”。未勾选类别不会检测、跟踪、绘制或出现在结果中。
8. 图片序列和视频显示实时计数预览；直播模式同时显示直播原始画面和实时计数结果；非直播模式只显示检测结果。
9. 任务完成后查看累计数量、类别数量、准确度、输出视频或图片和 JSON。
10. 如需中断，点击“停止计数”。

### 5.5 输出文件说明

默认输出目录：

```text
outputs/safety_web/
```

常见输出文件：

| 文件类型 | 说明 |
|---|---|
| `safety_combined_result_*.jpg` | 统一安全监测单图标注结果，仅绘制已勾选功能。 |
| `safety_combined_result_*.json` | 统一安全监测 JSON，仅包含已勾选功能。 |
| `helmet_result_*.jpg` | 安全帽检测标注图片。 |
| `helmet_result_*.json` | 安全帽检测 JSON。 |
| `forklift_work_result_*.jpg` | 叉车作业安全检测标注图片。 |
| `forklift_work_result_*.json` | 叉车作业安全检测 JSON。 |
| `forklift_speed_result_*.json` | 叉车速度检测 JSON。 |
| `material_count_*.jpg` | 单图物资计数结果。 |
| `material_count_*.mp4` | 图片序列、视频或直播物资计数结果。 |
| `material_count_*.json` | 物资计数 JSON。 |

### 5.6 常见问题

#### 选择输出目录没有弹窗

输出目录选择框由后端服务所在机器弹出系统目录选择对话框。如果服务运行在无桌面环境、远程服务器或受限环境中，可能无法弹出。此时可以手动填写服务端本地输出路径。

#### 视频不能回放

当前 `/api/file` 已支持 Range 请求，物资计数输出视频默认由 PyAV/libx264 保存为 H.264 MP4。如果视频过大或不能回放，建议：

- 使用 Chrome 或 Edge。
- 确保重新运行任务生成新视频。
- 确认已安装 `requirements.txt` 中的 `av`，以启用可控质量的 `libx264` 编码。
- 默认 `CRF 26`；需要更小文件时可在命令行使用 `--video-crf 28`，需要更高画质时可使用 `--video-crf 23`。
- 如仍不兼容，可使用 FFmpeg 将输出视频转码为 H.264 MP4。

#### 模型加载失败

检查：

- 模型路径是否存在。
- 模型是否与当前 Ultralytics 版本兼容。
- 当前 Python 环境是否正确安装 PyTorch、OpenCV、NumPy。
- GPU 推理时 `device` 是否正确，例如 `0` 或 `cpu`。

#### NumPy 与 PyTorch 版本报错

如果出现类似 NumPy ABI 不兼容提示，可以尝试：

```powershell
pip install "numpy<2"
```

或安装支持当前 NumPy 版本的 PyTorch。

#### 电脑没有GPU

目前程序能够自动识别电脑是否有GPU从而调用GPU或CPU运行，如没有GPU可以自动调用CPU运行，但是CPU运行的速度较慢，为保险起见，请在算法运行前在device控件中输入cpu从而调用cpu运行
