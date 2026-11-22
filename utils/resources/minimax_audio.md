# MiniMax 音频配置与参数

这些资源供所有音频任务共用，和具体的 user_query 无关。样例脚本、音频文件、运行报告仍保存在各自的 query 缓存目录。

| 文件 | 用途 |
| --- | --- |
| [minimax_voices.json](minimax_voices.json) | 批量合成默认读取的配置 |
| [minimax_config.example.json](minimax_config.example.json) | 可复制后通过 `--config` 指定的配置模板 |
| [minimax_system_voices.json](minimax_system_voices.json) | 系统音色总表快照，包含 voice_id、voice_name 和 description |
| [minimax_english_voices.json](minimax_english_voices.json) | 英文音色子集，便于选择音色 |

目录快照于 2026-11-06 从当前国内站账号的 [Get Voice 接口](https://platform.minimax.cn/docs/api-reference/voice-management-get)读取，共 303 个系统音色，其中英文目录 16 个。它不是已经启用的音色池；实际候选由 `voice_pools` 决定。可参考[官方系统音色列表](https://platform.minimax.cn/docs/faq/system-voice-id)。

## 配置参数

| 参数 | 当前默认值或含义 |
| --- | --- |
| `base_url` | `https://api.minimax.cn/v1`，实际请求追加 `/t2a_v2`；海外站可按账号使用对应入口 |
| `model` | `speech-2.8-hd` |
| `sample_rate` | `32000` Hz，输出 16-bit 单声道音频 |
| `audio_format` | 共享配置默认 `flac`，可选 `wav`；旧配置省略该项时仍使用 WAV |
| `default_language` | `en`；片段有 `language` 时优先使用片段设置 |
| `voice_pools.female` | 女声候选 ID 列表，当前 5 个普通英文女声 |
| `voice_pools.male` | 男声候选 ID 列表，当前 4 个普通英文男声 |
| `voice_pools.auto` | 未指定 gender 的说话人候选池，当前为上述 9 个音色 |
| `speaker_voices` | 可选的 label 到 voice_id 的固定映射，优先级高于自动分配；默认空对象 |
| `voice_assignment_seed` | `20261106`；与题目标识一起确定音色，修改可重新分配 |
| `default_voice_id` | 单人脚本没有可用候选池时的兜底音色 |
| `voice_defaults.speed` | 默认语速 `1.1`，项目允许范围 0.9–1.8（含边界）；片段可在此范围内覆盖默认值 |
| `synthesis_mode` | `segment`（默认，逐片段请求）或 `speaker`（同题内按相同音色和参数合并请求，再切回片段） |
| `speaker_pause_s` | 合并模式的分隔停顿，默认 `1.0` 秒，允许 0.8–3 秒；最终对话仍采用原 `pause_after_ms` |
| `speaker_fallback` | 默认 `true`，合并音频无法可靠切分时回退到逐片段合成；实验可设为 `false`，保留原始返回供检查 |
| `voice_defaults.vol` | 默认音量 `1`，大于 0 且不超过 10 |
| `voice_defaults.pitch` | 默认音高 `0`，范围 -12–12 的整数 |
| `timeout_s` | 单次请求超时，当前 120 秒 |
| `concurrency` | 全批次共用的语音片段请求并发上限，默认 `20`；命令行 `--concurrency` 优先 |
| `rpm` | 全批次每分钟请求数上限，当前按账户额度设为 `20`；命令行 `--rpm` 优先，20 RPM 对应平均每 3 秒发出一次请求 |
| `max_attempts` | 每个片段最多请求 `6` 次，包含首次请求；鉴权、余额和参数等不可重试错误会直接报错 |

当前默认配置面向英文材料；其他语言或音色要求可以从完整音色目录中选取合适 ID，另存配置并使用 `--config`。API Key 仅通过环境变量读取，不写入共享 JSON 或本地 TXT 文件。一键脚本 `synthesize_audio.py` 默认使用 `MINIMAX_API`，也兼容旧名 `MINIMAX_API_KEY`；Windows 下支持读取用户和系统保存的环境变量。读取成功时控制台显示所用变量名和来源，不显示密钥值。`--dry-run` 不读取密钥。

在某个 PowerShell 终端切换到第二个密钥，只需指定变量名，无需复制密钥值：

```powershell
$env:MINIMAX_API_ENV = 'MINIMAX_API_2'
```

随后从同一终端启动合成命令，会优先读取该终端的 `MINIMAX_API_2`；若终端尚未刷新，也会读取 Windows 用户／系统环境中的 `MINIMAX_API_2`。找不到第二个密钥时会报错，不会自动改用第一个。此设置仅作用于当前终端和之后启动的子进程；已运行的合成进程需要重新启动。在当前终端切回第一个密钥：

```powershell
$env:MINIMAX_API_ENV = 'MINIMAX_API'
```

## 音色如何分配

`label` 仅标识同一道题中的人。程序依据 subtask_id、dataset_id、idx、id 和 voice_assignment_seed，按各人的 gender 从候选池选择互不重复的声音。跨题可以换组合，同题重跑时分配一致；独立片段和对话中相同 label 的声音一致。先分配候选较少的人，再分配 auto。显式 `speaker_voices` 固定映射优先。

改变音色池或 seed 后，发生变化的请求会自动重新合成；相同文本、音色和表达设置的片段仍能复用。每题实际分配记录在合成报告的 `voice_map` 中。

## 执行命令

在项目根目录运行；默认处理已有的 user_query_07，但配置始终来自本资源目录：

```powershell
python synthesize_audio.py --limit 3 --concurrency 20
```

只合成抽样文件中的题目：

```powershell
python synthesize_audio.py --eval-json cache/user_queries/user_query_07/generation/planned-v1/evaluation_sample.json --concurrency 20 --rpm 20
```

`--eval-json` 可以指定不含内部脚本字段的导出文件。程序从该文件同目录的 `evaluation_with_audio_settings.json` 中按完整公开记录匹配音频脚本，只处理选中的题目，并保留其原有音色分配和音频路径。不存在唯一匹配时会在调用 API 前报错。输入文件本身已有音频脚本时，无需另找完整清单。

上面的抽样命令会在输入文件目录生成 `evaluation_sample_with_audio_settings.json`、`evaluation_sample_ready.json` 和 `minimax_synthesis_report.evaluation_sample.json`。音频仍写入题目原有的 media 路径；参数一致且缓存完整的片段会复用。添加 `--dry-run` 可以先检查抽样脚本和音色，报告为 `minimax_dry_run_report.evaluation_sample.json`。其他文件名采用相同的命名规则。

输入为 `evaluation.json` 且输出 FLAC 时，音频脚本另存为 `evaluation_with_audio_settings.flac.json`，交付文件为 `evaluation_ready.json`。保留原来的 `evaluation_with_audio_settings.json`，使相同命令续跑时仍能匹配原始导出记录。

处理其他音频任务，只需改输入缓存目录：

```powershell
python synthesize_audio.py --input-cache-dir cache/user_queries/user_query_XX --concurrency 20
```

使用自定义配置或只做预检查：

```powershell
python synthesize_audio.py --input-cache-dir cache/user_queries/user_query_XX --config path/to/custom_voices.json --dry-run
```

底层批处理也默认读取 `utils/resources/minimax_voices.json`：

```powershell
python -m utils.audio_processing.minimax_batch --eval-json path/to/evaluation_with_audio_settings.json --dry-run
```

`--dry-run` 不读取密钥、不调用语音 API。去掉它后，一键脚本完成逐段 TTS、按片段 ID 组装、按配置最后加噪和交付检查。噪声库按 `--noise-root`、环境变量 `NOISE_ROOT`、项目旁的 `../datasets/data/TUT` 顺序选择；当前工作区的默认位置为 `E:\cuhk\BenchAgents\datasets\data\TUT`，可省略 `--noise-root`。独立加噪与音乐加噪混合也使用相同规则。`--limit 3` 的试跑结果带 `.limit3` 后缀；完整批次结果为 `evaluation_ready.json`。

`--input-cache-dir cache/user_queries/user_query_21` 自动读取该任务的
`generation/planned-v1/evaluation_with_audio_settings.json`，处理完整音频清单。
`--eval-json` 则处理指定文件中的记录；指定普通 `evaluation.json` 或筛选后的文件时，
从同目录音频清单匹配并补齐脚本。`user_query_XX` 中的 `XX` 是需要替换的任务编号。
例如保留显式文件选择、按说话人批处理，并使用默认噪声库：

```powershell
python synthesize_audio.py --eval-json cache/user_queries/user_query_21/generation/planned-v1/evaluation.json --synthesis-mode speaker
```

### FLAC 合成与交付

共享配置与示例配置均已设置 `"audio_format": "flac"`。MiniMax 请求直接生成 FLAC；拼接、重叠混音和加噪在内部处理 16-bit PCM，再无损编码为 FLAC，保持采样率、声道、顺序与时间安排。FLAC 读写使用 `soundfile`，该依赖已加入项目安装依赖，无需 FFmpeg；原有 WAV 输入与输出继续支持。

原始 `evaluation_sample.json` 和完整 `evaluation_with_audio_settings.json` 保留原样。合成时自动将当前任务拥有的音频路径及其引用改为 `.flac`，写入抽样的 `evaluation_sample_with_audio_settings.json`，最终 `evaluation_sample_ready.json` 中的路径也为 `.flac`。全量运行会另写 `evaluation_with_audio_settings.flac.json`，避免改写原始完整脚本。

```powershell
python synthesize_audio.py --eval-json cache/user_queries/user_query_07/generation/planned-v1/evaluation_sample.json --dry-run
python synthesize_audio.py --eval-json cache/user_queries/user_query_07/generation/planned-v1/evaluation_sample.json
```

切换格式会使用对应扩展名和请求参数的独立缓存。接收到的文件内容必须与请求的 WAV/FLAC 容器一致；交付检查同时验证文件内容和生成凭据，不会将 WAV 字节改名后作为 FLAC 使用。抽样测评应读取 `evaluation_sample_ready.json`；原始抽样题目仍可用于恢复合成脚本和重新生成。

部分 MiniMax FLAC 返回未填写总采样数。读取器对此使用固定大小缓冲区顺序解码到实际结尾，不按头部时长预分配内存，也不对未知长度流执行 seek；实际采样数据仍逐段校验。片段失败时立即打印片段 ID、异常类型和原因，无需等待整道题完成。

### 按说话人合并请求

```powershell
python synthesize_audio.py --eval-json cache/user_queries/user_query_07/generation/planned-v1/evaluation_sample.json --synthesis-mode speaker --output-dir cache/user_queries/user_query_07/experiments/speaker_batch
```

`--output-dir` 会重写本次任务拥有的音频路径和引用，将新清单、音频与报告写入独立目录；原始抽样、完整脚本和已生成音频保持不变。命令行 `--synthesis-mode` 优先于配置。默认继续使用逐片段模式，便于明确选择何时重新合成。

合并模式只合并同一道题中音色、语言、语速、音量、音高等实际请求参数相同的片段，不跨题合并。每段仍保留原有 ID；参考音频也单独切出。合并后文本达到接口字数上限前会另开一组。

各片段文本之间插入 `<#1.00#>`，并请求词级时间戳。切分时先核对字幕文本与原始台词，结合实际长静音选择边界；优先使用完整句子字幕，避免数字或缩写展开造成逐词文本重复。时间戳不可用时，只接受恰好符合预期数量的分隔静音，存在歧义便回退。切片保留语音边缘的少量静音，并检查切点前后 10 ms 的 RMS 不高于 -50 dBFS；必要时多保留一点词尾，再去掉分隔区的中间部分，交给原来的组装、重叠和加噪流程。不会用台词字数估计切分时间。

每组原始音频、字幕和切分指标保存在片段目录的 `_batches/` 下。切片丢失时可从整组缓存重建；切分失败也保留已收到的音频，便于修复后复用。报告的 `speaker_groups` 记录实际 API 请求次数、切分方式、移除的分隔时长和回退情况。合并主要减少 RPM 占用与请求开销，合成费仍以服务端计费字符数为准；改动组内台词会影响该组的合成缓存。

停顿标记和分隔换行也可能计入服务端的 `extra_info.usage_characters`。这次 20 题实测中，合并请求的该值包含了这些字符，因此减少请求次数不代表同比减少合成费用。

并发池作用在语音片段层，整批题目共享最多 20 个同时执行的请求，不是每道题再开 20 个。即使只试跑 3 道题，也能并行生成其中的多段台词。片段可能乱序返回，但每道题会等到需要的语音全部生成完成，再按 tracks 组装。音色分配在请求前完成，所有结果仍按输入题目顺序写入报告；加噪在语音生成与组装完成后执行。缓存复用、有限重试继续有效，改变并发数不会改变音色或语音请求内容。并发上限不等同于每秒请求数；服务端限流按已有重试机制处理。用 `--concurrency 1` 可串行处理。

并发上限和 RPM 同时生效。账户为 20 RPM 时，即使设了 20 并发，也会按至少 3 秒的间隔发出请求；实际同时运行的请求可能少于 20。所有样本的首次调用和重试共享同一个限速器，缓存命中不占用 RPM。本限速器作用于当前批处理进程；同账户的其他进程仍会共同消耗服务端额度。

收到业务错误 `1002` 或 HTTP `429` 时，整个批次暂停新请求至少 60 秒；若服务端的 `Retry-After` 指定更长时间，则按更长时间等待，再继续重试。控制台会显示等待信息，重试耗尽后保留具体错误。需要切换新版本脚本时，先 `Ctrl+C` 停止原进程，再用同一命令重跑；已成功且参数未变的片段显示为 `cached`，只补生成缺失或失效的片段。

片段、label、tracks、输出引用的格式见[音频片段与多输出协议](../../tools/executor_tools/tools.md#deferred-audio-generation)。
