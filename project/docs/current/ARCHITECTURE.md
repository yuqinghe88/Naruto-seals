# 当前模块结构

统一入口为 `python seal.py`。已完成核心提取、分阶段编排及一轮历史案例归档、重复实现合并和固定输入搬迁；没有修复卯印。

| 模块 | 当前职责与实现 |
|---|---|
| compiler | `relations.py`保存唯一关系匹配实现；`service.py`只读取版本化语义材料，再调用已有有限规划算法；不创建手模型或执行FK |
| pose | `initializers.py`保存显式端态初始化；`solvers/hare_reference.py`保存原有限求解实现；`service.py`负责来源、人体审核及后端分发 |
| trajectory | `timing.py`、`motifs.py`、`sequence.py`统一缓动、模体和多关键态控制曲线；三印曲线复用N节点核心；采样不执行FK |
| assembly | `contracts.py`是无求解依赖的数据契约；`pipeline.py`编排阶段；`service.py`只组装控制并运行既有检查；`artifacts.py`统一保存运行产物 |
| animation_model | 固定rig、只读几何/观察、身体挂载和原三视图渲染。rig与render子模块分离；spring/NIH身份与版本明确，不能继承对方验收 |
| tests | 新模块契约与端到端回归；旧根目录测试暂留，和新测试一起运行，不宣称全部历史测试已经迁移 |
| docs | 当前入口说明、决策、报告及机器索引；`manager.py`只扫描和索引，不执行实验 |
| assets | 单一catalog索引、独立spring模型与构造种子、语义节点材料、模型/来源SHA和人工审核记录 |

## 数据边界

依次传递 `semantic_plan → pose_set → trajectory_controls → assembled_frames → render_result`，都使用 `naruto.stage_artifact.v1`，显式记录rig_id、模型SHA、权威SHA、来源与阶段结果。`assembly.contracts`是叶模块，其他模块可以依赖它；禁止反向导入编排器。

模型加载读取assets，语义编译读取语义材料，渲染只读取已组装帧。编译不再为了拿到语义目标而构造整套手；spring构造器也不再调用RAM语义生成器，只读保存的兼容种子。矩阵、根点、骨长、耦合与碰撞算法保持原值。

只读基础几何（线段最近点、掌盒距离、盒相交、类别距离比较）从Feedback中抽出，原函数名继续兼容导出；只读快照诊断也独立出来。完整关系观察器、NIH执行域和旧有限规划辅助函数仍有遗留依赖，尚未宣称所有循环都解除。

## 人工验收与预算

卯印被用户明确否决，`assets/review_status.json`记录rejected_by_user。旧static proxy pass保留为历史测量，不能覆盖人工否决。普通组装在创建模型前阻断；只有显式`--allow-rejected`的迁移诊断回放可以读取它，而且结果继续标为rejected_endpoint_present。没有重新求解。

一个RunBudget贯穿阶段，FK和抽象展开分别计数，阶段文件可以携带预算续跑。保存的旧计划复用不重复计入新编译展开，旧成本单列source_work；重新编译得到190/200后，后续阶段不能把它清零。分阶段初始化13次FK、最终组装后174/200，验证无预算重置。

## 历史与发布边界

92个无静态入向导入的历史叶案例移到legacy/cases，cases和seal_registry中的可执行script/solver路径同步更新。仅登记，不自动调度；旧文件名、原SHA和独立策略仍可追溯。仍被导入的求解器、兼容导出和特殊诊断保留。

77个普通三视图脚本合为animation_model/render/historical.py与historical_cases.py，保留标题、说明、掌根显示差别和三份显示坐标变换。额外对齐流程的bird_pair_weave仍独立。9份指对覆盖率、3份拇指覆盖率改调pose/legacy_metrics.py，阶段和自由度掩码没有合并。

204份被代码、策略或配方读取的数值输入原字节搬到assets/inputs/legacy_20261031。assets/input_paths.json登记旧名、版本路径及SHA。input_path只用于读取，output_path禁止新阶段写入固定输入；不留下symlink，也不搜索最新结果。冻结报告中的旧source字符串保留为历史身份，当前catalog使用实体素材路径。查看器需要的3份JSON由publish-viewer生成发布副本，程序输入始终仍取assets。

剩余非统一诊断、旧测试文件、研究分支动态dist读取和旧Markdown还需按依赖整理，不宣称全库已迁移。主线编译与组装已验证不读取dist。详细数量与入口见CLEANUP.md。

渲染器增加combined-only模式，新入口默认只输出合并视频、GIF、输入帧报告和一个媒体清单，不再每次复制十段视频和上百张静态图；仍使用原蓝/橙配色、三视图与身体参照，没有新播放器。
