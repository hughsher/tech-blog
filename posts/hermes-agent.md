# 深入理解 Hermes Agent：多智能体博客写作协作系统

一篇博客从选题到发布，如果拆成"研究、写作、审校、发布"四道工序交给四个 AI 各自完成，它们该怎么协作？答案往往是一堆脚本和祈祷：脚本调度脆、状态丢了没人管、哪个环节卡死无从排查。Hermes Agent 给出的答案要硬核得多——一套持久化的任务看板，让多个有名字、有记忆的 agent 像一支真正的编辑团队那样接力工作。本文就拆开看看这套系统是怎么运转的，以及它背后的设计为什么值得抄。

## Hermes Agent 是什么

Hermes Agent 是 Nous Research 开源的一个 AI agent 框架，与 Claude Code、Codex、OpenClaw 属同一类：一个通过工具调用与系统交互、能自主写代码和干活的终端原生 agent。源码在 github.com/NousResearch/hermes-agent，一条 `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash` 命令即可装上，跑在 Linux、macOS、Windows（原生）、WSL，甚至 Android 的 Termux 上。

但它和同类相比，打的不是"会写代码"这张牌，而是"越用越强"和"可以住下来"。三个差异化最突出：

**第一，通过 skills 自我改进。** Hermes 把可复用流程存成 skills，下次会话按需加载；还能从经验里学习——自主创建 skill、使用中改进、周期性把知识固化下来。这是一个闭环，不是一次性工具。

**第二，跨会话持久记忆。** 它有两个记忆文件：MEMORY.md（agent 自己的笔记，2200 字符上限）和 USER.md（对用户的画像，1375 字符上限），每次会话开始以冻结快照注入。改过的东西下一场对话还在。

**第三，多平台 gateway 与 profiles 多实例。** 同一个 agent 能跑在 Telegram、Discord、Slack、WhatsApp、Signal、Email、钉钉、飞书、微信、QQ 等 21+ 平台（19 个 gateway 原生支持，IRC 和 Teams 走插件），且拥有完整工具访问权，不只是聊天壳。provider 也无关——20+ 家模型供应商，能在工作流中途切换，还支持 credential pools 在多把 API key 间自动轮换。而 profiles 让你在同一台机器上跑多个互不干扰的 Hermes 实例，各自隔离配置、会话、skills 和记忆。

这几条拼在一起，指向同一个能力：**它不是一个用完就丢的代码助手，而是一个可以长期驻留、被编排成团队的平台。** 多界面（CLI、Ink TUI、Electron 桌面、Web dashboard、给 IDE 用的 ACP）驱动的都是同一套内核。

## 多智能体协作的底层机制

"多个 agent 协作"听起来玄，Hermes 的落地其实是一块看板加一个调度器，再加上几块积木。

**Kanban 是核心。** 它是一块持久化的任务看板，底层是一个 SQLite 文件 `~/.hermes/kanban.db`，跨所有 profile 共享。每个任务就是库里的一行，每次交接也是任何人都能读写的一行，每个 worker 是一个拥有自己身份的完整 OS 进程。任务走一个明确的状态机：`triage → todo → ready → running → blocked → review → done → archived`。任务之间可以用 `parents` 声明依赖：子任务在所有父任务都 `done` 之前，一直停在 `todo`，父任务完成后才被自动提升为 `ready`。这就是"依赖门"——流水线里 writer 卡在 researcher 完成前根本不会被派发。

**Dispatcher 负责调度。** 它是一个长驻循环，默认每 60 秒 tick 一次，跑在 gateway 进程内。每轮它做四件事：回收过期的 claim、回收崩溃的 worker、把 `ready` 的任务提升出来、原子地 claim 一个任务并 spawn 出对应的 profile 进程。worker 醒来第一件事是调用 `kanban_show()` 读自己的任务，干活，最后用 `kanban_complete`、`kanban_request_review` 或 `kanban_block` 收尾。同一任务连续 spawn 失败两次，dispatcher 会自动 block 它，避免对着一个不存在的 profile 空转。

**delegate_task 是另一条路，但性质完全不同。** 它是函数调用式的委派：父 agent 派生一个匿名 subagent，阻塞等它返回，失败即失败、不可恢复、人类插不上手，审计还会随上下文压缩丢失。Kanban 则是持久的工作队列：worker 是有名字、有记忆的 profile，能 block→unblock→重跑，崩溃了 dispatcher 会 reclaim 重派，人随时能 comment 介入，每个交接在 SQLite 里永久留痕。一句话总结：**delegate_task 是函数调用，Kanban 是工作队列，其中每次交接都是一行谁都能读改的记录。** 需要跨 agent 边界、跨重启存活、可能需要人介入、需要可追溯时，用 Kanban；只需要一个短暂推理答案就继续时，用 delegate_task。

**skills 和 memory 是让角色"专业"起来的积木。** skills 是按需加载的知识文档，走渐进披露——先看 `skills_list()` 的名字和描述，真需要了再 `skill_view(name)` 拉全文，省 token。memory 则让常驻角色跨会话累积，注入的是冻结快照，中途改动落盘但要下一场会话才生效。这两块再加上 cron（定时/事件触发投递）和 curator（自动维护 agent 自建 skill 的生命周期，避免越用越堆积），构成了一套能自持运转的底层系统。

## 博客写作流水线：五个角色如何接力

把上面的机制套进一个真实场景，就是本文的选题——一条五角色的博客写作流水线：

```
orchestrator → researcher → writer → reviewer → publisher
（拆解）        （研究简报）   （正文）   （审校定稿）  （发布）
```

orchestrator 扇出的四张卡通过 `parents` 串成一条链：researcher 是 writer 的父任务，writer 是 reviewer 的父任务，以此类推。dispatcher 会严格按照这条链解锁——父卡没 `done`，子卡就一直停在 `todo`。

**orchestrator（编排者）** 不碰具体实现，只做分解和路由。它读任务后用 `kanban_create` 扇出四张子卡，每张写明 assignee 和 body，并用 `parents=[...]` 表达依赖，然后 `kanban_complete` 收工。设计决策（命名、结构、交接格式）在这里一次性定死，写进每张子卡的 body，下游 worker 看不到兄弟卡，全靠这些写好的决策对齐。

**researcher（研究员）** 用 terminal、文件、web 工具收集核实素材，产出一份带来源的研究简报，`kanban_attach` 挂到卡片上，再 `kanban_complete(summary=..., metadata=...)` 交接。交接分三层：summary 是人类可读的收尾描述，metadata 是机器可读的结构化字段（改了哪些文件、如何验证、留了什么风险），artifacts/attachments 是真正的交付文件——上限 25MB，存在 `<hermes-home>/kanban/attachments/<task_id>/`。

**writer（写作者）** 醒来后第一件事，是用 `kanban_attachments(parent_task_id)` 拿到上游简报的绝对路径，`read_file` 通读，然后基于简报写正文。关键约束是事实只来自简报，不确定的细节宁可省略——这正是交接链的意义：每个角色只处理上一环已经核实过的素材，不必各自回访源头。

**reviewer（审校者）** 走的是 review 列。实现者用 `kanban_request_review` 交审，reviewer 通过就 `kanban_complete`，不满意就 `kanban_request_changes(reason=...)` 打回实现者重改——这条路径不触发 block 计数，可以反复迭代，直到定稿。

**publisher（发布者）** 拿到终稿做发布。纯本地工作的卡 local-only 即可收工；涉及 PR 的卡可以用 completion contract 卡住，直到仓库要求的 CI 真正通过才算 done。

这条链里最值得玩味的是两件事。一是**故障自愈是现场发生的，不是文档里的承诺**：本次流水线的 orchestrator 卡第一次 run 直接崩溃，dispatcher 记为 crashed 后把它重派为 `ready`，第二次 run 成功扇出全部下游卡——"崩溃→reclaim→重试"就是这么实打实跑通的。二是**本文本身就是这条流水线的产物**：你正在读的这篇文章，正是 researcher 卡交付的研究简报经 writer 改写而来，后面还要过 reviewer 和 publisher 两道关。

## 这套设计的启发

把 Hermes 的多智能体协作拎出来看，三个设计选择最值得借鉴。

**用持久队列替代进程内子代理。** 进程内的 subagent swarm 脆弱、不可恢复、人类进不去。Hermes 把协作从"函数调用"升级成"工作队列"——每个任务是一行数据，每个交接是另一行数据，状态机是公开的，任何人（agent 或人）随时能读能改。这让协作从一次性动作变成了可审计、可重试、可介入的长期工程。

**用依赖门替代靠喊话协调。** 流水线的顺序不是靠某个 agent 记住"等 writer 写完我再审"，而是硬编码在 `parents` 依赖里，由 dispatcher 机械执行。子卡在父卡完成前物理上无法被派发，死锁和顺序错乱从机制上被排除——这比任何"请等待上游完成"的提示词都可靠。

**用分层交接替代共享上下文。** 每个角色只拿它需要的那一层：summary 给人读，metadata 给机器读，artifacts 给下一道工序读。没有谁要把整个上游的会话历史装进脑子，也没有谁会凭空猜一个没核实过的细节。这种"交接面"的设计，是让多个 agent 能真正并行且不出错的关键。

对想做多 agent 系统的人，这三个选择比任何具体 API 都更有迁移价值：把协作建模成持久状态，把顺序建模成显式依赖，把信息建模成分层交接面。

## 结语

Hermes Agent 的有趣之处，不在"又一个会写代码的 agent"，而在它把 agent 从一次性工具做成了能长期驻留、能自我改进、能像团队一样被编排的平台——skills 让它越用越强，memory 让它记得住你，Kanban 让一群有名字的 agent 在一条可审计的流水线上接力。当一个博客写作任务能被拆成五道工序、由五个角色靠依赖门和分层交接自动跑通，它演示的其实是一套更普适的协作范式：不是更聪明的单体，而是更可靠的编排。
