"""自研 Agent 运行时：【原创增量 ①】

不依赖任何第三方 Agent/环境框架——工具协议、循环、状态机、超时兜底、预算闸门
全部由本项目实现。这不是"重复造轮子"的洁癖，而是三个具体理由：

1. **成本闸门必须能拦住调用**。预算触顶时要在**下一次模型调用之前**终止，
   而不是在框架回调里事后统计——后者只能告诉你花了多少钱，拦不住继续花；
2. **工具参数解析的容错策略是本项目的**。推理模型给出畸形 JSON 是常态，
   重试几次、何时降级、降级成什么，是与本项目数据分布绑定的决定；
3. **依赖倒置**。依赖一个 Agent 框架会把"消息格式""工具调用协议""用量口径"
   一并外包，换提供商时改动会扩散到全部业务代码。
"""

from scitrace.agent.budget import Budget
from scitrace.agent.runtime import AgentRunResult, AgentRuntime
from scitrace.agent.state import AgentState, status_line
from scitrace.agent.tools import Tool, ToolOutcome, build_default_tools

__all__ = [
    "AgentRunResult",
    "AgentRuntime",
    "AgentState",
    "Budget",
    "Tool",
    "ToolOutcome",
    "build_default_tools",
    "status_line",
]
