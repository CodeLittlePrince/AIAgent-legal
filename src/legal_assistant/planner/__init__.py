"""意图规划（Planner）子包。

负责理解用户消息的意图（法律咨询、天气查询、通用对话等），
并决定后续 Agent 应走哪条处理路径。
"""

from legal_assistant.planner.intent import classify_by_rules
from legal_assistant.planner.router import PlanResult, classify

# 对外公开的 API：其他模块应从此处导入，而非直接引用子模块
__all__ = ["PlanResult", "classify", "classify_by_rules"]
