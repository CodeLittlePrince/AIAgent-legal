# legal_assistant 源码说明

**混合架构：**
- **天气**：`planner/weather_rules` 规则 → `weather` 节点直调 adapter
- **法律 / 闲聊**：Tool Calling Agent，工具在 `tools/` 下注册

## tools 目录

```
tools/
├── constants.py      # tool 名称常量
├── context.py        # AgentToolContext（单次调用副作用）
├── builder.py        # build_agent_tools()
├── registry.py       # 天气 adapter 工厂
├── legal/search.py   # search_legal_knowledge（封装 knowledge/retriever）
└── weather/
    ├── open_meteo.py # 天气 adapter 实现
    └── forecast.py   # get_weather_forecast tool
```

`knowledge/` 负责向量库与检索实现；`tools/legal/` 负责 Agent 可调用的 tool 包装。

## 请求链路

```
route → weather | agent (build_agent_tools) → save_memory
```
