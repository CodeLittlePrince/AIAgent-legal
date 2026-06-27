"""法律助手离线评估工具包。

本包提供用于衡量法律助手质量的各类评估能力，主要包括：

- RAG 检索指标（如 Recall@K）：检验知识库检索是否命中预期来源
- LLM 评审（LLM Judge）：用大模型对回答的相关性、准确性等维度打分
- Agent 端到端基准测试（agent_benchmark）：模拟多轮对话任务并生成报告

典型用法是从子模块直接导入所需函数，例如 ``rag_metrics``、``llm_judge``、
``agent_benchmark`` 等。
"""
