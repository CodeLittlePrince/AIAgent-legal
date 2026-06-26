# 法律文档库（完整版）

本目录存放用于 RAG 检索的**完整**中国法律 Markdown 文本。

## 免责声明

1. 文本来源于国家法律法规数据库（flk.npc.gov.cn）或公开镜像，仅供参考。
2. 正式法律适用请以全国人大常务委员会公报及官方公布文本为准。
3. AI 助手基于本文档生成的回答不构成法律意见。

**最近更新：** 2026-06-26

## 文档列表

| 文件 | 来源 | 大小 |
|------|------|------|
| `中华人民共和国民法典.md` | flk.npc.gov.cn | 329,741 bytes |
| `中华人民共和国劳动法.md` | flk.npc.gov.cn | 25,674 bytes |
| `中华人民共和国劳动合同法.md` | flk.npc.gov.cn | 35,582 bytes |
| `中华人民共和国消费者权益保护法.md` | flk.npc.gov.cn | 24,162 bytes |
| `中华人民共和国刑法.md` | flk.npc.gov.cn | 220,008 bytes |

## 更新方式

```bash
python scripts/download_legal_docs.py
```

国内优先从 flk.npc.gov.cn 下载；失败时使用 GitHub 镜像（可配置 HTTP 代理）。
