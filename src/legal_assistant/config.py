"""应用配置模块。

从环境变量和 .env 文件读取各项配置，供整个法律助手项目使用。
使用 Pydantic Settings 自动完成类型校验和默认值填充。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局应用配置类。

    所有字段均可通过同名环境变量覆盖默认值。
    例如设置环境变量 DEEPSEEK_API_KEY 即可覆盖 deepseek_api_key。
    """

    # env_file：启动时自动读取项目根目录下的 .env 文件
    # extra="ignore"：忽略 .env 中存在但本类未声明的字段，避免报错
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---------- DeepSeek 大语言模型 ----------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ---------- 数据存储 ----------
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/legal_assistant"
    redis_url: str = "redis://localhost:6379/0"

    # ---------- 向量数据库 Chroma（法律知识库） ----------
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = ""  # 为空时使用远程 Chroma 服务；非空则本地持久化

    # ---------- 文本嵌入模型（用于法律文档检索） ----------
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ---------- 天气工具 ----------
    weather_provider: str = "open_meteo"  # 默认使用免费的 Open-Meteo API
    qweather_api_key: str = ""  # 和风天气 API 密钥（可选）
    gaode_api_key: str = ""  # 高德地图 API 密钥（可选）

    # ---------- 会话与记忆 ----------
    max_history_turns: int = 20  # 单次对话最多保留的历史轮数
    redis_session_ttl_seconds: int = 86400  # Redis 中会话过期时间（秒），默认 24 小时

    # ---------- Langfuse 可观测性（链路追踪与评分） ----------
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # ---------- API 安全与启动行为 ----------
    api_key: str = ""  # 若设置，则 API 请求需携带对应密钥
    skip_auto_ingest: bool = False  # 为 True 时跳过启动时的自动法律文档入库

    # 法律类回答末尾附加的免责声明文案
    legal_disclaimer: str = "本回答仅供参考，不构成法律意见，具体问题请咨询执业律师。"


# 全局单例：项目各处通过 `from legal_assistant.config import settings` 引用
settings = Settings()
