# logging_setup.py （建议文件名，或保留在原文件）
import logging
import os
from logging.handlers import RotatingFileHandler

# --- 配置参数（保持你的设计）---
DEFAULT_LOG_FILE = os.getenv("TABLEAGENT_LOG_FILE", "table_agent.log")
DEFAULT_LOG_LEVEL = os.getenv("DATAFLOW_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_FILE_LEVEL = os.getenv("DATAFLOW_LOG_FILE_LEVEL", "DEBUG").upper()
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5
SUCCESS_LEVEL_NUM = 25

# ANSI 颜色码（保持不变）
COLOR_MAP = {
    "DEBUG": "\033[46m\033[30m",
    "SUCCESS": "\033[32m",
    "WARNING": "\033[43m\033[30m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m\033[37m",
    "RESET": "\033[0m",
}
FIELD_COLORS = {
    "time": "\033[90m",
    "name": "\033[35m",
    "location": "\033[96m",
}

# 🔹 只注册一次 SUCCESS（全局生效）
if not hasattr(logging, 'SUCCESS'):
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)
    logging.Logger.success = success


class ColorFormatter(logging.Formatter):
    def format(self, record):
        level_name = record.levelname
        level_color = COLOR_MAP.get(level_name, "")
        reset = COLOR_MAP["RESET"]
        asctime = self.formatTime(record, self.datefmt)
        return (
            f"{FIELD_COLORS['time']}{asctime}{reset} | "
            f"{level_color}{level_name:<8}{reset} | "
            f"{FIELD_COLORS['name']}{record.name}{reset} | "
            f"{FIELD_COLORS['location']}{record.filename}:{record.lineno}{reset} | "
            f"{level_color}{record.funcName:<10}{reset} | "
            f"{level_color}{record.getMessage()}{reset}"
        )


class PlainFormatter(logging.Formatter):
    def format(self, record):
        record.levelname = record.levelname.ljust(5)
        return super().format(record)


# 🔹 核心：setup_logging —— 主程序调用一次即可
def setup_logging(
    log_path: str = None,
    console_level: str = None,
    file_level: str = None,
    force: bool = False,
):
    """
    主应用日志初始化（必须在 get_logger() 被大量使用前调用！）
    """
    log_path = log_path or DEFAULT_LOG_FILE
    console_level = (console_level or DEFAULT_LOG_LEVEL).upper()
    file_level = (file_level or DEFAULT_LOG_FILE_LEVEL).upper()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root = logging.getLogger()

    # 幂等处理
    if root.handlers and not force:
        logging.warning("Root logger already configured. Skip setup_logging.")
        return

    if force:
        for h in root.handlers[:]:
            root.removeHandler(h)

    # 设置 root 级别（取最细粒度）
    root_level = min(
        getattr(logging, console_level, logging.INFO),
        getattr(logging, file_level, logging.DEBUG)
    )
    root.setLevel(root_level)

    # Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(PlainFormatter(
        fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(filename)s:%(lineno)d | %(funcName)-10s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    logging.info(f"✅ Logging initialized. Console: {console_level}, File: {file_level} → {log_path}")


# 🔹 保留你的 get_logger() 接口（关键！零改动迁移）
def get_logger(name: str = "TableLogger") -> logging.Logger:
    """
    获取 logger（兼容旧接口）
    - 不再创建 handler
    - 不再设 propagate=False
    - 直接返回标准 logger，自动继承 root 配置
    """
    logger = logging.getLogger(name)
    
    
    return logger


# 🔹 （可选）为了 debug，仍可暴露全局路径查询接口
def get_current_log_path() -> str:
    """获取当前生效的日志文件路径（从 root handler 中提取）"""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RotatingFileHandler):
            return handler.baseFilename
    return DEFAULT_LOG_FILE

def switch_log_file(
    new_log_path: str,
    console_level: str = None,
    file_level: str = None,
    max_bytes: int = MAX_LOG_SIZE,
    backup_count: int = BACKUP_COUNT,
) -> None:
    """
    动态切换日志文件路径（不影响 console handler）
    适用于 per-task/per-request 场景
    """
    console_level = (console_level or DEFAULT_LOG_LEVEL).upper()
    file_level = (file_level or DEFAULT_LOG_FILE_LEVEL).upper()

    # 确保目录存在
    os.makedirs(os.path.dirname(new_log_path), exist_ok=True)

    root = logging.getLogger()

    # 1️⃣ 移除所有 RotatingFileHandler（保留 StreamHandler 等）
    for handler in root.handlers[:]:
        if isinstance(handler, RotatingFileHandler):
            handler.close()  # 安全关闭文件
            root.removeHandler(handler)

    # 2️⃣ 创建新 file handler
    new_handler = RotatingFileHandler(
        new_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    new_handler.setLevel(file_level)
    new_handler.setFormatter(PlainFormatter(
        fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(filename)s:%(lineno)d | %(funcName)-10s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # 3️⃣ 添加到 root
    root.addHandler(new_handler)

    # 可选：同步更新 console level（如果需要）
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(console_level)

    logging.info(f"🔄 Switched log file to: {new_log_path}")

# 在 setup_logging() 之后调用：
def suppress_noisy_loggers():
    # 压制 HTTP 库（最常见）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # 其他可选压制
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)  # if used
    logging.getLogger("fsspec").setLevel(logging.WARNING)     # if used

# 为兼容旧测试，保留 log 实例（但推荐改用 getLogger(__name__)）
log = get_logger()


if __name__ == "__main__":
    # 示例：测试
    setup_logging(log_path="test_main_mode.log", console_level="DEBUG")
    logger = get_logger(__name__)
    logger.info("✅ 主应用模式测试：get_logger() 无需改动，自动继承 root 配置！")
    logger.success("SUCCESS works!")