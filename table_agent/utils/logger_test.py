import logging
import os
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_FILE = os.getenv("TABLEAGENT_LOG_FILE", "table_agent.log")
DEFAULT_LOG_LEVEL = os.getenv("DATAFLOW_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_FILE_LEVEL = os.getenv("DATAFLOW_LOG_FILE_LEVEL", "DEBUG").upper()
MAX_LOG_SIZE = 10 * 1024 * 1024
BACKUP_COUNT = 5
SUCCESS_LEVEL_NUM = 25
# ANSI 颜色码
COLOR_MAP = {
    "DEBUG": "\033[46m\033[30m",    # 青色底黑字
    "SUCCESS": "\033[32m",              # 绿色 INFO->SUCESS
    "WARNING": "\033[43m\033[30m",   # 黄色底黑字
    "ERROR": "\033[31m",             # 红色
    "CRITICAL": "\033[41m\033[37m",  # 红底白字
    "RESET": "\033[0m",
}

# 字段颜色
FIELD_COLORS = {
    "time": "\033[90m",      # 灰色
    "name": "\033[35m",      # 紫色/洋红
    "location": "\033[96m",  # 亮青色
}

logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)

logging.Logger.success = success

_GLOBAL_LOG_PATH = None

def set_global_log_path(log_path: str):
    global _GLOBAL_LOG_PATH
    _GLOBAL_LOG_PATH = log_path



class ColorFormatter(logging.Formatter):
    """
    支持不同字段高亮显示的 Formatter,仅限控制台输出。
    """
    def format(self, record):
        level_name = record.levelname
        level_color = COLOR_MAP.get(level_name, "")
        reset = COLOR_MAP["RESET"]
        
        # 格式化各个字段-
        asctime = self.formatTime(record, self.datefmt)
        levelname = record.levelname
        name = record.name
        filename = record.filename
        lineno = record.lineno
        message = record.getMessage()
        
        # 组合带颜色的输出 - 每个字段不同颜色
        formatted = (
            f"{FIELD_COLORS['time']}{asctime}{reset} | "
            f"{level_color}{levelname:<8}{reset} | "
            f"{FIELD_COLORS['name']}{name}{reset} | "
            f"{FIELD_COLORS['location']}{filename}:{lineno}{reset} | "
            f"{level_color}{record.funcName:<5}{reset} | "
            f"{level_color}{message}{reset}"  # 消息使用级别颜色
        )
        
        return formatted

def _create_handler():
    """创建控制台和文件的日志处理器。"""
    global _GLOBAL_LOG_PATH
    # 控制台输出（带颜色）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(DEFAULT_LOG_LEVEL)
    color_formatter = ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler.setFormatter(color_formatter)

    if _GLOBAL_LOG_PATH is None:
        _GLOBAL_LOG_PATH = DEFAULT_LOG_FILE
    file_handler = RotatingFileHandler(_GLOBAL_LOG_PATH, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setLevel(DEFAULT_LOG_FILE_LEVEL)
    plain_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(filename)s:%(lineno)d |%(funcName)-5s| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(plain_formatter)
    return [stream_handler, file_handler]

def get_logger(name: str = "TableLogger") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handlers = _create_handler()
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(DEFAULT_LOG_FILE_LEVEL)
        logger.propagate = False
    if _GLOBAL_LOG_PATH is None:
        logger.warning("未配置全局日志路径(_GLOBAL_LOG_PATH)。使用默认日志文件路径: %s", DEFAULT_LOG_FILE)
    if not os.path.exists(_GLOBAL_LOG_PATH):
        os.makedirs(os.path.dirname(_GLOBAL_LOG_PATH), exist_ok=True)

    return logger

log = get_logger()

if __name__ == "__main__":
    log.info("Logger 初始化成功")
    log.debug("This is a debug message.")
    log.warning("This is a warning.")
    log.error("This is an error.")
    log.critical("This is CRITICAL!")
    log.success("This is a SUCCESS message.")   