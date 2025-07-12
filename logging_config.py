"""Logging configuration for VRP application
===========================================
Centralized logging configuration for the VRP application.
"""

import logging
import os
from datetime import datetime
from typing import Optional

# ログディレクトリの作成
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ログファイル名（日付付き）
LOG_FILE = os.path.join(LOG_DIR, f"vrp_app_{datetime.now().strftime('%Y%m%d')}.log")


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    ログ設定をセットアップする

    Args:
        level: ログレベル ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_file: ファイルにログを出力するかどうか
        log_to_console: コンソールにログを出力するかどうか
        max_file_size: ログファイルの最大サイズ（バイト）
        backup_count: 保持するログファイルの数

    Returns:
        設定されたロガー
    """
    # ルートロガーの設定
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # 既存のハンドラーをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # ログフォーマット
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # コンソールハンドラー
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ファイルハンドラー
    if log_to_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    指定された名前のロガーを取得する

    Args:
        name: ロガー名

    Returns:
        ロガーインスタンス
    """
    return logging.getLogger(name)


def log_solver_execution(logger: logging.Logger, solver_name: str, operation: str, details: Optional[dict] = None):
    """
    Solverの実行をログに記録する

    Args:
        logger: ロガーインスタンス
        solver_name: ソルバー名
        operation: 実行操作
        details: 詳細情報（オプション）
    """
    message = f"[{solver_name}] {operation}"
    if details:
        message += f" - {details}"
    logger.info(message)


def log_solution_info(logger: logging.Logger, solver_name: str, solution_info: dict):
    """
    ソリューション情報をログに記録する

    Args:
        logger: ロガーインスタンス
        solver_name: ソルバー名
        solution_info: ソリューション情報
    """
    logger.info(f"[{solver_name}] Solution found:")
    logger.info(f"  - Objective: {solution_info.get('objective', 'N/A')}")
    logger.info(f"  - Constraints: {solution_info.get('constraints', [])}")
    logger.info(f"  - Search strategy: {solution_info.get('search_strategy', 'N/A')}")
    logger.info(f"  - Time limit: {solution_info.get('time_limit_seconds', 'N/A')} seconds")
