# -*- coding: utf-8 -*-
"""
MCM 2025 Problem C - Configuration File
配置文件：定义项目路径和数据文件路径
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "2025_Problem_C_Data"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 数据文件路径
ATHLETES_FILE = DATA_RAW_DIR / "summerOly_athletes.csv"
HOSTS_FILE = DATA_RAW_DIR / "summerOly_hosts.csv"
MEDALS_FILE = DATA_RAW_DIR / "summerOly_medal_counts.csv"
PROGRAMS_FILE = DATA_RAW_DIR / "summerOly_programs.csv"
DATA_DICT_FILE = DATA_RAW_DIR / "data_dictionary.csv"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

# 其他配置
RANDOM_SEED = 42
