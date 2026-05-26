import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
from table_agent.utils.logger import get_logger



def simple_data_profile(df: pd.DataFrame) -> dict:
    """为单个 DataFrame 生成轻量级数据概要"""
    profile = {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "columns": {}
    }

    for col in df.columns:
        series = df[col]
        col_info = {
            "dtype": str(series.dtype),
            "non_null_count": int(series.count()),
            "null_count": int(series.isnull().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }

        # 数值列统计
        if pd.api.types.is_numeric_dtype(series):
            col_info["stats"] = {
                "mean": float(series.mean()) if not series.isna().all() else None,
                "std": float(series.std()) if not series.isna().all() else None,
                "min": float(series.min()) if not series.isna().all() else None,
                "max": float(series.max()) if not series.isna().all() else None,
                # "25%": float(series.quantile(0.25)) if not series.isna().all() else None,
                # "50%": float(series.median()) if not series.isna().all() else None,
                # "75%": float(series.quantile(0.75)) if not series.isna().all() else None,
            }
        # 类别/文本列：Top 10 频次
        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            top_values = series.value_counts().head(10).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_info

    return profile

def profile_multiple_csvs(csv_json_files: List[str], output_json: str = "data_profiles.json") -> Dict[str, Any]:
    all_profiles = {}
    logger = get_logger()
    for file_path in csv_json_files:
        path = Path(file_path)
        try:
            df = _read_file_as_dataframe(path)  # 读取json，csv，jsonl数据文件并转换为dataframe
            logger.debug(f'数据预览：{df.head()}')
            # 生成数据画像
            profile = simple_data_profile(df)
            profile["filename"] = path.name
            all_profiles[path.name] = profile

        except Exception as e:
            # 捕获所有错误，记录到 profile
            all_profiles[path.name] = {
                "error": str(e),
                "file_path": str(path)
            }

    # === 处理输出路径 ===
    output_path = Path(output_json)
    if output_path.name != "default_data_profiling.json":
        # 如果 output_json 是目录，则拼接文件名
        if output_path.is_dir():
            output_path = output_path / "default_data_profiling.json"
        else:
            # 否则视为完整文件路径
            pass
    else:
        # 默认情况：output_json 是文件名
        output_path = Path(output_json)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2, ensure_ascii=False)

    return all_profiles



def _read_file_as_dataframe(path: Path) -> pd.DataFrame:
    """智能读取 CSV / JSON / JSONL，并处理常见问题"""
    suffix = path.suffix.lower()
    
    try:
        if suffix == '.json':
            # 尝试直接读取
            df = pd.read_json(path)
            # 如果是字典（非列表），且包含 'data' 字段，尝试提取
            if isinstance(df, pd.Series) or (len(df) == 1 and isinstance(df.iloc[0], (dict, list))):
                # 重新加载原始 JSON
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and 'data' in raw:
                    df = pd.json_normalize(raw['data'])
                elif isinstance(raw, list):
                    df = pd.json_normalize(raw)
                else:
                    df = pd.json_normalize([raw]) if isinstance(raw, dict) else pd.DataFrame()
        elif suffix == '.jsonl':
            # 逐行读取并标准化（处理嵌套）
            records = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            df = pd.json_normalize(records)  # 自动展开嵌套字段
        elif suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")
        
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to read {path.name}: {e}")
        
    