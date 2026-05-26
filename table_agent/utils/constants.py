BENCHMARK_TASK_TYPES = [
    # Table Cleaning
    "TableCleaning-ErrorDetectionANDCorrection",
    "TableCleaning-ColumnTypeAnnotation",
    "TableCleaning-DataImputation",
    "TableCleaning-Deduplication",

    # Table Transformation
    "TableTransformation-RowToRowTransform",
    "TableTransformation-SplittingANDConcatenation",
    "TableTransformation-RowColumnSwapping",
    "TableTransformation-Filtering",
    "TableTransformation-Grouping",
    "TableTransformation-Sorting",
    "TableTransformation-ListExtraction",

    # Table Augmentation
    "TableAugmentation-RowPopulation",
    "TableAugmentation-SchemaAugmentation",
    "TableAugmentation-ColumnAugmentation",  # 单独列出以强调列级扩充

    # Table Matching
    "TableMatching-SchemaMatching",
    "TableMatching-EntityMatching"
]

MODEL_RATES = {
    "gpt-5.2":{
        "模型倍率": 3.5,
        "补全倍率": 8,
        "分组倍率": 1 
    },
    "gpt-5": {
        "模型倍率": 2.5,
        "补全倍率": 8,
        "分组倍率": 1
    },
    "gpt-4o": {
        "模型倍率": 1.25,
        "补全倍率": 4,
        "分组倍率": 1
    },
    "claude-4-opus": {
        "模型倍率": 30,
        "补全倍率": 5,
        "分组倍率": 1
    },
    "claude-4-sonnet": {
        "模型倍率": 6,
        "补全倍率": 5,
        "分组倍率": 1
    },
    "gemini-2.5-pro": {
        "模型倍率": 1.5,
        "补全倍率": 8,
        "分组倍率": 1
    },
    "qwen-max-latest": {
        "模型倍率": 1.45,
        "补全倍率": 4,
        "分组倍率": 1
    },
    "deepseek-v3": {
        "模型倍率": 0.66,
        "补全倍率": 4,
        "分组倍率": 1
    },
    "grok-4": {
        "模型倍率": 4,
        "补全倍率": 10,
        "分组倍率": 1
    }
}
