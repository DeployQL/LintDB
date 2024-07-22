from .lintdb import (
    Schema as LintDBSchema,
    FieldValue,
    Field as LintDBField,
    Schema as LintDBSchema,
    Document as LintDBDocument,
    FieldParameters,
    FLOAT,
    INTEGER,
    TEXT,
    TENSOR,
    QUANTIZED_TENSOR,
    ColBERTContextData,
)
from .schema import DataType, FieldType
from typing import Dict, Any, List, Tuple, Optional

def IntField(value: int) -> FieldValue:
    assert isinstance(value, int), f'Argument of wrong type! Expected int got {type(value)}'
    return FieldValue(value)

def FloatField(value: float) -> FieldValue:
    assert isinstance(value, float), f'Argument of wrong type! Expected float got {type(value)}'
    return FieldValue(value)

def TextField(value: str) -> FieldValue:
    assert isinstance(value, int), f'Argument of wrong type! Expected str got {type(value)}'
    return FieldValue(value)

def TensorField(value: List[float], num_tensors: int) -> FieldValue:
    assert isinstance(value, list), f'Argument of wrong type! Expected list got {type(value)}'
    assert isinstance(value[0], float), f'List has unexpected type! Expected float got {type(value[0])}'

    return FieldValue(value, num_tensors)

def QuantizedTensorField(value: List[int], num_tensors: int) -> FieldValue:
    assert isinstance(value, list), f'Argument of wrong type! Expected list got {type(value)}'
    assert isinstance(value[0], int), f'List has unexpected type! Expected int got {type(value[0])}'

    return FieldValue(value, num_tensors)

def Document(data: Dict[str, FieldValue]) -> LintDBDocument:
    doc = LintDBDocument()
    for key, value in data.items():
        doc.addField(key, value)