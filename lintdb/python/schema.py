from .lintdb import (
    Schema as LintDBSchema,
    FieldValue,
    Field as LintDBField,
    Schema as LintDBSchema,
    FieldParameters as LintDBFieldParameters,
    FieldTypeArray,
    FLOAT,
    INTEGER,
    TEXT,
    TENSOR,
    QUANTIZED_TENSOR,
    COLBERT,
    ColBERTContextData,
    FieldType_Stored as LintDBStored,
    FieldType_Indexed as LintDBIndexed,
    FieldType_Colbert as LintDBColBERT,
    IndexedField,
    StoredField,
    ContextField,
    ColbertField,
    BINARIZER,
    PRODUCT_ENCODER,
    NONE,
)
from typing import Dict, Any
from enum import Enum

class DataType(Enum):
    FLOAT = FLOAT
    INTEGER = INTEGER
    TEXT = TEXT
    TENSOR = TENSOR
    QUANTIZED = QUANTIZED_TENSOR
    COLBERT = ColBERTContextData

class FieldType(Enum):
    Stored = LintDBStored
    Indexed = LintDBIndexed
    Colbert = LintDBColBERT

Quantizer_Binarizer = BINARIZER
Quantizer_ProductEncoder = PRODUCT_ENCODER
Quantizer_None = NONE

def Stored(name: str, data_type: DataType, params=LintDBFieldParameters()):
    if isinstance(params, dict):
        params = FieldParameters(**params)

    return StoredField(name, data_type.value, params)

def Context(name: str, data_type: DataType, params=LintDBFieldParameters()):
    if isinstance(params, dict):
        params = FieldParameters(**params)

    return ContextField(name, data_type.value, params)

def Indexed(name: str, data_type: DataType, params=LintDBFieldParameters()):
    if isinstance(params, dict):
        params = FieldParameters(**params)

    return IndexedField(name, data_type.value, params)

def Colbert(name: str, data_type: DataType, params=LintDBFieldParameters()):
    if isinstance(params, dict):
        params = FieldParameters(**params)

    dt = data_type.value
    field = ColbertField(name, dt, params)
    field.add_field_type(LintDBColBERT)
    return field

def FieldParameters(**kwargs):
    """
        size_t dimensions = 0;
    std::string analyzer = "";
    QuantizerType quantization = QuantizerType::UNKNOWN;
    size_t num_centroids = 0;
    size_t num_iterations = 10;
    size_t num_subquantizers = 0; // used for PQ quantizer
    size_t nbits = 1; // used for PQ quantizer
    :param kwargs:
    :return:
    """
    result = LintDBFieldParameters()
    for key, value in kwargs.items():
        setattr(result, key, value)

    return result

def Schema(fields: [LintDBField]):
    result = LintDBSchema()
    for field in fields:
        result.add_field(field)

    return result

