"""
Stores the column types for the values parsed from a MAF file, for example
columns that have specific types of values, nullable values, along with any
custom validation to ensure proper types and values.
"""


import abc
import inspect
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID

from .column import MafColumnRecord, MafCustomColumnRecord
from .column_values import (
    FeatureTypeEnum,
    GdcValidationStatusEnum,
    ImpactEnum,
    MC3OverlapEnum,
    MutationStatusEnum,
    NullableYesOrNoEnum,
    NullableYOrNEnum,
    PickEnum,
    SequencerEnum,
    StrandEnum,
    ValidationStatusEnum,
    VariantClassificationEnum,
    VariantSupportEnum,
    VariantTypeEnum,
    VerificationStatusEnum,
    YesNoOrUnknownEnum,
)


def get_column_types() -> List[Tuple[str, Any]]:
    """Gets all the column types defined in maflib"""

    def predicate(obj: object) -> bool:
        """A predicate to get all classes that are subclasses of
        MafColumnRecord"""
        return inspect.isclass(obj) and issubclass(obj, MafColumnRecord)  # type: ignore

    # Get all available column types
    return inspect.getmembers(sys.modules["maflib.column_types"], predicate)


class NullableEmptyStringIsNone:
    """Mix this in to a MafCustomColumnRecord to make the only nullable value
    be an empty string that is treated as None
    """

    @classmethod
    def __nullable_dict__(cls) -> Optional[Dict[str, Optional[str]]]:
        return {"": None}


class NullableEmptyStringIsEmptyList:
    """Mix this in to a MafCustomColumnRecord to make the only nullable value
    be an empty string that is treated as an empty list
    """

    @classmethod
    def __nullable_dict__(cls) -> Optional[Dict[str, Any]]:
        return {"": []}


class RequireNullValue(MafColumnRecord):
    """Mix this in to a MafCustomColumnRecord so that it only validates if the
    value is a null
    """

    def __validate__(self) -> str:
        return f"'{self.value}' was not a null value"


class _BuildStringColumn(MafCustomColumnRecord):
    """Mix this in to require a string as the value"""

    @classmethod
    def __build__(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(
                "'%s' was not a string (was %s)" % (str(value), str(value.__class__))
            )
        return str(value)


class NullableStringColumn(
    NullableEmptyStringIsNone,
    _BuildStringColumn,
):
    """A column where the value must be a string, with its null value being
    an empty string"""

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, str):
            return f"'{self.value}' was not a string"
        else:
            return None


class StringColumn(NullableStringColumn):
    """A column where the value must be a non-empty string"""

    EmptyStringMessage = "Empty string is not allowed"

    @classmethod
    def __nullable_dict__(cls) -> None:
        return None

    def __validate__(self) -> Optional[str]:
        msg = super(StringColumn, self).__validate__()
        if msg is None and not self.value:
            return self.EmptyStringMessage
        else:
            return msg


class StringIntegerOrFloatColumn(MafCustomColumnRecord):
    """A column where the value is either a string or float"""

    @classmethod
    def __type_error_message__(cls, value: Any) -> str:
        return f"'{value}' of type '{value.__class__.__name__}' was neither an int, float, nor string"

    @classmethod
    def __build__(cls, value: str) -> Union[float, int, str]:
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return int(value)
        except ValueError:
            pass
        return str(value)

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, (int, float, str)):
            return self.__type_error_message__(self.value)
        else:
            return None


class StringOrIntegerColumn(MafCustomColumnRecord):
    """A column that is a string or integer"""

    @classmethod
    def __type_error_message__(cls, value: Any) -> str:
        return f"'{value}' of type '{value.__class__.__name__}' was neither an int nor string"

    @classmethod
    def __build__(cls, value: Any) -> Union[str, int]:
        try:
            return int(value)
        except ValueError:
            pass
        return str(value)

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, (int, str)):
            return self.__type_error_message__(self.value)
        else:
            return None


class IntegerColumn(MafCustomColumnRecord):
    """A column that is an integer"""

    @classmethod
    def __type_error_message__(cls, value: Any) -> str:
        return f"'{value}' of type '{type(value)}' was not an int"

    @classmethod
    def __min_value__(cls) -> Optional[int]:
        return None

    @classmethod
    def __max_value__(cls) -> Optional[int]:
        return None

    @classmethod
    def __build__(cls, value: Any) -> int:
        return int(value)

    def __validate__(self) -> Optional[str]:
        min_value = self.__min_value__()
        max_value = self.__max_value__()
        if not isinstance(self.value, int):
            return self.__type_error_message__(self.value)
        elif min_value is not None and self.value < min_value:
            return "'%d' was out of range (<%s)" % (self.value, str(min_value))
        elif max_value is not None and max_value < self.value:
            return "'%d' was out of range (>%s)" % (self.value, str(max_value))
        else:
            return None


class NullableIntegerColumn(NullableEmptyStringIsNone, IntegerColumn):
    """A column that is either an integer, or an empty string treated as the
    null value"""

    pass


class ZeroBasedIntegerColumn(IntegerColumn):
    """A column that represents a zero-based integer"""

    @classmethod
    def __min_value__(cls) -> int:
        return 0


class OneBasedIntegerColumn(IntegerColumn):
    """A column that represents a zero-based integer"""

    @classmethod
    def __min_value__(cls) -> int:
        return 1


class NullableZeroBasedIntegerColumn(NullableEmptyStringIsNone, IntegerColumn):
    """A column that represents a zero-based integer, or an empty string
    treated as the null value"""

    @classmethod
    def __min_value__(cls) -> int:
        return 0


class NullableOneBasedIntegerColumn(NullableEmptyStringIsNone, IntegerColumn):
    """A column that represents a one-based integer, or an empty string
    treated as the null value"""

    @classmethod
    def __min_value__(cls) -> int:
        return 1


class FloatColumn(MafCustomColumnRecord):
    """A column that represents a floating point number"""

    @classmethod
    def __build__(cls, value: Any) -> float:
        return float(value)

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, float):
            return f"'{self.value}' was not a float"
        else:
            return None


class NullableFloatColumn(NullableEmptyStringIsNone, FloatColumn):
    """A column that is either a floating point number, or an empty string
    treated as the null value"""

    pass


class EnumColumn(MafCustomColumnRecord):
    """An abstract class whose value is an enumeration value."""

    __metaclass__ = abc.ABCMeta

    @classmethod
    @abc.abstractmethod
    def __enum_class__(cls) -> Type[Enum]:
        """
        :return: the class that extends `Enum`.
        """

    # FIXME: Return types depend on specific enum
    @classmethod
    def __build__(cls, value: Any) -> Any:
        enum_cls = cls.__enum_class__()
        try:
            return enum_cls(value)
        except ValueError:
            return enum_cls[value]

    def __validate__(self) -> Optional[str]:
        enum_cls = self.__enum_class__()
        if not isinstance(self.value, enum_cls):
            return f"'{self.value}' was not of type '{enum_cls.__name__}'"
        else:
            return None

    def __string_it__(self) -> str:
        return str(self.value.value)


class SequenceOfValuesColumn(MafCustomColumnRecord):
    """An abstract class whose value is a list of zero or more values defined by
    an existing column class"""

    @classmethod
    @abc.abstractmethod
    def __column_class__(cls) -> Type[MafCustomColumnRecord]:
        """
        :return: the class of the column to use
        """

    @classmethod
    def __build__(cls, value: Any) -> list:
        column_cls = cls.__column_class__()
        if not isinstance(value, str):
            raise ValueError(
                "'%s' was not a string (was %s)" % (str(value), str(value.__class__))
            )
        values = value.split(";")
        return [column_cls.__build__(val) for val in values]

    def __validate__(self) -> Optional[str]:
        column_cls = self.__column_class__()
        if not isinstance(self.value, list) and not isinstance(self.value, tuple):
            return f"'{self.value}' was neither a list nor tuple"
        for i, value in enumerate(self.value):
            column = column_cls("", value)  # create a new column to validate
            msg = column.__validate__()
            if msg:
                return "For the %dth value in '%s': %s" % (i + 1, str(self.value), msg)
        return None

    def __string_it__(self) -> str:
        return ";".join([str(v) for v in self.value])


class SequenceOfStrings(NullableEmptyStringIsEmptyList, SequenceOfValuesColumn):
    """A column that represents a sequence of zero or more strings"""

    @classmethod
    def __column_class__(cls) -> Type[StringColumn]:
        """
        :return: the class of the column to use
        """
        return StringColumn


class SequenceOfIntegers(NullableEmptyStringIsEmptyList, SequenceOfValuesColumn):
    """A column that represents a sequence of zero or more integers"""

    @classmethod
    def __column_class__(cls) -> Type[IntegerColumn]:
        """
        :return: the class of the column to use
        """
        return IntegerColumn


class NullableDnaString(_BuildStringColumn, MafCustomColumnRecord):
    """A column that represents a string of DNA bases, or an empty string
    treated as the null value"""

    @classmethod
    def __nullable_dict__(cls) -> Optional[Dict[str, None]]:
        return {"": None}

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, str):
            return f"'{self.value}' was not a string"
        elif "-" == self.value:
            return None
        for i, base in enumerate(self.value):
            if base not in ('A', 'C', 'G', 'T'):
                return "The %dth base in '%s' was not in [ACGT]" % (i + 1, self.value)
        return None


class DnaString(NullableDnaString):
    """A column that represents a non-empty string of DNA bases"""

    @classmethod
    def __nullable_dict__(cls) -> Optional[dict]:
        return None

    def __validate__(self) -> Optional[str]:
        msg = super(DnaString, self).__validate__()
        if msg is None and not self.value:
            return "Found an empty string"
        else:
            return msg


class Canonical(MafCustomColumnRecord):
    """A column representing the 'canonical' MAF column.  The value is a
    boolean: True if the input string was "Yes" (case insensitive), or False
    if the empty string."""

    @classmethod
    def __build__(cls, value: str) -> bool:
        if not isinstance(value, str):
            raise ValueError("'%s' was not a string" % str(value))
        value = value.upper()
        if value != "" and value != "YES":
            raise ValueError("Value must be '' or 'YES'")
        else:
            return value == "YES"

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, bool):
            return f"'{self.value}' was not a bool"
        else:
            return None

    def __string_it__(self) -> str:
        return "YES" if self.value else ""


class NullableYesOrNo(EnumColumn):
    """A column that represents the string "Yes" or "No", with the empty
    string treated as a null value."""

    @classmethod
    def __enum_class__(cls) -> Type[NullableYesOrNoEnum]:
        return NullableYesOrNoEnum

    @classmethod
    def __build__(cls, value: Any):  # type: ignore
        if not isinstance(value, str):
            raise ValueError(f"'{value}' was not a string")
        return super(NullableYesOrNo, cls).__build__(value.capitalize())

    @classmethod
    def __nullable_dict__(cls) -> dict:
        return {
            NullableYesOrNoEnum.Null.name: NullableYesOrNoEnum.Null,
            "": NullableYesOrNoEnum.Null,
        }


class NullableYOrN(EnumColumn):
    """A column that represents the string "Y" or "N", with the empty
    string treated as a null value."""

    @classmethod
    def __enum_class__(cls) -> Type[NullableYOrNEnum]:
        return NullableYOrNEnum

    @classmethod
    def __build__(cls, value: Any):  # type: ignore
        if not isinstance(value, str):
            raise ValueError("'%s' was not a string" % str(value))
        return super(NullableYOrN, cls).__build__(value.capitalize())

    @classmethod
    def __nullable_dict__(cls) -> dict:
        return {
            NullableYOrNEnum.Null.name: NullableYOrNEnum.Null,
            "": NullableYOrNEnum.Null,
        }


class YesNoOrUnknown(EnumColumn):
    """A column that represents the string "Yes", "No", or "Unknown"."""

    @classmethod
    def __enum_class__(cls) -> Type[YesNoOrUnknownEnum]:
        return YesNoOrUnknownEnum

    @classmethod
    def __build__(cls, value):  # type: ignore
        if not isinstance(value, str):
            raise ValueError("'%s' was not a string" % str(value))
        return super(YesNoOrUnknown, cls).__build__(value)


class SequenceOfNullableYesOrNo(NullableEmptyStringIsEmptyList, SequenceOfValuesColumn):
    """A column that represents a sequence of nullable yes or no."""

    @classmethod
    def __column_class__(cls) -> Type[NullableYesOrNo]:
        """
        :return: the class of the column to use
        """
        return NullableYesOrNo


class PickColumn(EnumColumn):
    """A column that represents the 'Pick' MAF column, with possible values
    "1", or the empty string, which is treated as None"""

    @classmethod
    def __enum_class__(cls) -> Type[PickEnum]:
        return PickEnum

    @classmethod
    def __build__(cls, value: Any):  # type: ignore
        if not isinstance(value, str):
            raise ValueError("'%s' was not a string" % str(value))
        return super(PickColumn, cls).__build__(value.capitalize())

    @classmethod
    def __nullable_dict__(cls) -> dict:
        return {
            PickEnum.Null.name: PickEnum.Null,
            "": PickEnum.Null,
        }


class BooleanColumn(MafCustomColumnRecord):
    """A column that represents a boolean value, with input values either
    "True" or "False" (case insensitive)."""

    @classmethod
    def __build__(cls, value: str) -> bool:
        if not isinstance(value, str):
            raise ValueError(f"'{value}' was not a string")
        value = value.upper()
        if value not in ["TRUE", "FALSE"]:
            raise ValueError("'%s' was not either 'True' or 'False'" % str(value))
        return value == 'TRUE'

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, bool):
            return "'%s' was not a bool" % str(self.value)
        else:
            return None


class EntrezGeneId(ZeroBasedIntegerColumn):
    """A column that represents the MAF Entrez_Gene_Id column as an integer,
    where zero is treated as null."""

    @classmethod
    def __nullable_dict__(cls) -> Dict[str, None]:
        return {"0": None}


class Strand(EnumColumn):
    """A column that represents the 'Strand' MAF column, where strand is either
    '+' or '-' (positive strand or negative strand)"""

    @classmethod
    def __enum_class__(cls) -> Type[StrandEnum]:
        return StrandEnum


class VariantClassification(EnumColumn):
    """A column that represents the 'Variant_Classification' MAF column."""

    @classmethod
    def __enum_class__(cls) -> Type[VariantClassificationEnum]:
        return VariantClassificationEnum


class VariantType(EnumColumn):
    """A column that represents the 'Variant_Type' MAF column."""

    @classmethod
    def __enum_class__(cls) -> Type[VariantTypeEnum]:
        return VariantTypeEnum


class VariantSupport(EnumColumn):
    """A column that represents the RNA_Support MAF column"""

    @classmethod
    def __enum_class__(cls) -> Type[VariantSupportEnum]:
        return VariantSupportEnum


class VerificationStatus(NullableEmptyStringIsNone, EnumColumn):
    """A column that represents the 'Verification_Status' MAF column,
    where the empty string is treated as null."""

    @classmethod
    def __enum_class__(cls) -> Type[VerificationStatusEnum]:
        return VerificationStatusEnum


class ValidationStatus(NullableEmptyStringIsNone, EnumColumn):
    """A column that represents the 'Validation_Status' MAF column,
    where the empty string is treated as null."""

    @classmethod
    def __enum_class__(cls) -> Type[ValidationStatusEnum]:
        return ValidationStatusEnum


class MutationStatus(EnumColumn):
    """A column that represents the 'Mutation_Status' MAF column"""

    @classmethod
    def __enum_class__(cls) -> Type[MutationStatusEnum]:
        return MutationStatusEnum


class Sequencer(EnumColumn):
    """A column that represents the a single value in the 'Sequencer' MAF
    column"""

    @classmethod
    def __enum_class__(cls) -> Type[SequencerEnum]:
        return SequencerEnum


class SequenceOfSequencers(NullableEmptyStringIsEmptyList, SequenceOfValuesColumn):
    """A column that represents the 'Sequencer' MAF column"""

    @classmethod
    def __column_class__(cls) -> Type[Sequencer]:
        """
        :return: the class of the column to use
        """
        return Sequencer


class UUIDColumn(MafCustomColumnRecord):
    """A column that represents a UUID"""

    @classmethod
    def __build__(cls, value: Any) -> UUID:
        return UUID(value)

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, UUID):
            return "'{self.value}' was not a UUID"
        else:
            return None


class NullableUUIDColumn(NullableEmptyStringIsNone, UUIDColumn):
    """A column that represents a UUID.  An empty string is allowed if no
    UUID is given."""

    pass


class FeatureType(NullableEmptyStringIsNone, EnumColumn):
    """A column that represents the 'Feature_Type' MAF column, with the empty
    string treated as a null value."""

    @classmethod
    def __enum_class__(cls) -> Type[FeatureTypeEnum]:
        return FeatureTypeEnum


class TranscriptStrand(NullableEmptyStringIsNone, MafCustomColumnRecord):
    """A column that represents the 'Transcript_Strand' MAF column as an
    integer, either -1 or 1"""

    @classmethod
    def __build__(cls, value: Any) -> int:
        return int(value)

    def __validate__(self) -> Optional[str]:
        if not isinstance(self.value, int):
            return "'{self.value}' was not an integer"
        elif self.value not in (-1, 1):
            return "'{self.value}' was neither -1 nor 1"
        else:
            return None


class Impact(EnumColumn):
    """A column that represents the 'Impact' MAF column"""

    @classmethod
    def __enum_class__(cls) -> Type[ImpactEnum]:
        return ImpactEnum


class MC3Overlap(EnumColumn):
    """A column that represents the 'MC3_Overlap' MAF column"""

    @classmethod
    def __enum_class__(cls) -> Type[MC3OverlapEnum]:
        return MC3OverlapEnum


class GdcValidationStatus(EnumColumn):
    """A column that represents the 'GDC_Validation_Status' MAF column"""

    @classmethod
    def __enum_class__(cls) -> Type[GdcValidationStatusEnum]:
        return GdcValidationStatusEnum
