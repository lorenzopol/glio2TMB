"""This modules implements a container type for a column value in a MafRecord.

* MafColumnRecord        generic container for storing key and value pairs for
                         a given column in a MafRecord.
* MafCustomColumnRecord  a MafColumnRecord to simplify the creation of
                         sub-classes that wish to constrain both the type and
                         value of the column.
"""
import abc
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

from .validation import MafValidationError, MafValidationErrorType

if TYPE_CHECKING:
    from .schemes import MafScheme


class MafColumnRecord:
    """
    A generic container for storing key and value pairs for a given column in a
    MafRecord.  Provides methods to validate the value of the column,
    determine if the value is equal to a null value.

    Sub-classes should override the
    :func:``~maflib.column.MafColumnRecord.__nullable_dict__`` method to give
    the values that should be treated as null.
    """

    def __init__(
        self, key: str, value: Any, column_index: int = None, description: str = None
    ):
        """
        :param key: the name of the column
        :param value: the value of the column
        :param column_index: optionally the zero-based index of the column
        :param description: optionally any description about the column
        """
        self.key = key
        self.value = value
        self.column_index = column_index
        self.description = description
        self.validation_errors: List[Optional[MafValidationError]] = list()

        # check that all nullable keys are strings
        if self.is_nullable():
            for key in self.__nullable_keys__():
                if not isinstance(key, str):
                    raise ValueError(
                        "Nullable key '%s' was not a 'str' but"
                        " instead '%s' (%s)"
                        % (str(key), key.__class__.__name__, self.__class__.__name__)
                    )

    def validate(
        self,
        reset_errors: bool = True,
        scheme: Optional['MafScheme'] = None,
        line_number: int = None,
    ) -> List[Optional[MafValidationError]]:
        """
        Validates that the value is of the correct type and an acceptable
        value.
        :return: a list of validation errors found, if any.
        """

        if reset_errors:
            self.validation_errors = list()

        if scheme:

            def add_errors(error: MafValidationError) -> None:
                """Adds an error"""
                self.validation_errors.append(error)

            scheme_column_index: Optional[int] = scheme.column_index(name=self.key)
            scheme_column_class: Optional[MafColumnRecord] = scheme.column_class(
                name=self.key
            )

            if scheme_column_index is None:
                add_errors(
                    MafValidationError(
                        MafValidationErrorType.SCHEME_MISMATCHING_COLUMN_NAMES,
                        "No column '%s' present in the scheme '%s'"
                        % (self.key, scheme.version()),
                        line_number=line_number,
                    )
                )
            elif (
                self.column_index is not None
                and scheme_column_index != self.column_index
            ):
                add_errors(
                    MafValidationError(
                        MafValidationErrorType.RECORD_COLUMN_OUT_OF_ORDER,
                        "Column with name '%s' was found in the %dth column"
                        ", but expected the %dth column with scheme "
                        "'%s''"
                        % (
                            self.key,
                            self.column_index,
                            scheme_column_index,
                            scheme.version(),
                        ),
                        line_number=line_number,
                    )
                )
            elif not isinstance(self, scheme_column_class):  # type: ignore
                add_errors(
                    MafValidationError(
                        MafValidationErrorType.RECORD_COLUMN_WRONG_FORMAT,
                        "Column with name '%s' is in the wrong format. "
                        "Found '%s' expected '%s'"
                        % (self.key, str(self.__class__), str(scheme_column_class)),
                        line_number=line_number,
                    )
                )

        return self.validation_errors

    def is_null(self) -> bool:
        """
        :return: ``True`` if the value is a "null" value, ``False`` otherwise
        """
        values = self.__nullable_values__()
        if values is None:
            return False
        else:
            return self.value in values

    @classmethod
    def build(
        cls,
        name: str,
        value: Any,
        column_index: Optional[int] = None,
        description: Optional[str] = None,
        scheme: Optional['MafScheme'] = None,
    ) -> 'MafColumnRecord':
        """
        If ``scheme`` is given, then the the appropriate column type will be
        built by matching the provided name with the column name in the
        scheme.  Otherwise, a column of type ``MafColumnRecord`` will be
        returned.
        :return: builds a ``MafColumnRecords`` from the given string.  Raises a
        ``ValueError`` if there was a formatting error.
        """
        if scheme:
            scheme_column_index = scheme.column_index(name=name)
            if scheme_column_index is None:
                raise KeyError(
                    "Column with name '%s' not found in scheme '%s'"
                    % (name, str(scheme))
                )
            elif column_index is not None and column_index is not scheme_column_index:
                raise ValueError(
                    "Mismatch column index: found '%s', expected '%s'"
                    % (str(column_index), str(scheme_column_index))
                )
            # NB: do not pass the scheme!
            return scheme.column_class(name=name).build(  # type: ignore
                name=name,
                value=value,
                column_index=scheme_column_index,
                description=description,
            )
        else:
            return MafColumnRecord(
                key=name,
                value=value,
                column_index=column_index,
                description=description,
            )

    @classmethod
    def is_nullable(cls) -> bool:
        """
        :return: ``True`` if this column has a possible "null" value, ``False``
        otherwise.
        """
        return bool(cls.__nullable_values__())

    @classmethod
    def __nullable_dict__(cls) -> Optional[Dict[str, Any]]:
        """
        :return: a map from the string representation of nullable values to
        the actual nullable value.  For example, an empty string may map to
        ``None``.  ``None`` should be returned if no nullable values exist.
        """
        return None

    @classmethod
    def __nullable_values__(cls) -> List[str]:
        """
        This method should not be overridden by sub-classes.
        :return: a list of values that should be treated as "null", ``None``
        otherwise.
        """
        if cls.__nullable_dict__() is not None:
            return list(cls.__nullable_dict__().values())  # type: ignore
        else:
            return []

    @classmethod
    def __nullable_keys__(cls) -> List[str]:
        """
        This method should not be overridden by sub-classes.
        :return: a list of values that should be treated as "null", ``None``
        otherwise.
        """
        if cls.__nullable_dict__() is not None:
            return list(cls.__nullable_dict__().keys())  # type: ignore
        else:
            return []

    def __str__(self) -> str:
        """Delegates the conversion to a string for non-null values to
        __string_it__()"""

        # check to see if the value is a "nullable value"
        nullable_dict = self.__nullable_dict__()
        if nullable_dict is not None:
            possible_keys = [
                key for key, value in nullable_dict.items() if value == self.value
            ]

            # FIXME: Too-clever solution for grabbing first item of list without IndexError
            key = next(iter(possible_keys), None)

            # did we find a key for the given null value?
            if key is not None:
                # always prefer the empty string
                if "" in possible_keys:
                    return ""
                return key
        return self.__string_it__()

    def __string_it__(self) -> str:
        """Sub-classes can override this method to print a string when the
        value is not null"""
        return str(self.value)


class MafCustomColumnRecord(MafColumnRecord):
    """
    A MafColumnRecord to simplify the creation of sub-classes that wish to
    constrain both the type and value of the column.

    Sub-classes should implement the ``__build__`` and ``__validate__``
    methods.
    """

    __metaclass__ = abc.ABCMeta

    @classmethod
    def __build__(cls, value: str) -> Union[list, Union[float, int, str], UUID]:
        """
        Builds the column's value from the given string.  Raises a
        ``ValueError`` if there was a formatting error.  Any logic about
        converting the value or type should be done here.
        """

    @classmethod
    def build_nullable(
        cls,
        name: str,
        column_index: Optional[int] = None,
        description: Optional[str] = None,
        scheme: Optional['MafScheme'] = None,
    ) -> Union['MafCustomColumnRecord', MafColumnRecord]:
        """
        This method should not be overridden by sub-classes.

        The class should have at least one nullable key and value, from which
        the column is built.
        """
        if not cls.is_nullable():
            raise ValueError(
                "Column name '%s' is not nullable, "
                "but build_nullable was called ('%s')" % (name, cls.__name__)
            )
        key = cls.__nullable_keys__()[0]  # type: ignore
        return cls.build(
            name=name,
            value=key,
            column_index=column_index,
            description=description,
            scheme=scheme,
        )

    @classmethod
    def build(
        cls,
        name: str,
        value: Any,
        column_index: Optional[int] = None,
        description: Optional[str] = None,
        scheme: Optional['MafScheme'] = None,
    ) -> Union['MafColumnRecord', MafColumnRecord]:
        """
        This method should not be overridden by sub-classes.

        Builds the column's value from the given string.  Raises a
        ``ValueError`` if there was a formatting error.  The passed value is
        first checked to see if it is in the ``__nullable_dict__`` dictionary,
        and if so, the value in the dictionary is returned.  Otherwise,
        the ``__build__`` method is called.
        """
        if scheme:
            return super(MafCustomColumnRecord, cls).build(
                name=name,
                value=value,
                column_index=column_index,
                description=description,
                scheme=scheme,
            )
        nullable_dict = cls.__nullable_dict__()
        if nullable_dict is not None and value in nullable_dict:
            built_value = nullable_dict[value]
        else:
            built_value = cls.__build__(value=value)
        return cls(
            key=name,
            value=built_value,
            column_index=column_index,
            description=description,
        )

    def __validate__(self) -> Optional[Any]:
        """
        A sub-class should implement this to perform any custom validation on
        the type and value of the value returned by ``__build__``.
        :return: None if the column's value is valid, otherwise a
        string message to return the user.
        """
        return None

    def validate(
        self,
        reset_errors: bool = True,
        scheme: Optional['MafScheme'] = None,
        line_number: Optional[int] = None,
    ) -> List[Optional[MafValidationError]]:
        """
        This method should not be overridden by sub-classes.

        Checks to see if the value is one of the nullable values.  If not,
        calls ``__validate__``.  If no message was returned, calls ``validate``
        on the super-class.
        :return: a list of validation errors, if any.
        """
        if reset_errors:
            self.validation_errors = list()
        nullable_values = self.__nullable_values__()
        if nullable_values is not None and self.value in nullable_values:
            msg = None
        else:
            msg = self.__validate__()
        if msg is not None:
            error = MafValidationError(
                MafValidationErrorType.RECORD_COLUMN_WRONG_FORMAT,
                "%s in column with name '%s'" % (msg, self.key),
                line_number=line_number,
            )
            self.validation_errors.append(error)
        return super(MafCustomColumnRecord, self).validate(
            reset_errors=False,  # we reset above!
            scheme=scheme,
            line_number=line_number,
        )
