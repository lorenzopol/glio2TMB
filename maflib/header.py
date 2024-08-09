"""This module implements a container data type for a Mutation Annotation File
(MAF) header.

* MafHeader                      container for MafHeaderRecords, each typically
                                 represented by a pragma line in the MAF
* MafHeaderRecord                container for one pragma line in the MAF, with
                                 a key and value
* MafHeaderVersionRecord         specialized container for storing the
                                 "version"
                                 pragma
* MafHeaderAnnotationSpecRecord  specialized container for storing the
                                 "annotation.spec" pragma.
* MafHeaderSortOrderRecord       specialized container for storing the
                                 "sort.order" pragma.
"""

import logging
from collections import OrderedDict
from collections.abc import MutableMapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

from .logger import Logger
from .scheme_factory import all_schemes, find_scheme
from .sort_order import Coordinate, SortOrder, Unknown, Unsorted
from .util import LineReader
from .validation import (
    MafValidationError,
    MafValidationErrorType,
    ValidationStringency,
)

if TYPE_CHECKING:
    from .reader import MafReader
    from .schemes import MafScheme

SortOrderType = Optional[SortOrder]


class MafHeaderRecord:
    """
    A header line for MAF files.
    """

    def __init__(self, key: str, value: Any):
        self._key = key
        self._value = value

    @property
    def key(self) -> str:
        """gets the key"""
        return self._key

    @key.setter
    def key(self, key: str) -> None:
        """sets the key"""
        self._key = key

    @property
    def value(self) -> Any:
        """gets the value"""
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        """sets the value"""
        self._value = value

    def __str__(self) -> str:
        """gets the text representation of this header record"""
        return f"{MafHeader.HeaderLineStartSymbol}{self._key} {self._value}"

    @classmethod
    def from_line(
        cls, line: str, line_number: Optional[int] = None
    ) -> Tuple[Optional['MafHeaderRecord'], Optional[MafValidationError]]:
        """Reads a single line in the MAF file header.

        If a formatting error is encountered, returns (error, None), otherwise
        returns (None, record).  Formatting errors include:
        - the line does not start with the correct symbol (i.e. #)
        - the line is missing a space separator for the key and value
        - the line has an empty key
        - the line has an empty value
        """
        error: Optional[MafValidationError] = None
        record: 'MafHeaderRecord' = None  # type: ignore
        if not line.startswith(MafHeader.HeaderLineStartSymbol):
            error = MafValidationError(
                MafValidationErrorType.HEADER_LINE_MISSING_START_SYMBOL,
                "Header line did not start with a '#'",
                line_number=line_number,
            )
        else:
            tokens = line[1:].split(" ", 1)
            if len(tokens) != 2:
                error = MafValidationError(
                    MafValidationErrorType.HEADER_LINE_MISSING_SEPARATOR,
                    "Header line did not have a key and value separated by a " "space",
                    line_number=line_number,
                )
            else:
                key, value = tokens
                value = value.rstrip()
                if not key:
                    error = MafValidationError(
                        MafValidationErrorType.HEADER_LINE_EMPTY_KEY,
                        "Header line had an empty key",
                        line_number=line_number,
                    )
                elif not value:
                    error = MafValidationError(
                        MafValidationErrorType.HEADER_LINE_EMPTY_VALUE,
                        "Header line had an empty value",
                        line_number=line_number,
                    )
                elif key == MafHeader.VersionKey:
                    record = MafHeaderVersionRecord(value=value)
                elif key == MafHeader.AnnotationSpecKey:
                    record = MafHeaderAnnotationSpecRecord(value=value)
                elif key == MafHeader.SortOrderKey:
                    try:
                        record = MafHeaderSortOrderRecord(value=value)
                    except Exception:
                        error = MafValidationError(
                            MafValidationErrorType.HEADER_UNSUPPORTED_SORT_ORDER,
                            "Sort order '%s' was not recognized" % value,
                            line_number=line_number,
                        )
                elif key == MafHeader.ContigKey:
                    record = MafHeaderContigRecord(value=value)
                else:
                    record = MafHeaderRecord(key=key, value=value)
        return record, error


class MafHeaderVersionRecord(MafHeaderRecord):
    """A marker MAF header record for storing the version"""

    def __init__(self, value: Any):
        super(MafHeaderVersionRecord, self).__init__(
            key=MafHeader.VersionKey, value=value
        )


class MafHeaderAnnotationSpecRecord(MafHeaderRecord):
    """A marker MAF header record for storing the annotation specification"""

    def __init__(self, value: Any):
        super(MafHeaderAnnotationSpecRecord, self).__init__(
            key=MafHeader.AnnotationSpecKey, value=value
        )


class MafHeaderSortOrderRecord(MafHeaderRecord):
    """A marker MAF header record for storing the sort order"""

    def __init__(
        self,
        value: Optional[Union[SortOrder, str]],
        fasta_index: str = None,
        contigs: list = None,
    ):
        """:param: value: a string representing the name of the sort order,
        or an instance of SortOrder."""
        # TODO: Implement class finder here
        if isinstance(value, str):
            value: SortOrder = next(  # type: ignore
                (so() for so in SortOrder.all() if so.name() == value),
                Unknown,
            )
        if not issubclass(type(value), SortOrder):
            # TODO: warn? log? return None? validation error?
            raise Exception(
                f"Value of type '{value.__class__.__name__}' is not a subclass of 'SortOrder'"
            )
        if (fasta_index or contigs) and issubclass(type(value), Coordinate):
            value: Coordinate = value.__class__(fasta_index=fasta_index, contigs=contigs)  # type: ignore

        super(MafHeaderSortOrderRecord, self).__init__(
            key=MafHeader.SortOrderKey, value=value
        )


class MafHeaderContigRecord(MafHeaderRecord):
    """A marker MAF header record for storing a list of contigs in the
    order for sorting"""

    def __init__(self, value: Any):
        """:param: value: a comma separate string or a list of
        chromosome names"""
        if isinstance(value, str):
            value = value.split(',')
        super(MafHeaderContigRecord, self).__init__(
            key=MafHeader.ContigKey, value=value
        )

    def __str__(self) -> str:
        """gets the text representation of this header record"""
        return f"{MafHeader.HeaderLineStartSymbol}{self._key} {','.join(self._value)}"


class MafHeader(MutableMapping):
    """
    A header for a MAF file storing zero or more header records.  Each record
    represents a single line from the original MAF file.

    Provides methods for accessing the records in the order they were added, as
    well as methods for returning the version
    (:func:`~header.MafHeader.version`), annotation specification
    (:func:`~header.MafHeader.annotation`),
    scheme (:func:`~header.MafHeader.scheme`), and sort order (
    :func:`~header.MafHeader.sort_order`).  Additionally, the
    :func:`~header.MafHeader.validate` method can be used
    to validate the format of the header as well as validate the contents
    relative to the given scheme. If no sort order is found in the header,
    the sort order will be "unsorted".
    """

    VersionKey = "version"

    AnnotationSpecKey = "annotation.spec"

    SortOrderKey = "sort.order"  # NOQA

    ContigKey = "contigs"

    SupportedVersions = [s.version() for s in all_schemes()]

    SupportedAnnotationSpecs = [s.annotation_spec() for s in all_schemes()]

    SupportedSortOrders = [so.name() for so in SortOrder.all()]

    HeaderLineStartSymbol = "#"

    def __init__(self, validation_stringency: ValidationStringency = None):
        self.validation_errors: List[MafValidationError] = []
        self.validation_stringency = (
            ValidationStringency.Silent
            if (validation_stringency is None)
            else validation_stringency
        )
        self.__records: Dict[str, MafHeaderRecord] = OrderedDict()
        self.__scheme = None

    def __getitem__(self, key: str) -> MafHeaderRecord:
        return self.__records[key]

    def __setitem__(self, key: str, value: MafHeaderRecord) -> None:
        assert key == value.key
        assert isinstance(value, MafHeaderRecord)
        self.__records[key] = value

    def __delitem__(self, key: str) -> None:
        del self.__records[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__records.keys())

    def __len__(self) -> int:
        return len(self.__records)

    def version(self) -> Optional[str]:
        """Gets the version or `None` if not present"""
        if MafHeader.VersionKey in self.__records:
            return str(self.__records[MafHeader.VersionKey].value)
        else:
            return None

    def annotation(self) -> Optional[str]:
        """Gets the annotation specification or `None` if not present"""
        if MafHeader.AnnotationSpecKey in self.__records:
            return str(self.__records[MafHeader.AnnotationSpecKey].value)
        else:
            return None

    def contigs(self) -> Optional[List[str]]:
        """Gets the contig list or `None` if not present"""
        if MafHeader.ContigKey in self.__records:
            return self.__records[MafHeader.ContigKey].value  # type: ignore
        else:
            return None

    def sort_order(self) -> SortOrderType:
        """Gets the sort order or `Unsorted` if not present"""
        if MafHeader.SortOrderKey in self.__records:
            return self.__records[MafHeader.SortOrderKey].value  # type: ignore
        else:
            return Unsorted()

    def scheme(self) -> Optional['MafScheme']:
        """Gets the scheme according to the version and annotation, None if
        no suitable scheme was found.
        """
        try:
            return find_scheme(version=self.version(), annotation=self.annotation())
        except ValueError:
            return None

    def validate(
        self,
        validation_stringency: ValidationStringency = None,
        logger: logging.Logger = Logger.RootLogger,
        reset_errors: bool = True,
    ) -> List[MafValidationError]:
        """Validates the header and returns a list of errors.
        Ensures that:
        - there is a version line in the header
        - the version is supported
        - the annotation specification is not in the header if the scheme is
          basic
        - the annotation specification is in the header if the scheme is basic
        - the annotation specification, when present, is supported
        """

        if reset_errors:
            self.validation_errors = list()

        def add_error(error: MafValidationError) -> None:
            self.validation_errors.append(error)

        # get the scheme!
        scheme = self.scheme()

        if not validation_stringency:
            validation_stringency = self.validation_stringency

        # ensure there's a version record
        if MafHeader.VersionKey not in self:
            add_error(
                MafValidationError(
                    MafValidationErrorType.HEADER_MISSING_VERSION,
                    "No version line found in the header",
                )
            )
        else:
            # ensure that the version is a supported version
            version = self[MafHeader.VersionKey].value
            if version not in MafHeader.SupportedVersions:
                add_error(
                    MafValidationError(
                        MafValidationErrorType.HEADER_UNSUPPORTED_VERSION,
                        "The version '%s' is not supported" % version,
                    )
                )

        # Check the annotation spec
        # 1. basic annotation specs should not be in the header
        # 2. non-basic annotation specs should be present (in the header) and
        # have a known value
        if scheme is not None and scheme.is_basic():
            if MafHeader.AnnotationSpecKey in self:
                add_error(
                    MafValidationError(
                        MafValidationErrorType.HEADER_UNSUPPORTED_ANNOTATION_SPEC,
                        "Unexpected annotation.spec line found in the header",
                    )
                )
        else:
            if MafHeader.AnnotationSpecKey not in self:
                add_error(
                    MafValidationError(
                        MafValidationErrorType.HEADER_MISSING_ANNOTATION_SPEC,
                        "No annotation.spec line found in the header",
                    )
                )
            else:
                # ensure that the annotation spec is a supported annotation spec
                annotation = self[MafHeader.AnnotationSpecKey].value
                if annotation not in MafHeader.SupportedAnnotationSpecs:
                    add_error(
                        MafValidationError(
                            MafValidationErrorType.HEADER_UNSUPPORTED_ANNOTATION_SPEC,
                            "The annotation.spec '%s' is not supported" % annotation,
                        )
                    )

        # process validation errors
        MafValidationError.process_validation_errors(
            validation_errors=self.validation_errors,
            validation_stringency=validation_stringency,
            logger=logger,
        )

        return self.validation_errors

    def __str__(self) -> str:
        """gets the text representation of the header"""
        return "\n".join([str(record) for record in self.values()])

    @classmethod
    def from_lines(
        cls,
        lines: List[str],
        validation_stringency: ValidationStringency = None,
        logger: logging.Logger = Logger.RootLogger,
    ) -> 'MafHeader':
        """
        :param lines: a sequence of lines
        :param validation_stringency: optionally the validation stringency to
        use, otherwise use the default (Silent)
        :param logger the logger to which to write errors
        :return: a MafHeader
        """

        header = cls(validation_stringency=validation_stringency)

        def add_error(error: MafValidationError) -> None:
            header.validation_errors.append(error)

        for line_number, line in enumerate(lines):
            line_number = line_number + 1  # 1-based
            record, error = MafHeaderRecord.from_line(line, line_number)
            if error:
                assert record is None
                add_error(error)
            else:
                assert record is not None
                if record.key in header:
                    add_error(
                        MafValidationError(
                            MafValidationErrorType.HEADER_DUPLICATE_KEYS,
                            "Multiple header lines with key '%s' found" % record.key,
                            line_number=line_number,
                        )
                    )
                else:
                    header[record.key] = record

        if header.contigs():
            if header.sort_order() and issubclass(
                header.sort_order().__class__, Coordinate
            ):
                sokey = header[MafHeader.SortOrderKey].value.name()
                header[MafHeader.SortOrderKey] = MafHeaderSortOrderRecord(
                    value=sokey, contigs=header.contigs()
                )

        header.validate(logger=logger, reset_errors=False)

        return header

    @classmethod
    def from_line_reader(
        cls,
        line_reader: LineReader,
        validation_stringency: ValidationStringency = None,
        logger: logging.Logger = Logger.RootLogger,
    ) -> 'MafHeader':
        """Reads a header from a line reader.
        :param line_reader: a line reader
        :param validation_stringency: optionally the validation stringency to
        use, otherwise use the default (Silent)
        :param logger the logger to which to write errors
        :return: a MafHeader
        """
        lines = list()
        while True:
            line = line_reader.peek_line()
            if not line.startswith(MafHeader.HeaderLineStartSymbol):
                break
            lines.append(line_reader.read_line())

        return cls.from_lines(
            lines=lines, validation_stringency=validation_stringency, logger=logger
        )

    @classmethod
    def from_reader(
        cls,
        reader: 'MafReader',
        version: Optional[str] = None,
        annotation: Optional[str] = None,
        sort_order: SortOrderType = None,
        fasta_index: Optional[str] = None,
        contigs: Optional[list] = None,
    ) -> 'MafHeader':
        header: 'MafHeader' = deepcopy(reader.header())
        if version:
            header[MafHeader.VersionKey] = MafHeaderVersionRecord(value=version)
        if annotation:
            header[MafHeader.AnnotationSpecKey] = MafHeaderAnnotationSpecRecord(
                value=annotation
            )
        if fasta_index:
            handle = open(fasta_index, "r")
            contigs = [line.rstrip("\r\n").split("\t")[0] for line in handle]
            handle.close()
            header[MafHeader.ContigKey] = MafHeaderContigRecord(value=contigs)
        elif contigs:
            assert isinstance(
                contigs, list
            ), "contigs must be a list, but {0} found".format(type(contigs))
            header[MafHeader.ContigKey] = MafHeaderContigRecord(value=contigs)
        if sort_order:
            header[MafHeader.SortOrderKey] = MafHeaderSortOrderRecord(
                value=sort_order, fasta_index=fasta_index, contigs=contigs
            )
            if (
                not contigs
                and hasattr(header[MafHeader.SortOrderKey].value, "_contigs")
                and getattr(header[MafHeader.SortOrderKey].value, "_contigs")
            ):
                header[MafHeader.ContigKey] = MafHeaderContigRecord(
                    value=header[MafHeader.SortOrderKey].value._contigs
                )
        return header

    @classmethod
    def from_defaults(
        cls,
        version: Optional[str] = None,
        annotation: Optional[str] = None,
        sort_order: SortOrderType = None,
        fasta_index: Optional[str] = None,
        contigs: Optional[list] = None,
    ) -> 'MafHeader':
        header = MafHeader()
        if version:
            header[MafHeader.VersionKey] = MafHeaderVersionRecord(value=version)
        if annotation:
            header[MafHeader.AnnotationSpecKey] = MafHeaderAnnotationSpecRecord(
                value=annotation
            )
        if fasta_index:
            handle = open(fasta_index, "r")
            _contigs = [line.rstrip("\r\n").split("\t")[0] for line in handle]
            handle.close()
            header[MafHeader.ContigKey] = MafHeaderContigRecord(value=_contigs)
        elif contigs:
            assert isinstance(
                contigs, list
            ), "contigs must be a list, but {0} found".format(type(contigs))
            header[MafHeader.ContigKey] = MafHeaderContigRecord(value=contigs)
        if sort_order:
            header[MafHeader.SortOrderKey] = MafHeaderSortOrderRecord(
                value=sort_order, fasta_index=fasta_index, contigs=contigs
            )
            if (
                not contigs
                and hasattr(header[MafHeader.SortOrderKey].value, "_contigs")
                and getattr(header[MafHeader.SortOrderKey].value, "_contigs")
            ):
                header[MafHeader.ContigKey] = MafHeaderContigRecord(
                    value=header[MafHeader.SortOrderKey].value._contigs
                )
        return header

    @classmethod
    def scheme_header_lines(cls, scheme: 'MafScheme') -> List[str]:
        """Gets the list of header lines as they would be printed in a
        MafHeader for the given scheme."""
        return [
            f"{MafHeader.HeaderLineStartSymbol}{MafHeader.VersionKey} {scheme.version()}",
            f"{MafHeader.HeaderLineStartSymbol}{MafHeader.AnnotationSpecKey} {scheme.annotation_spec()}",
        ]
