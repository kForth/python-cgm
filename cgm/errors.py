"""Custom exceptions for python-cgm."""

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"


class CGMError(Exception):
    """Base exception for CGM parsing and extraction errors."""


class CGMParseError(CGMError):
    """Raised when the CGM stream is malformed or unsupported."""
