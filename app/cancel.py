class CancelledRun(Exception):
    """
    Raised when the user requested cancel.
    We use this to bail out of long loops immediately,
    without treating it as a hard error.
    """
    pass
