""" Setup logging """

import logging


class DisableLogging:
    """
    Temporarily disable logging (except for the `CRITICAL` level messages).
    Adapted from https://stackoverflow.com/a/20251235. Use as

    .. code-block:: python

        with DisableLogging():
            do_your_stuff
    """

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


class FirstPartFilter(logging.Filter):
    """
    A filter that provides the `name_first` variable for formatting. For a
    logger called "aaa.bbb.ccc", name_first="aaa".
    Adapted from https://stackoverflow.com/a/46961676.
    """

    def filter(self, record):
        record.name_first = record.name.rsplit(".", 1)[0]
        return True


class NoUsingBackendFilter(logging.Filter):
    """
    A filter that removes the "Using ... backend" log record of geomstats.
    """

    def filter(self, record):
        msg = record.getMessage()
        # TODO: when geomstats implements better logging, add
        # msg.name_first == "geomstats"
        # as the third condition for filtering.
        return not (msg.startswith("Using ") and msg.endswith(" backend"))


root_handler = logging.StreamHandler()
root_handler.addFilter(FirstPartFilter())
root_handler.addFilter(NoUsingBackendFilter())
formatter = logging.Formatter("%(levelname)s (%(name_first)s): %(message)s")
root_handler.setFormatter(formatter)
# Note: using baseConfig allows the "outermost" code to define the logging
# policy: once one baseConfig has been called, each subsequent basicConfig
# call is ignored. That is unless force=True parameter is set, which,
# hopefully, is only done sparingly and with good reason.
logging.basicConfig(handlers=[root_handler])

logger = logging.getLogger("geometric_kernels")
logger.setLevel(logging.INFO)  # can be easily changed by downstream code
