"""Shared exception types."""


class PipelineError(Exception):
    """A failure whose message is safe and useful to show to the user.

    Services raise this for expected problems (bad input, missing API key, an external
    API being down). The pipeline catches it and stores the message on the job.
    """
