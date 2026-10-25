"""Dependency-free guards for the opt-in AI VTuber inference contract."""


class InferenceCancelled(RuntimeError):
    """The request ended by cancellation, never by successful audio EOS."""


def check_request_cancelled(event):
    if event is not None:
        if not callable(getattr(event, "is_set", None)):
            raise TypeError("_cancel_event must provide is_set()")
        if event.is_set():
            raise InferenceCancelled("TTS request cancelled")
