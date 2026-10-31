class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, *, code: str = "EXECUTION_FAILED", retryable: bool = True, retry_after: float | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after


class DeliveryUnknown(ToolExecutionError):
    def __init__(self, message: str):
        super().__init__(message, code="DELIVERY_UNKNOWN", retryable=False)
