"""Global fixed-window rate limit, enforced at the MCP execution boundary."""
import hashlib

from app.services.tool_errors import ToolExecutionError

_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], 60) end
return {count, redis.call('TTL', KEYS[1])}
"""


def require_email_capacity(settings):
    if settings.alert_email_rate_limit_per_minute <= 0:
        return
    if settings.alert_rate_limit_backend != "redis":
        raise ToolExecutionError("邮件限流必须使用 redis；测试可将额度设为 0", code="CONFIGURATION", retryable=False)
    import redis
    identity = f"{settings.smtp_host}:{settings.alert_email_from or settings.smtp_username}"
    key = "caretrace:alert-rate:" + hashlib.sha256(identity.encode()).hexdigest()
    client = redis.Redis.from_url(settings.redis_url, socket_timeout=settings.redis_socket_timeout_seconds,
                                  socket_connect_timeout=settings.redis_socket_timeout_seconds)
    try:
        count, ttl = client.eval(_SCRIPT, 1, key)
    except redis.RedisError as exc:
        raise ToolExecutionError("全局限流服务不可用，暂缓发送", code="RATE_LIMIT_UNAVAILABLE", retry_after=5) from exc
    finally:
        client.close()
    if count > settings.alert_email_rate_limit_per_minute:
        raise ToolExecutionError("邮件发送已限流", code="RATE_LIMITED", retry_after=max(1, ttl))
