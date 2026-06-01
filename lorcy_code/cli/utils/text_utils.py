def mask_api_key(key: str, mask: str = "...", short_mask: str = "***") -> str:
    if not key:
        return "未配置"
    if len(key) <= 10:
        return short_mask
    return f"{key[:6]}{mask}{key[-4:]}"