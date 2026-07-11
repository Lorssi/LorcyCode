RETRY_DELAYS = [3, 10, 30, 60]


class ModelRetryException(Exception):
    """模型调用失败，且没有更多备用模型可用时抛出此异常。"""
    pass

class ModelRetryManager:
    def __init__(self):
        self._fallback_models: list[dict] = []
        self._fallback_index: int = 0

    def set_fallback_models(self, models: list[dict]) -> None:
        self._fallback_models = models
        self._fallback_index = 0

    def get_fallback_model(self) -> dict | None:
        if self._fallback_index < len(self._fallback_models):
            return self._fallback_models[self._fallback_index]
        return None

    def advance_fallback(self) -> None:
        self._fallback_index += 1

    def load_fallback_models(self, current_model: str | None = None) -> list[dict]:
        """Load fallback models from config and reset index."""
        from lorcy_code.config.storage import load_model_json

        data = load_model_json()
        fallback = data.get("fallback", {})
        if not fallback:
            self.set_fallback_models([])
            return []

        models = list(fallback.values())
        if current_model:
            models = [m for m in models if m.get("model") != current_model]
        self.set_fallback_models(models)
        return models

    def get_or_load_fallback_model(
        self, current_model: str | None = None
    ) -> dict | None:
        """Return current fallback model, loading from config if needed."""
        if not self._fallback_models:
            self.load_fallback_models(current_model=current_model)
        return self.get_fallback_model()


fallback_manager = ModelRetryManager()
