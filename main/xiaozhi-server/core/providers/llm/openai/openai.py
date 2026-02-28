import json
import httpx
import openai
from openai.types import CompletionUsage
from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key")
        if "base_url" in config:
            self.base_url = config.get("base_url")
        else:
            self.base_url = config.get("url")
        self.stream = self._parse_bool(config.get("stream", True), default=True)
        self.extra_body = self._parse_json_dict(config.get("extra_body"))
        self.default_headers = self._parse_json_dict(config.get("default_headers"))
        self.allow_message_extras = self._parse_bool(
            config.get("allow_message_extras", None),
            default=("openrouter" in str(self.base_url).lower()),
        )
        self._last_assistant_message_extras = None
        self._reasoning_details_buffer = []
        timeout = config.get("timeout", 300)
        self.timeout = int(timeout) if timeout else 300

        param_defaults = {
            "max_tokens": int,
            "temperature": lambda x: round(float(x), 1),
            "top_p": lambda x: round(float(x), 1),
            "frequency_penalty": lambda x: round(float(x), 1),
        }

        for param, converter in param_defaults.items():
            value = config.get(param)
            try:
                setattr(
                    self,
                    param,
                    converter(value) if value not in (None, "") else None,
                )
            except (ValueError, TypeError):
                setattr(self, param, None)

        logger.debug(
            f"意图识别参数初始化: {self.temperature}, {self.max_tokens}, {self.top_p}, {self.frequency_penalty}"
        )

        model_key_msg = check_model_key("LLM", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": httpx.Timeout(self.timeout),
        }
        if isinstance(self.default_headers, dict) and self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        self.client = openai.OpenAI(**client_kwargs)

    @staticmethod
    def _parse_bool(value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1", "yes", "y", "on"):
                return True
            if s in ("false", "0", "no", "n", "off", ""):
                return False
        return default

    @staticmethod
    def _parse_json_dict(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def normalize_dialogue(self, dialogue):
        """自动修复 dialogue 中缺失 content 的消息，并按需剥离不兼容的扩展字段。"""
        allowed_keys = {"role", "content", "name", "tool_calls", "tool_call_id"}
        for msg in dialogue:
            if "role" in msg and "content" not in msg and "tool_calls" not in msg:
                msg["content"] = ""

            if not self.allow_message_extras and isinstance(msg, dict):
                for k in list(msg.keys()):
                    if k not in allowed_keys:
                        msg.pop(k, None)
        return dialogue

    @staticmethod
    def _extract_extra_field(obj, field_name: str):
        if obj is None:
            return None
        if hasattr(obj, field_name):
            value = getattr(obj, field_name)
            if value is not None:
                return value
        model_extra = getattr(obj, "model_extra", None)
        if isinstance(model_extra, dict) and field_name in model_extra:
            return model_extra.get(field_name)
        return None

    def _reset_last_extras(self):
        self._last_assistant_message_extras = None
        self._reasoning_details_buffer = []

    @staticmethod
    def _to_plain_data(value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [LLMProvider._to_plain_data(v) for v in value]
        if isinstance(value, dict):
            return {k: LLMProvider._to_plain_data(v) for k, v in value.items()}
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return {
                    k: LLMProvider._to_plain_data(v)
                    for k, v in vars(value).items()
                    if not str(k).startswith("_")
                }
            except Exception:
                pass
        return value

    @classmethod
    def _normalize_reasoning_details(cls, value):
        plain = cls._to_plain_data(value)
        if plain is None:
            return []
        if isinstance(plain, list):
            return [x for x in plain if x is not None]
        return [plain]

    def _append_reasoning_item(self, item):
        if item is None:
            return
        for existing in self._reasoning_details_buffer:
            if existing == item:
                return
        self._reasoning_details_buffer.append(item)

    def _capture_reasoning_details(self, obj):
        reasoning_details = self._extract_extra_field(obj, "reasoning_details")
        if reasoning_details is None:
            return

        for item in self._normalize_reasoning_details(reasoning_details):
            self._append_reasoning_item(item)

        if self._reasoning_details_buffer:
            self._last_assistant_message_extras = {
                "reasoning_details": list(self._reasoning_details_buffer)
            }

    def consume_last_assistant_message_extras(self):
        extras = self._last_assistant_message_extras
        self._last_assistant_message_extras = None
        return extras

    def response(self, session_id, dialogue, **kwargs):
        try:
            dialogue = self.normalize_dialogue(dialogue)
            self._reset_last_extras()

            request_params = {
                "model": self.model_name,
                "messages": dialogue,
                "stream": self.stream,
            }

            # 添加可选参数,只有当参数不为None时才添加
            optional_params = {
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            }

            for key, value in optional_params.items():
                if value is not None:
                    request_params[key] = value

            extra_body = kwargs.get("extra_body", None)
            if extra_body is None:
                extra_body = self.extra_body
            if isinstance(extra_body, dict) and extra_body:
                request_params["extra_body"] = extra_body

            if self.stream:
                responses = self.client.chat.completions.create(**request_params)

                is_active = True
                for chunk in responses:
                    try:
                        delta = (
                            chunk.choices[0].delta
                            if getattr(chunk, "choices", None)
                            else None
                        )
                        self._capture_reasoning_details(delta)
                        content = getattr(delta, "content", "") if delta else ""
                    except IndexError:
                        content = ""
                    if content:
                        if "<think>" in content:
                            is_active = False
                            content = content.split("<think>")[0]
                        if "</think>" in content:
                            is_active = True
                            content = content.split("</think>")[-1]
                        if is_active:
                            yield content
            else:
                resp = self.client.chat.completions.create(**request_params)
                msg = resp.choices[0].message if getattr(resp, "choices", None) else None
                self._capture_reasoning_details(msg)
                content = getattr(msg, "content", "") if msg else ""
                if content:
                    # keep consistent with stream path
                    if "<think>" in content:
                        content = content.split("<think>")[0]
                    if "</think>" in content:
                        content = content.split("</think>")[-1]
                    if content:
                        yield content

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in response generation: {e}")

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        try:
            dialogue = self.normalize_dialogue(dialogue)
            self._reset_last_extras()

            request_params = {
                "model": self.model_name,
                "messages": dialogue,
                "stream": self.stream,
                "tools": functions,
            }

            optional_params = {
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            }

            for key, value in optional_params.items():
                if value is not None:
                    request_params[key] = value

            extra_body = kwargs.get("extra_body", None)
            if extra_body is None:
                extra_body = self.extra_body
            if isinstance(extra_body, dict) and extra_body:
                request_params["extra_body"] = extra_body

            if self.stream:
                stream = self.client.chat.completions.create(**request_params)

                for chunk in stream:
                    if getattr(chunk, "choices", None):
                        delta = chunk.choices[0].delta
                        self._capture_reasoning_details(delta)
                        content = getattr(delta, "content", "")
                        tool_calls = getattr(delta, "tool_calls", None)
                        yield content, tool_calls
                    elif isinstance(getattr(chunk, "usage", None), CompletionUsage):
                        usage_info = getattr(chunk, "usage", None)
                        logger.bind(tag=TAG).info(
                            f"Token 消耗：输入 {getattr(usage_info, 'prompt_tokens', '未知')}，"
                            f"输出 {getattr(usage_info, 'completion_tokens', '未知')}，"
                            f"共计 {getattr(usage_info, 'total_tokens', '未知')}"
                        )
            else:
                resp = self.client.chat.completions.create(**request_params)
                msg = resp.choices[0].message if getattr(resp, "choices", None) else None
                self._capture_reasoning_details(msg)
                content = getattr(msg, "content", "") if msg else ""
                tool_calls = getattr(msg, "tool_calls", None) if msg else None
                yield content, tool_calls

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in function call streaming: {e}")
            yield f"【OpenAI服务响应异常: {e}】", None
