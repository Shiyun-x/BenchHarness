# tools/llm_caller.py
# -----------------------------------------------------------------------------
# A lightweight, generic LLM caller as a plain tool.
# - Supports text / JSON two modes.
# - Uses litellm.completion, or opt-in native litellm.responses.
# -----------------------------------------------------------------------------

from typing import Dict, Any, List, Optional, Union
import os
import json
import re
import ast
import base64
import mimetypes
import time
import random
from pathlib import Path
from litellm import completion
import litellm
from utils.native_responses import responses_input, inspect_native_response, sum_usage
from utils.audio_inputs import input_audio

# Configure litellm
litellm.drop_params = True

try:
    from utils.model_config import get_tool_model, get_api_key, get_api_base_url
    DEFAULT_MODEL = get_tool_model("default")
except Exception:
    DEFAULT_MODEL = "gpt-6.1-sol"
    def get_api_key(_config_path=None):
        return os.getenv("YOUR_LLM_API_KEY_ENV", "")
    def get_api_base_url(_config_path=None):
        return None

# ===== Safety / size limits =====
MAX_USER_JSON_CHARS = 16000
MAX_SYS_PROMPT_CHARS = 8000
MAX_USER_PROMPT_CHARS = 12000
MAX_TOKENS = 12000
DEFAULT_LLM_REQUEST_TIMEOUT_S = int(os.getenv("LLM_REQUEST_TIMEOUT_S", "900"))
# Type aliases
JSONType = Union[Dict[str, Any], List[Any]]

def _extract_fenced_blocks(text: str) -> List[str]:
    """
    Extract ```json ... ``` or ``` ... ``` fenced blocks that look like JSON object/array.
    """
    if not text:
        return []
    blocks = []
    # Capture either {...} or [...] inside fences
    for m in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        s = m.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            blocks.append(s)
    return blocks

def _fix_common_json_issues(snippet: str) -> str:
    """
    Attempt light repairs on common non-JSON issues:
      - single quotes -> double quotes
      - True/False/None -> true/false/null
      - trailing commas before ] or }
    """
    if not isinstance(snippet, str):
        return snippet
    fixed = snippet

    # Replace single quotes with double quotes (naive but effective for many cases)
    fixed = fixed.replace("'", '"')

    # Python literals -> JSON
    fixed = re.sub(r'(?<!")\bTrue\b(?!")', 'true', fixed)
    fixed = re.sub(r'(?<!")\bFalse\b(?!")', 'false', fixed)
    fixed = re.sub(r'(?<!")\bNone\b(?!")', 'null', fixed)

    # Remove trailing commas before } or ]
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

    return fixed

def _literal_eval_fallback(snippet: str) -> Optional[JSONType]:
    """
    Fallback using Python's ast.literal_eval for dict/list-like strings.
    """
    try:
        pyobj = ast.literal_eval(snippet)
        if isinstance(pyobj, (dict, list)):
            return pyobj
    except Exception:
        pass
    return None

def _json5_fallback(text: str) -> Optional[JSONType]:
    """
    Optional fallback via json5 if installed.
    """
    try:
        import json5  # type: ignore
        obj = json5.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    return None

def _load_with_redundant_root_closer(text: str) -> Optional[JSONType]:
    """Recover one extra root closer after an otherwise complete JSON value.

    Do not add missing syntax, merge siblings, or ignore other trailing content.
    The complete decoded object still goes through the caller's schema checks.
    """
    try:
        obj, end = json.JSONDecoder().raw_decode(text.lstrip())
        closer = '}' if isinstance(obj, dict) else ']' if isinstance(obj, list) else None
        if closer and text.lstrip()[end:].strip() == closer:
            return obj
    except (ValueError, TypeError):
        pass
    return None


def _safe_json_loads(text: str) -> Optional[JSONType]:
    """
    Strong JSON extractor (top-level dict OR list):
      1) direct json.loads
      2) fenced blocks ```json ... ```
      3) outermost {...} slice (raw) OR outermost [...] slice (raw)
      4) slice + auto-fixes (quotes, literals, trailing commas)
      5) ast.literal_eval
      6) json5 (optional)
    """
    if not text or not isinstance(text, str):
        return None

    # 1) direct
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    obj = _load_with_redundant_root_closer(text)
    if obj is not None:
        return obj

    # 2) fenced blocks
    for block in _extract_fenced_blocks(text):
        try:
            obj = json.loads(block)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            obj = _load_with_redundant_root_closer(block)
            if obj is not None:
                return obj
            fixed = _fix_common_json_issues(block)
            try:
                obj = json.loads(fixed)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                obj = _literal_eval_fallback(block)
                if obj is not None:
                    return obj
                obj = _json5_fallback(block)
                if obj is not None:
                    return obj

    # Keep the first outer container: never salvage an internal array from a broken object.
    obj_start, obj_end = text.find("{"), text.rfind("}")
    arr_start, arr_end = text.find("["), text.rfind("]")

    candidates = []
    if obj_start >= 0 and (arr_start < 0 or obj_start < arr_start):
        if obj_end > obj_start:
            candidates.append(text[obj_start:obj_end+1])
    elif arr_start >= 0 and arr_end > arr_start:
        candidates.append(text[arr_start:arr_end+1])

    # try longer snippet first
    candidates.sort(key=len, reverse=True)

    for snippet in candidates:
        # 3a) raw
        try:
            obj = json.loads(snippet)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
        # 4) fixes
        fixed = _fix_common_json_issues(snippet)
        try:
            obj = json.loads(fixed)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
        # 5) literal eval
        obj = _literal_eval_fallback(snippet)
        if obj is not None:
            return obj
        # 6) json5
        obj = _json5_fallback(snippet)
        if obj is not None:
            return obj

    # last resort: try json5 on full text
    obj = _json5_fallback(text)
    if obj is not None:
        return obj

    return None

def _build_messages(
    system_prompt: str,
    user_prompt: str,
    user_json: Optional[Dict[str, Any]],
    images: Optional[List[str]] = None,
    audios: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-style messages (system + user).

    - If no attachments and no user_json: content is a simple string.
    - Otherwise, user.content is a list of text/image_url/input_audio items.
    - images: Can be remote URLs or local file paths.
      Local paths are read and converted to base64 data URLs.
    - audios: Local files or public URLs, encoded for Qwen Omni.
      Loading errors fail the call instead of silently omitting evidence.
    """
    sys_p = system_prompt
    usr_p = user_prompt

    messages: List[Dict[str, Any]] = []

    if sys_p:
        messages.append({"role": "system", "content": sys_p})

    if not images and not audios and user_json is None:
        if usr_p:
            messages.append({"role": "user", "content": usr_p})
        return messages

    # ---- multimodal mode: OpenAI Vision style ----
    user_content: List[Dict[str, Any]] = []

    if usr_p:
        user_content.append({
            "type": "text",
            "text": usr_p,
        })

    if user_json is not None:
        dumped = json.dumps(user_json, ensure_ascii=False)
        user_content.append({
            "type": "text",
            "text": f"```json\n{dumped}\n```"
        })

    # Process images: remote URLs used directly, local paths converted to base64 data URLs
    if images:
        for img_path in images:
            if not img_path:
                continue

            if re.match(r"^https?://", img_path):
                # Remote URL, use directly
                url = img_path
            else:
                # Local file path: read file and convert to base64 data URL
                abs_path = os.path.abspath(img_path)
                try:
                    with open(abs_path, "rb") as f:
                        data = f.read()
                    b64 = base64.b64encode(data).decode("utf-8")

                    mime, _ = mimetypes.guess_type(abs_path)
                    if not mime:
                        # Windows registry entries can resolve .webp to an empty
                        # string. Keep supported image formats labeled correctly.
                        mime = {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png',
                                '.gif':'image/gif','.webp':'image/webp'}.get(Path(abs_path).suffix.lower(),'image/png')

                    url = f"data:{mime};base64,{b64}"
                except Exception as e:
                    # If reading fails, add a warning text instead of failing the entire call
                    user_content.append({
                        "type": "text",
                        "text": f"[WARN] failed to load image '{abs_path}': {e}"
                    })
                    continue

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            })

    for source in audios or []:
        user_content.append(input_audio(source))

    if user_content:
        messages.append({
            "role": "user",
            "content": user_content
        })

    return messages

def _response_metadata(response) -> Dict[str, Any]:
    """Retain provider citations, rather than URLs invented in generated JSON."""
    try:
        obj = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
    except (TypeError, ValueError):
        return {'usage': {}, 'citations': []}
    citations = []

    def visit(value):
        if isinstance(value, dict):
            if value.get('type') in ('url_citation', 'web_search_result'):
                citation = value.get('url_citation', value)
                url = citation.get('url') if isinstance(citation, dict) else None
                if isinstance(url, str) and url.startswith(('https://', 'http://')):
                    citations.append({'url': url, 'title': citation.get('title', '')})
            for key, child in value.items():
                # Do not harvest generated JSON/content URLs as provider citations.
                if key not in ('content', 'text') or isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(obj)
    hidden = getattr(response, '_hidden_params', {}) or {}
    original_response = hidden.get('original_response') if isinstance(hidden, dict) else None
    if hasattr(original_response, 'model_dump'):
        visit(original_response.model_dump())
    elif isinstance(original_response, dict):
        visit(original_response)
    unique = {c['url']: c for c in citations}
    return {'usage': obj.get('usage') or {}, 'citations': list(unique.values())}



def _retryable_call_error(error):
    """Retry transport failures and transient HTTP statuses, not auth/bad requests."""
    response = getattr(error, 'response', None)
    status = getattr(response, 'status_code', None) or getattr(error, 'status_code', None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    message = (type(error).__name__ + ': ' + str(error)).lower()
    if any(marker in message for marker in ('authentication', 'permissiondenied', 'unauthorized',
                                            'invalid_api_key', 'badrequest', 'notfound')):
        return False
    return any(marker in message for marker in ('timeout', 'timed out', 'connection', 'proxy',
        'ratelimit', 'rate limit', 'internalserver', 'serviceunavailable', 'temporarily unavailable'))


def _completion_response(create_params):
    response = completion(**create_params)
    if not create_params.get('stream'):
        return response
    try:
        chunks = list(response)
        rebuilt = litellm.stream_chunk_builder(chunks, messages=create_params.get('messages'))
    finally:
        close = getattr(response, 'close', None)
        if callable(close):
            close()
    if rebuilt is None:
        raise ConnectionError('Streaming connection ended without a usable completion')
    return rebuilt


def llm_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic LLM call tool.

    Args:
        payload: Dictionary containing:
            - mode: "text" | "json" (default "text")
            - api_mode: "completion" | "responses" (default "completion")
            - model: str (default from config or "gpt-6.1-sol")
            - system_prompt: str (optional)
            - user_prompt: str (optional)
            - user_json: dict (optional; appended as fenced block)
            - images: List[str] (optional; image URLs or local paths)
            - audios: List[str] (optional; Qwen Omni audio URLs or local paths)
            - temperature: float (optional, default 0.2)
            - max_tokens: int (optional, default None → model decides)
            - extra_create_params: dict (optional; pass-through to litellm)
            - response_validator: callable (optional; parsed JSON -> error string or None)
            - json_format_repair: callable (optional; invalid text -> candidate JSON or None;
              stopped responses only, requires response_validator; never sent to the API)
              Local validation shares max_attempts and is never sent to the provider.

    Returns:
        Dictionary with:
            - raw_text: str - The raw response text
            - json: dict or list - Parsed JSON (empty dict if mode is "text" or parsing failed)
            - ok: bool - Whether the call succeeded
            - error: str or None - Error message if failed
    """
    mode: str = payload.get("mode", "text")
    api_mode = payload.get('api_mode', 'completion')
    if api_mode not in ('completion', 'responses'):
        raise ValueError('api_mode must be completion or responses')
    model: str = payload.get("model", DEFAULT_MODEL)
    system_prompt: str = payload.get("system_prompt", "")
    user_prompt: str = payload.get("user_prompt", "")
    user_json: Optional[Dict[str, Any]] = payload.get("user_json")
    images: Optional[List[str]] = payload.get("images")
    audios: Optional[List[str]] = payload.get("audios")
    temperature: float = float(payload.get("temperature", 0.2))
    max_tokens: Optional[int] = payload.get("max_tokens")
    extra_create_params: Dict[str, Any] = payload.get("extra_create_params", {}) or {}
    # response_format is not universally supported by OpenAI-compatible gateways.
    # Default to OFF; callers can explicitly enable when their provider supports it.
    use_response_format: bool = bool(payload.get("use_response_format", False))
    json_root_type = payload.get("json_root_type")
    json_tool_name = payload.get('json_tool_name')
    if json_tool_name and (mode != 'json' or api_mode != 'completion'):
        raise ValueError('json_tool_name requires completion JSON mode')
    if json_root_type not in (None, "object", "array"):
        raise ValueError("json_root_type must be object, array or None")

    # JSON mode enforcement: use stricter settings for better JSON output reliability
    json_mode_original_prompt = system_prompt
    if mode == "json":
        # For JSON mode, default to temperature=0 for more deterministic output
        # unless explicitly set by user
        if "temperature" not in payload:
            temperature = 0.0
        # Add JSON enforcement prompt
        root_description = ("a single top-level JSON " + json_root_type if json_root_type else
                            "a single top-level JSON object OR a JSON array")
        system_prompt += ("\n\n[IMPORTANT] Return ONLY valid JSON (" + root_description +
                          "). Use double quotes for all keys/strings. No trailing commas. [IMPORTANT]")

    messages = _build_messages(system_prompt, user_prompt, user_json, images, audios)

    # API from config (models.yaml) or env; optional model_config_path in payload for flow-specific config
    _config_path = payload.get("model_config_path")
    _api_key = get_api_key(_config_path)
    _base_url = get_api_base_url(_config_path)

    base_create_params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    # Avoid unbounded LiteLLM/provider waits. Callers can override these through
    # extra_create_params when they need a shorter or longer request timeout.
    if DEFAULT_LLM_REQUEST_TIMEOUT_S > 0:
        base_create_params["timeout"] = DEFAULT_LLM_REQUEST_TIMEOUT_S
        base_create_params["request_timeout"] = DEFAULT_LLM_REQUEST_TIMEOUT_S
    if _base_url:
        base_create_params["base_url"] = _base_url
    if _api_key:
        base_create_params["api_key"] = _api_key
    # if max_tokens is not None:
    #     base_create_params["max_tokens"] = int(max_tokens)
    from utils.model_config import get_output_token_budget
    max_tokens = get_output_token_budget(int(payload.get('max_tokens') or MAX_TOKENS), _config_path)
    base_create_params["max_tokens"] = max_tokens
    # Optional: response_format for JSON mode when provider supports it
    if mode == "json" and use_response_format:
        base_create_params["response_format"] = {"type": "json_object"}

    # Apply extra params last, allowing override of response_format if needed
    if extra_create_params:
        base_create_params.update(extra_create_params)
    if api_mode == 'completion' and _config_path:
        policy_path = Path(_config_path).resolve().parent/'generation_policy.json'
        if policy_path.is_file():
            policy = json.loads(policy_path.read_text(encoding='utf-8-sig'))
            if policy.get('stream_completion'):
                base_create_params.setdefault('stream', True)
                if base_create_params.get('stream'):
                    base_create_params.setdefault('stream_options', {'include_usage': True})

    # Local validation is opt-in; it is never forwarded to the API.
    response_validator = payload.get("response_validator")
    if response_validator is not None and not callable(response_validator):
        raise TypeError("response_validator must be callable")
    json_format_repair = payload.get('json_format_repair')
    if json_format_repair is not None and (not callable(json_format_repair) or response_validator is None):
        raise TypeError('json_format_repair requires a callable and response_validator')
    last_validation_error = None
    last_response_error_kind = None
    validation_errors = []
    last_error = None
    last_content = ""
    attempt_usage = []
    attempt_records = []
    last_metadata = {}
    max_attempts = max(1, min(5, int(payload.get('max_attempts', 3 if mode == 'json' else 2))))

    def totals():
        result = sum_usage(attempt_usage)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'input_tokens', 'output_tokens'):
            result.setdefault(key, 0)
        return result

    for attempt in range(1, max_attempts + 1):
        last_validation_error = None
        last_response_error_kind = None
        start = time.monotonic()
        record = {'attempt': attempt, 'ok': False, 'output_chars': 0,
                  'finish_reason': None, 'usage': {}, 'retry_delay_s': 0}
        retryable = True
        try:
            create_params = dict(base_create_params)
            if mode == "json":
                create_params["temperature"] = temperature
            if api_mode == 'responses':
                native_params = dict(create_params)
                native_params['input'] = responses_input(native_params.pop('messages'))
                native_params['max_output_tokens'] = native_params.pop('max_tokens')
                native_params.pop('request_timeout', None)
                native_params.pop('temperature', None)
                native_params.pop('custom_llm_provider', None)
                if 'base_url' in native_params:
                    native_params['api_base'] = native_params.pop('base_url')
                if 'response_format' in native_params:
                    native_params['text'] = {'format': native_params.pop('response_format')}
                native_model = native_params['model']
                if native_model.startswith('openai/responses/'):
                    native_model = 'openai/' + native_model[len('openai/responses/'):]
                elif native_model.startswith('gpt-'):
                    native_model = 'openai/' + native_model
                native_params['model'] = native_model
                native_params['drop_params'] = False
                resp = litellm.responses(**native_params)
                raw = resp.model_dump(mode='json')
                metadata = inspect_native_response(raw)
                content = metadata.pop('raw_text')
                finish_reason = metadata.pop('finish_reason')
            else:
                resp = _completion_response(create_params)
                content = resp.choices[0].message.content or ""
                finish_reason = getattr(resp.choices[0], 'finish_reason', None)
                metadata = _response_metadata(resp)
                if json_tool_name:
                    calls = getattr(resp.choices[0].message, 'tool_calls', None) or []
                    valid = (finish_reason == 'tool_calls' and len(calls) == 1
                             and calls[0].function.name == json_tool_name)
                    content = calls[0].function.arguments if valid else ''
                    metadata['json_tool_name'] = json_tool_name
            last_metadata = metadata
            if api_mode == 'responses':
                record['provider_response'] = metadata['provider_response']
                record['web_search_calls'] = metadata['web_search_calls']
            attempt_usage.append(metadata.get('usage', {}))
            record.update(output_chars=len(content), finish_reason=finish_reason, usage=metadata.get('usage', {}),
                          stream=bool(create_params.get('stream')))
            parsed = _safe_json_loads(content) if mode == 'json' else {}
            format_repaired = False
            if (mode == 'json' and parsed is None and finish_reason == 'stop'
                    and json_format_repair is not None):
                parsed = json_format_repair(content)
                format_repaired = isinstance(parsed, (dict, list))
            if mode == 'json' and finish_reason in ('length', 'incomplete'):
                last_response_error_kind = 'output_limit'
                last_error = f'Model response is incomplete (finish_reason={finish_reason})'
            else:
                wrong_root = (mode == 'json' and isinstance(parsed, (dict, list)) and
                    ((json_root_type == 'object' and not isinstance(parsed, dict)) or
                     (json_root_type == 'array' and not isinstance(parsed, list))))
                last_validation_error = ('Expected a complete top-level JSON ' + json_root_type
                    if wrong_root else (response_validator(parsed) if mode == 'json' and
                    response_validator is not None and isinstance(parsed, (dict, list)) else None))
                if mode != 'json' or (isinstance(parsed, (dict, list)) and not last_validation_error):
                    record.update(ok=True, seconds=round(time.monotonic()-start, 3))
                    if format_repaired:
                        record['format_repaired'] = True
                    attempt_records.append(record)
                    return {**metadata, 'attempts': attempt, 'attempt_usage': list(attempt_usage),
                            'attempt_records': attempt_records, 'usage': totals(), 'finish_reason': finish_reason,
                            'validation_errors': list(validation_errors),
                            'raw_text': content, 'json': parsed, 'ok': True, 'error': None}
                if last_validation_error:
                    last_response_error_kind = 'response_validation'
                    validation_errors.append({'attempt': attempt, 'error': last_validation_error})
                    last_error = 'Response validation failed: ' + last_validation_error
                else:
                    last_response_error_kind = 'json_parse'
                    last_error = f"Failed to parse JSON from model response. content_len={len(content)}"
                    try:
                        json.loads(content)
                    except json.JSONDecodeError as error:
                        last_error += f'; {error.msg} at line {error.lineno}, column {error.colno}'
            last_content = content
            if attempt < max_attempts:
                feedback = last_error[:1200]
                stronger_prompt = json_mode_original_prompt + (
                    f"\n\n[IMPORTANT - RETRY {attempt + 1}] Return ONLY valid JSON with the required "
                    "top-level keys, nesting and value types. Use double quotes, no trailing commas. "
                    f"Previous response error: {feedback}. Re-evaluate the same request and complete all checks.")
                base_create_params["messages"] = _build_messages(stronger_prompt, user_prompt, user_json, images, audios)
        except Exception as exc:
            last_response_error_kind = 'call_error'
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = _retryable_call_error(exc)
            # This is SDK-reported status; a synthetic response does not prove a remote HTTP error.
            http_status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if isinstance(http_status, int):
                record['http_status'] = http_status
            record['exception_type'] = type(exc).__name__
            if retryable and attempt < max_attempts:
                record['retry_delay_s'] = round(min(30.0, 5.0 * (2 ** (attempt - 1)) + random.uniform(0, 1)), 3)
        record.update(seconds=round(time.monotonic()-start, 3), error=last_error,
                      error_kind=last_response_error_kind, retryable=retryable)
        attempt_records.append(record)
        if not retryable:
            break
        if record['retry_delay_s']:
            time.sleep(record['retry_delay_s'])

    return {**last_metadata, 'attempts': len(attempt_records), 'attempt_usage': attempt_usage,
            'attempt_records': attempt_records, 'response_error_kind': last_response_error_kind,
            'validation_error': last_validation_error, 'validation_errors': validation_errors,
            'usage': totals(), 'raw_text': last_content, 'json': {}, 'ok': False,
            'error': f"LLM call failed after {len(attempt_records)} attempts. Last error: {last_error}"}

# -------------------------- Convenience wrappers ------------------------------

def llm_call_text(
    system_prompt: str,
    user_prompt: str = "",
    user_json: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Simple wrapper for text mode."""
    if model is None:
        model = DEFAULT_MODEL
    return llm_call({
        "mode": "text",
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "user_json": user_json,
        **kwargs
    })

def llm_call_json(
    system_prompt: str,
    user_prompt: str = "",
    user_json: Optional[Dict[str, Any]] = None,
    images: Optional[List[str]] = None,
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper for JSON mode with enhanced reliability.

    This function ensures JSON output by:
    1. Setting temperature=0 by default for deterministic output
    2. Using stronger prompts and multiple retry attempts
    3. Robust JSON parsing with fallback strategies
    4. (Optional) Using response_format when explicitly enabled and supported by provider

    Note: While this significantly improves JSON output reliability,
    it cannot guarantee 100% success if the model doesn't support
    structured outputs or if the prompt is fundamentally incompatible.

    Args:
        system_prompt: System prompt (will be enhanced with JSON enforcement)
        user_prompt: User prompt
        user_json: Optional JSON to include in the request
        images: Optional list of image URLs or paths
        model: Model name (defaults to DEFAULT_MODEL)
        **kwargs: Additional parameters passed to llm_call (e.g., use_response_format=True)

    Returns:
        Dictionary with 'ok', 'json', 'raw_text', and 'error' fields.
        If 'ok' is False, check 'error' for details.
    """
    if model is None:
        model = DEFAULT_MODEL
    return llm_call({
        "mode": "json",
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "user_json": user_json,
        "images": images,
        **kwargs
    })

