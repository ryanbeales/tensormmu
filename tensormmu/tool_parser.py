"""Unified OpenAI & Anthropic Tool-Calling + Message Protocol Layer for TensorMMU (`Qwen3-Next` & `Gemma 4`).

Handles:
  1. Canonical normalization of OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) requests,
     including `tools`, `tool_choice`, `thinking` budgets, `tool_use`/`tool_result`, and Gemma-4 dict-typed
     `tool_calls[].function.arguments`.
  2. Lossless parsing of both Qwen3-Next `<tool_call>{...}</tool_call>` and Gemma-4
     `<|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>` outputs.
  3. Real-time streaming holdback (`ToolAwareStreamFilter`) so raw `<tool_call>` markup never leaks into
     streamed text deltas before structured `tool_calls` / `tool_use` SSE blocks are emitted.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ParsedToolCall:
    call_id: str
    anthropic_id: str
    name: str
    arguments_dict: Dict[str, Any]
    arguments_json: str

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }

    def to_anthropic_block(self) -> Dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.anthropic_id,
            "name": self.name,
            "input": self.arguments_dict,
        }


def _flatten_text_blocks(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                itype = item.get("type", "text")
                if itype in ("text", "input_text") and "text" in item:
                    parts.append(str(item["text"]))
        return "".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _coerce_args_to_dict(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if raw_args is None or raw_args == "":
        return {}
    if isinstance(raw_args, str):
        s = raw_args.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except Exception:
            return {"_raw": raw_args}
    return {"value": raw_args}


def normalize_openai_tools(
    raw_tools: Optional[Sequence[Dict[str, Any]]],
    tool_choice: Any = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Returns `(normalized_tools_or_None, forced_tool_instruction_or_None)`."""
    if not raw_tools:
        return None, None

    # Check if `tool_choice` explicitly disables tools
    if tool_choice == "none" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "none"):
        return None, None

    norm_tools: List[Dict[str, Any]] = []
    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            fn = t["function"]
            name = str(fn.get("name", "")).strip()
            if not name:
                continue
            norm_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(fn.get("description", "") or ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
        elif "name" in t:
            # Anthropic-style or flat tool dict
            name = str(t.get("name", "")).strip()
            if not name:
                continue
            schema = t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}}
            norm_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(t.get("description", "") or ""),
                        "parameters": schema,
                    },
                }
            )

    if not norm_tools:
        return None, None

    forced_instruction: Optional[str] = None
    forced_name: Optional[str] = None
    if tool_choice in ("required", "any"):
        forced_instruction = "You MUST invoke at least one of the provided tools in this response."
    elif isinstance(tool_choice, dict):
        tc_type = str(tool_choice.get("type", "auto")).lower()
        if tc_type in ("any", "required"):
            forced_instruction = "You MUST invoke at least one of the provided tools in this response."
        elif tc_type == "function" and isinstance(tool_choice.get("function"), dict):
            forced_name = str(tool_choice["function"].get("name", "")).strip() or None
        elif tc_type == "tool" and tool_choice.get("name"):
            forced_name = str(tool_choice["name"]).strip() or None

    if forced_name:
        filtered = [t for t in norm_tools if t["function"]["name"] == forced_name]
        if filtered:
            norm_tools = filtered
        forced_instruction = f"You MUST call the `{forced_name}` tool in this response."

    return norm_tools, forced_instruction


def normalize_chat_messages_with_tools(
    raw_messages: Sequence[Dict[str, Any]],
    forced_tool_instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Normalizes OpenAI messages while preserving `tool_calls` (dict arguments) and `role: 'tool'` turns."""
    norm: List[Dict[str, Any]] = []
    call_id_to_name: Dict[str, str] = {}

    for m in raw_messages:
        role = str(m.get("role", "user")).strip().lower()
        if role == "developer":
            role = "system"
        elif role not in ("system", "user", "assistant", "tool"):
            role = "user"

        if role == "assistant":
            entry: Dict[str, Any] = {
                "role": "assistant",
                "content": _flatten_text_blocks(m.get("content", "")),
            }
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if reasoning:
                entry["reasoning_content"] = str(reasoning)
            raw_tcs = m.get("tool_calls")
            if isinstance(raw_tcs, list) and raw_tcs:
                clean_tcs: List[Dict[str, Any]] = []
                for idx, tc in enumerate(raw_tcs):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                    fn_name = str(fn.get("name", "")).strip()
                    if not fn_name:
                        continue
                    tc_id = str(tc.get("id") or f"call_{idx}_{fn_name}")
                    args_dict = _coerce_args_to_dict(fn.get("arguments"))
                    call_id_to_name[tc_id] = fn_name
                    clean_tcs.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                # Dict satisfies Gemma 4 Jinja requirement AND Qwen3-Next `| tojson`!
                                "arguments": args_dict,
                            },
                        }
                    )
                if clean_tcs:
                    entry["tool_calls"] = clean_tcs
            norm.append(entry)
        elif role == "tool":
            tc_id = str(m.get("tool_call_id") or "")
            fn_name = str(m.get("name") or call_id_to_name.get(tc_id, "tool"))
            entry = {
                "role": "tool",
                "content": _flatten_text_blocks(m.get("content", "")),
                "name": fn_name,
            }
            if tc_id:
                entry["tool_call_id"] = tc_id
            norm.append(entry)
        else:
            norm.append({"role": role, "content": _flatten_text_blocks(m.get("content", ""))})

    if forced_tool_instruction:
        if norm and norm[0]["role"] == "system":
            norm[0]["content"] = f"{norm[0]['content']}\n\n{forced_tool_instruction}".strip()
        else:
            norm.insert(0, {"role": "system", "content": forced_tool_instruction})

    return norm


def anthropic_messages_to_openai(
    system_param: Any,
    anthropic_messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Converts Anthropic `/v1/messages` payload (`system` + `messages` with `tool_use`/`tool_result`) to canonical."""
    out: List[Dict[str, Any]] = []

    sys_text = _flatten_text_blocks(system_param).strip()
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    id_to_name: Dict[str, str] = {}
    for m in anthropic_messages:
        role = str(m.get("role", "user")).strip().lower()
        content = m.get("content", "")

        if isinstance(content, str) or not isinstance(content, list):
            out.append({"role": role, "content": _flatten_text_blocks(content)})
            continue

        if role == "assistant":
            text_parts: List[str] = []
            think_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for blk in content:
                if not isinstance(blk, dict):
                    text_parts.append(str(blk))
                    continue
                btype = blk.get("type", "text")
                if btype == "text":
                    text_parts.append(str(blk.get("text", "")))
                elif btype == "thinking":
                    think_parts.append(str(blk.get("thinking", "")))
                elif btype == "tool_use":
                    tid = str(blk.get("id") or f"toolu_{uuid.uuid4().hex[:16]}")
                    tname = str(blk.get("name", "tool"))
                    tinput = _coerce_args_to_dict(blk.get("input"))
                    id_to_name[tid] = tname
                    tool_calls.append(
                        {
                            "id": tid,
                            "type": "function",
                            "function": {"name": tname, "arguments": tinput},
                        }
                    )
            entry: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if think_parts:
                entry["reasoning_content"] = "\n".join(think_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        else:
            # Anthropic puts `tool_result` blocks inside `role: "user"` messages!
            tool_results: List[Dict[str, Any]] = []
            user_text_parts: List[str] = []
            for blk in content:
                if not isinstance(blk, dict):
                    user_text_parts.append(str(blk))
                    continue
                btype = blk.get("type", "text")
                if btype == "tool_result":
                    tuid = str(blk.get("tool_use_id") or "")
                    res_txt = _flatten_text_blocks(blk.get("content", ""))
                    if blk.get("is_error"):
                        res_txt = f"[Tool Error]\n{res_txt}"
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tuid,
                            "name": id_to_name.get(tuid, "tool"),
                            "content": res_txt,
                        }
                    )
                elif btype == "text":
                    user_text_parts.append(str(blk.get("text", "")))

            out.extend(tool_results)
            joined_user = "".join(user_text_parts).strip()
            if joined_user or not tool_results:
                out.append({"role": "user", "content": "".join(user_text_parts)})

    return out


# ---------------------------------------------------------------------------
# Gemma 4 `<|tool_call>call:name{...}<tool_call|>` Recursive Value Parser
# ---------------------------------------------------------------------------

class _Gemma4ArgCursor:
    def __init__(self, src: str) -> None:
        self.src = src
        self.n = len(src)
        self.i = 0

    def skip_ws(self) -> None:
        while self.i < self.n and self.src[self.i] in " \t\r\n":
            self.i += 1

    def startswith(self, lit: str) -> bool:
        return self.src.startswith(lit, self.i)

    def parse_value(self) -> Any:
        self.skip_ws()
        if self.i >= self.n:
            return None
        if self.startswith('<|"|>'):
            self.i += 5
            end = self.src.find('<|"|>', self.i)
            if end == -1:
                val = self.src[self.i :]
                self.i = self.n
                return val
            val = self.src[self.i : end]
            self.i = end + 5
            return val
        ch = self.src[self.i]
        if ch == '"':
            self.i += 1
            chars: List[str] = []
            while self.i < self.n:
                c = self.src[self.i]
                self.i += 1
                if c == "\\" and self.i < self.n:
                    esc = self.src[self.i]
                    self.i += 1
                    chars.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(esc, esc))
                elif c == '"':
                    break
                else:
                    chars.append(c)
            return "".join(chars)
        if ch == "{":
            return self.parse_object()
        if ch == "[":
            return self.parse_array()
        if self.startswith("true"):
            self.i += 4
            return True
        if self.startswith("false"):
            self.i += 5
            return False
        if self.startswith("null"):
            self.i += 4
            return None
        # Number or bare token up to `,` / `}` / `]`
        start = self.i
        while self.i < self.n and self.src[self.i] not in ",}]":
            self.i += 1
        tok = self.src[start : self.i].strip()
        if not tok:
            return None
        try:
            if "." in tok or "e" in tok.lower():
                return float(tok)
            return int(tok)
        except ValueError:
            return tok

    def parse_key(self) -> str:
        self.skip_ws()
        if self.startswith('<|"|>'):
            return str(self.parse_value())
        if self.i < self.n and self.src[self.i] == '"':
            return str(self.parse_value())
        start = self.i
        while self.i < self.n and self.src[self.i] not in ":}":
            self.i += 1
        return self.src[start : self.i].strip()

    def parse_object(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.i < self.n and self.src[self.i] == "{":
            self.i += 1
        while self.i < self.n:
            self.skip_ws()
            if self.i >= self.n or self.src[self.i] == "}":
                self.i += 1
                break
            if self.src[self.i] == ",":
                self.i += 1
                continue
            k = self.parse_key()
            self.skip_ws()
            if self.i < self.n and self.src[self.i] == ":":
                self.i += 1
            v = self.parse_value()
            if k:
                obj[k] = v
            self.skip_ws()
            if self.i < self.n and self.src[self.i] == ",":
                self.i += 1
        return obj

    def parse_array(self) -> List[Any]:
        arr: List[Any] = []
        if self.i < self.n and self.src[self.i] == "[":
            self.i += 1
        while self.i < self.n:
            self.skip_ws()
            if self.i >= self.n or self.src[self.i] == "]":
                self.i += 1
                break
            if self.src[self.i] == ",":
                self.i += 1
                continue
            arr.append(self.parse_value())
            self.skip_ws()
            if self.i < self.n and self.src[self.i] == ",":
                self.i += 1
        return arr


_QWEN_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_GEMMA_TOOL_RE = re.compile(r"<\|tool_call>call:([A-Za-z0-9_.\-]+)\{(.*?)\}<tool_call\|>", re.DOTALL)


def _make_parsed_call(name: str, args_dict: Dict[str, Any]) -> ParsedToolCall:
    uid = uuid.uuid4().hex[:20]
    return ParsedToolCall(
        call_id=f"call_{uid}",
        anthropic_id=f"toolu_{uid}",
        name=name,
        arguments_dict=args_dict,
        arguments_json=json.dumps(args_dict, ensure_ascii=False),
    )


def parse_tool_calls_from_output(
    answer_text: str,
    raw_with_special: Optional[str] = None,
) -> Tuple[str, List[ParsedToolCall]]:
    """Extracts structured tool calls (`Qwen3-Next` or `Gemma 4`) and returns `(cleaned_text, tool_calls)`."""
    calls: List[ParsedToolCall] = []

    # 1. Check Gemma 4 `<|tool_call>call:name{...}<tool_call|>` in `raw_with_special` (or `answer_text`)
    gemma_src = raw_with_special if (raw_with_special and "<|tool_call>" in raw_with_special) else answer_text
    if "<|tool_call>" in gemma_src:
        for m in _GEMMA_TOOL_RE.finditer(gemma_src):
            fn_name = m.group(1).strip()
            inner = "{" + m.group(2) + "}"
            args_dict = _Gemma4ArgCursor(inner).parse_object()
            calls.append(_make_parsed_call(fn_name, args_dict))
        cleaned = _GEMMA_TOOL_RE.sub("", answer_text)
        cleaned = cleaned.replace("<|tool_response>", "").replace("<tool_response|>", "").strip()
        if calls:
            return cleaned, calls

    # 2. Check Qwen3-Next `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`
    if "<tool_call>" in answer_text:
        for m in _QWEN_TOOL_RE.finditer(answer_text):
            body = m.group(1).strip()
            try:
                j = json.loads(body)
                if isinstance(j, dict) and j.get("name"):
                    fn_name = str(j["name"]).strip()
                    args_dict = _coerce_args_to_dict(j.get("arguments", {}))
                    calls.append(_make_parsed_call(fn_name, args_dict))
            except Exception:
                pass

        # Handle edge case where `max_tokens` or stop cut right after JSON before `</tool_call>`
        if not calls and "</tool_call>" not in answer_text:
            _, tail = answer_text.split("<tool_call>", 1)
            tail = tail.strip()
            try:
                j = json.loads(tail)
                if isinstance(j, dict) and j.get("name"):
                    calls.append(_make_parsed_call(str(j["name"]).strip(), _coerce_args_to_dict(j.get("arguments", {}))))
            except Exception:
                pass

        if calls:
            cleaned = _QWEN_TOOL_RE.sub("", answer_text)
            if "<tool_call>" in cleaned:
                cleaned = cleaned.split("<tool_call>", 1)[0]
            return cleaned.strip(), calls

    return answer_text, []


class ToolAwareStreamFilter:
    """Streams visible text deltas while holding back `<tool_call>` / `<|tool_call>` markup."""

    _MARKERS = ("<tool_call>", "<|tool_call>")

    def __init__(self, active: bool = True) -> None:
        self.active = bool(active)
        self.buf: str = ""
        self.suppressed: bool = False

    def push_delta(self, delta: str) -> str:
        if not self.active or not delta:
            return "" if self.suppressed else delta
        if self.suppressed:
            return ""

        self.buf += delta
        for mk in self._MARKERS:
            pos = self.buf.find(mk)
            if pos != -1:
                emit = self.buf[:pos]
                self.buf = ""
                self.suppressed = True
                return emit

        # Hold back any trailing prefix of `<tool_call>` or `<|tool_call>` (up to 12 chars)
        hold = 0
        max_p = max(len(m) for m in self._MARKERS) - 1
        check_len = min(len(self.buf), max_p)
        for k in range(check_len, 0, -1):
            suffix = self.buf[-k:]
            if any(m.startswith(suffix) for m in self._MARKERS):
                hold = k
                break

        if hold > 0:
            emit = self.buf[:-hold]
            self.buf = self.buf[-hold:]
            return emit

        emit = self.buf
        self.buf = ""
        return emit

    def flush(self) -> str:
        if self.suppressed or not self.buf:
            self.buf = ""
            return ""
        out = self.buf
        self.buf = ""
        return out
