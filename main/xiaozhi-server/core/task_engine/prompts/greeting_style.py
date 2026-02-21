from __future__ import annotations

import random
from datetime import datetime

from jinja2 import Template


# 单一母提示词模板：后续只需要替换这段即可，不用改代码逻辑。
# 变量约定（Jinja2）：
# - {{ NOW_TIME }}: 本次任务的“计划时刻”(HH:MM)
# - {{ MAIN_HINT }}: 该 slot 的主意图（通常来自固定映射）
# - {{ OPTIONAL_HINTS }}: 从候选池随机抽取的风格提示（可能为空）
# - {{ CONTEXT_VARS }}: 额外上下文信息（例如：工作日/周末、wake_up_check_result 等）
SCHEDULED_TASK_BASE_PROMPT_TEMPLATE = """\
Task: Natural Engagement Generation

Based on the predefined [Role], generate a proactive, non-coercive engagement message.

Context adjustment:
- You MUST ground the message in CONTEXT_VARS: ({{ CONTEXT_VARS }}) to avoid mismatched context.
- Do NOT force-mention CONTEXT_VARS: only reference it if it naturally fits the topic; if it doesn't, leave it unmentioned.
- It is a background hint; do not repeat the main line verbatim.

Input integration (weave into one coherent passage; you may integrate the elements in any order; do not mechanically stitch blocks):
\t1) Time anchor:
\t- MUST include the time anchor (this run: {{ NOW_TIME }}) at least to the hour (HH), somewhere in the passage (position is flexible).
\t- Use it only as a light presence anchor; do not attach complex advice here.

\t2) Core narrative:
\t- Use 1-2 primary sentence to clearly express the core intent: {{ MAIN_HINT }}.
\t- Red line: NO questions.

\t3) Optional texture:
\t- Add 0–4 extra sentences inspired by OPTIONAL_HINTS: {{ OPTIONAL_HINTS }}
\t- If OPTIONAL_HINTS is empty or unnecessary, add nothing (no forced filler).

Greeting opener (optional):
- You MAY start with a short neutral hello (e.g., "Phew..."/"Hmm..."/ "Good evening"), only if it feels natural for [Role]; no questions.
- Keep it lightweight, then transition into the core narrative without repeating it verbatim.

Humanization rules:
- You MAY include a short playful/teasing line when context feels odd (e.g., late-night “morning call”).
- Interaction boundaries: NO questions; NO waiting/dependency statements.

Output requirements:
- Plain text only. No tags. No explanations.
- 2–6 sentences, but keep it coherent as one passage.
- Pre-check: includes {{ NOW_TIME }} (at least to the hour (HH)); no questions; no waiting/dependency tone.
- Output language should remain consistent with the setting defined in [Role].
- Response MUST in japanese.

Action:
Ignore all intermediate reasoning. Activate your [Role] and output the final message directly.
""".strip()

_SCHEDULED_TASK_TEMPLATE = Template(SCHEDULED_TASK_BASE_PROMPT_TEMPLATE)


MAIN_HINT_BY_SLOT: dict[str, str] = {
    "morning": (
        "Do Morning call. Your task is WAKE UP the user! Determine the user status as awake/asleep/unknown/nobody ONLY based on "
        "the wake_up_check_result in CONTEXT_VARS. If result missing or result show unknown means status uncertain. "
        "Verify that your response is consistent with the evidence to ensure reliable and non-contradictory conclusions. "
        "EXPLICITLY WAKE THE USER UP if the status is asleep. Use evidence only if provided; never fabricate. May use "
        "wake_check tool once again in follow-up chats for a double check."
    ),
    "noon": "Midday greeting, mood reset",
    "night": "Evening wrap-up, rest reminder",
    "commute": "Long day—good job",
}


_OPTIONAL_HINT_EMPTY = ""
_OPTIONAL_HINT_PRESENCE = (
    "Generate 1 lightweight “presence” line: like a casual hello or a brief, understated remark about the moment; "
    "do not solicit a reply; no question marks; no lecturing or commanding."
)
_OPTIONAL_HINT_ASIDE = (
    "Generate 1 improvised aside: choose one from “instant feeling / visual imagery / mild rant”; "
    "no grand takeaways or life summaries; do not push for a response."
)
_OPTIONAL_HINT_CHARACTER_STATE = (
    "Generate 1 first-person character-state line: pick one from "
    "“current state / something noticed / what you're doing / a small inner thought”."
)
_OPTIONAL_HINT_CONTEXT_HOOK = (
    "Generate 1 context hook line: draw atmosphere from the external environment (weather / news / holiday vibe / "
    "conversation history); avoid stiff, broadcast-style delivery."
)
_OPTIONAL_HINT_MICRO_ACTION_NUDGE = (
    "Generate 1 “optional” micro-action nudge: pick one from "
    "“take a sip of water / shift posture / sit up a bit / blink / glance out the window”; "
    "must use optional phrasing (e.g., “if / if you feel like it / while you're at it / you can / no rush”)."
)

OPTIONAL_HINT_TEXTS: list[str] = [
    _OPTIONAL_HINT_EMPTY,
    _OPTIONAL_HINT_PRESENCE,
    _OPTIONAL_HINT_ASIDE,
    _OPTIONAL_HINT_CHARACTER_STATE,
    _OPTIONAL_HINT_CONTEXT_HOOK,
    _OPTIONAL_HINT_MICRO_ACTION_NUDGE,
]

# 不同 slot 下给不同权重；权重越大，越容易被抽中。
# 若 slot 未配置，则回退到 "default"；若配置异常，则最终回退到空字符串。
OPTIONAL_HINT_WEIGHTS_BY_SLOT: dict[str, list[int]] = {
    "default": [3, 5, 5, 5, 5, 5],
    "morning": [3, 5, 5, 5, 8, 5],
    "noon": [3, 5, 10, 10, 5, 3],
    "night": [3, 5, 5, 5, 5, 0],
    "commute": [3, 5, 10, 10, 5, 0],
}


def pick_optional_hint(slot: str) -> str:
    # 1. 获取当前场景的权重，无则用default
    weights = OPTIONAL_HINT_WEIGHTS_BY_SLOT.get(slot) or OPTIONAL_HINT_WEIGHTS_BY_SLOT.get(
        "default"
    )
    # 2. 校验权重：长度不匹配/全负权重 → 返回空值
    if not weights or len(weights) != len(OPTIONAL_HINT_TEXTS):
        return ""
    safe_weights = [max(0, int(w)) for w in weights] # 确保权重非负
    if sum(safe_weights) <= 0:
        return ""
    # 3. 按权重随机抽取1个可选提示
    return random.choices(OPTIONAL_HINT_TEXTS, weights=safe_weights, k=1)[0]

# 构建上下文变量
def build_context_vars(now: datetime, extra: str | None = None) -> str:
    parts: list[str] = []
    parts.append("work day" if now.weekday() < 5 else "weekend") # 判断工作日/周末
    if extra:
        parts.append(str(extra)) # 如果传入额外信息参数，追加额外上下文（如"wake_up_check_result=asleep"）
    return "；".join([p for p in parts if p]) # 拼接非空内容，用分号分隔

# 整合所有参数渲染最终提示词
def build_greeting_style_prompt(
    *,  # 强制关键字参数
    slot: str,  # 时间场景（morning/noon等）
    now: datetime,  # 当前时间（用于判断工作日/周末）
    planned_time: str,  # 计划时刻（HH:MM）
    extra_context: str | None = None,  # 额外上下文
) -> str:
    # 1. 安全处理场景：空值→default
    safe_slot = str(slot or "").strip() or "default"
    # 2. 获取核心意图：无则用"简短问候"
    main_hint = MAIN_HINT_BY_SLOT.get(safe_slot) or "Short greeting"
    # 3. 随机抽取可选提示
    optional_hint = pick_optional_hint(safe_slot)
    # 4. 构建上下文变量
    context_vars = build_context_vars(now, extra_context)
    # 5. 渲染模板：替换所有变量为实际值
    return _SCHEDULED_TASK_TEMPLATE.render(
        NOW_TIME=planned_time or now.strftime("%H:%M"),
        MAIN_HINT=main_hint,
        OPTIONAL_HINTS=optional_hint,
        CONTEXT_VARS=context_vars,
    )

