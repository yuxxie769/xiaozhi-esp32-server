"""
Test-only: server-initiated TTS push to a connected device.

How it works:
- This module is auto-imported by core/connection.py via auto_import_modules("plugins_func.functions").
- It monkeypatches ConnectionHandler.handle_connection at runtime, without modifying existing files.
- When the target device connects and TTS is ready, it enqueues a small TTS job (FIRST/MIDDLE/LAST)
  which results in audio frames being sent to the client over the existing WebSocket.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
import uuid

from config.logger import setup_logging
from core.providers.tts.dto.dto import ContentType, SentenceType, TTSMessageDTO

# 1. 导入依赖、初始化日志、定义配置常量（ENABLED/TARGET_DEVICE_ID等）
logger = setup_logging()
TAG = __name__

# ---- Edit these for your test ----
ENABLED = False
TARGET_DEVICE_ID = "5a:84:60:e4:12:17"
MODE = "llm_then_tts"  # choices: "direct_tts", "llm_then_tts"
TEST_QUERY = "请用一句话自我介绍。"
PUSH_TEXT = "これはサーバー側から主動的にプッシュされたテスト用音声です。"
WAIT_TTS_READY_SECONDS = 10.0
PUSH_DELAY_SECONDS = 20.0  # wait this long after ready+matched before pushing
DEDUP_SCOPE = "session"  # choices: "session", "device"
PUSH_ONCE = True  # when True, dedup based on DEDUP_SCOPE
PUSH_COOLDOWN_SECONDS = 0.0  # when >0, rate-limit per device_id

# 2. 定义全局变量（用于去重、限流）
_patched = False
_pushed_keys: set[str] = set()
_last_push_ts_by_device: dict[str, float] = {}

# 3. 定义核心函数

# “给主程序的 TTS 引擎下指令”，让它把指定文本转成语音并发送。
def _enqueue_tts(conn, text: str) -> None:
    """
    向TTS队列中添加文本转语音任务，触发音频推送
    :param conn: 设备连接实例（包含TTS队列、会话状态等）
    :param text: 要转语音并推送的文本（比如日语测试文本）
    """
    sentence_id = uuid.uuid4().hex
    conn.sentence_id = sentence_id
    conn.llm_finish_task = True
    conn.close_after_chat = False

    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=sentence_id,
            sentence_type=SentenceType.FIRST,
            content_type=ContentType.ACTION,
        )
    )
    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=sentence_id,
            sentence_type=SentenceType.MIDDLE,
            content_type=ContentType.TEXT,
            content_detail=text,
        )
    )
    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=sentence_id,
            sentence_type=SentenceType.LAST,
            content_type=ContentType.ACTION,
        )
    )


async def _push_when_ready(conn) -> None:
    """
    作用：等待设备就绪后推送TTS音频
    :param conn: ConnectionHandler实例（单个设备的ws会话对象，包含device_id/tts/llm等核心属性）
    """
    # 步骤1：如果插件总开关关闭，直接退出（
    if not ENABLED:
        return

    # 步骤2：等待TTS/LLM就绪（最多等WAIT_TTS_READY_SECONDS秒，默认10秒）
    start = time.time()
    while time.time() - start < WAIT_TTS_READY_SECONDS:
        if getattr(conn, "stop_event", None) and conn.stop_event.is_set(): # 检查设备是否已断开连接
            return
        # 检查核心条件：设备ID已初始化、TTS模块已加载、LLM模块已加载
        has_device_id = bool(getattr(conn, "device_id", None)) 
        has_tts = bool(getattr(conn, "tts", None))
        has_llm = bool(getattr(conn, "llm", None))
        # 所有必要模块就绪，退出等待循环
        if has_device_id and has_tts and (MODE != "llm_then_tts" or has_llm):
            break
        await asyncio.sleep(0.1)

    # 步骤3：校验设备是否为测试目标（只给指定DEVICE_ID推送）
    device_id = getattr(conn, "device_id", None)
    if not device_id or device_id != TARGET_DEVICE_ID:
        return

    # 步骤4：检查当前推送的去重键是否已经被记录在_pushed_keys中，如果是，则不再推送（避免重复推送、高频推送）
    # 4.1 生成去重键（按会话或设备维度去重）
    session_id = getattr(conn, "session_id", "") # 从设备连接实例中获取session_id（
    dedup_key = session_id if DEDUP_SCOPE == "session" else device_id # 根据DEDUP_SCOPE配置选择【会话】或【设备】信息作为去重键
    # 如果开启“只推一次”且去重键已存在，说明在当前去重维度下已经推送过，退出
    # 比如如果去重维度是 session id，那么当前会话内只能推送一次
    if PUSH_ONCE and dedup_key in _pushed_keys:
        return
    # 4.2 频率限流（如果设置了冷却时间，且未到时间，退出）
    if PUSH_COOLDOWN_SECONDS and PUSH_COOLDOWN_SECONDS > 0:
        last_ts = _last_push_ts_by_device.get(device_id, 0.0)
        if time.time() - last_ts < PUSH_COOLDOWN_SECONDS:
            return

    ## 步骤5：可选延迟（避免设备刚连接就推送，默认20秒）
    if PUSH_DELAY_SECONDS and PUSH_DELAY_SECONDS > 0:
        end_at = time.time() + PUSH_DELAY_SECONDS
        while time.time() < end_at:
            if getattr(conn, "stop_event", None) and conn.stop_event.is_set():
                return
            await asyncio.sleep(0.1)

    # 步骤6：执行推送逻辑（核心业务）
    try:
        # 6.1 记录去重键和推送时间
        _pushed_keys.add(dedup_key)
        _last_push_ts_by_device[device_id] = time.time()
        # 6.2 分模式处理推送
        if MODE == "llm_then_tts": # 模式1：先调用LLM生成回复，再转TTS
            query = TEST_QUERY
            logger.bind(tag=TAG).info(
                f"测试推送(LLM->TTS) -> device_id={device_id}, session_id={session_id}, query={query}"
            )
            # 异步执行LLM对话（用线程避免阻塞协程）
            def _run_chat():
                try:
                    # 重置连接状态，保证LLM对话正常执行
                    conn.client_abort = False
                    conn.close_after_chat = False
                    conn.chat(query) # 调用ws会话对象的的LLM对话方法
                except Exception as e:
                    logger.bind(tag=TAG).opt(exception=True).error(
                        f"测试推送(LLM->TTS)失败: {e}"
                    )

            threading.Thread(target=_run_chat, daemon=True).start()
        else: # 模式2：直接推送指定文本的TTS
            text = PUSH_TEXT
            logger.bind(tag=TAG).info(
                f"测试推送(Direct TTS) -> device_id={device_id}, session_id={session_id}, text={text}"
            )
            _enqueue_tts(conn, text)
    except Exception as e:
        logger.bind(tag=TAG).opt(exception=True).error(f"测试推送TTS失败: {e}")

# 安全地给主程序 core.connection 模块中的 ConnectionHandler 类替换它的 handle_connection 方法（设备 WebSocket 连接处理核心方法）
# 这里的目的：给原程序的执行逻辑里面多插一个 push_when_ready 自动推送方法
def _try_patch_connection_handler() -> bool:
    """尝试给ConnectionHandler打补丁，成功返回True，失败返回False"""
    # 检查是否已经完成替换（避免重复补丁）
    global _patched
    if _patched:
        return True

    # 获取/导入主程序的core.connection模块（补丁的目标模块）
    mod = sys.modules.get("core.connection")
    if mod is None:
        try:
            mod = importlib.import_module("core.connection")
        except Exception:
            return False

    # 从模块中获取目标类ConnectionHandler（处理设备连接的核心类）
    handler_cls = getattr(mod, "ConnectionHandler", None)
    if handler_cls is None:
        return False

    # 检查是否已经打过补丁（防重复）
    # 没打过补丁就跳过这个if块，打过（true）就直接返回
    if getattr(handler_cls, "__push_tts_test_patched__", False):
        _patched = True
        return True

    # 备份原方法
    original = handler_cls.handle_connection
    # 定义补丁方法（用来给原方法注入新的补丁内容）
    # 具体来说就是在原方法的基础上提前多跑一个 push_when_ready 方法
    async def patched_handle_connection(self, ws):
        try:
            asyncio.create_task(_push_when_ready(self))# 新建一个_push_when_ready协程（监听设备就绪并推送TTS）
        except Exception:
            pass # 即使启动协程失败，也不影响原方法执行（容错）
        return await original(self, ws)  # 调用原方法

    # 执行补丁方法
    handler_cls.handle_connection = patched_handle_connection
    handler_cls.__push_tts_test_patched__ = True # 更新类的“已补丁”标记
    _patched = True # 更新全局的“已补丁”标记
    logger.bind(tag=TAG).info(
        f"push_tts_test 已加载：ENABLED={ENABLED}, TARGET_DEVICE_ID={TARGET_DEVICE_ID}, MODE={MODE}"
    )
    return True

# 轮询尝试安装补丁
def _bootstrap_patch() -> None:
    for _ in range(200):
        if _try_patch_connection_handler():
            return
        time.sleep(0.1)

# 4. 启动后台补丁线程
threading.Thread(target=_bootstrap_patch, daemon=True).start()
