import json

TAG = __name__


async def handleAbortMessage(conn):
    conn.logger.bind(tag=TAG).info("Abort message received")
    # Abort 在部分客户端可能会“自动发送”（例如 TTS 结束或状态切换时），
    # 因此这里不把它当作“用户侧活动”去刷新 last_user_activity_time，
    # 只在确实处于播报中时才视为用户打断并取消期待回复状态。
    if getattr(conn, "client_is_speaking", False):
        conn.expect_user_reply = False
    # 设置成打断状态，会自动打断llm、tts任务
    conn.client_abort = True
    conn.clear_queues()
    # 打断客户端说话状态
    await conn.websocket.send(
        json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
    )
    conn.clearSpeakStatus()
    conn.logger.bind(tag=TAG).info("Abort message received-end")
