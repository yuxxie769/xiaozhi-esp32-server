import asyncio
from pathlib import Path
from aiohttp import web
from config.logger import setup_logging
from core.api.ota_handler import OTAHandler
from core.api.vision_handler import VisionHandler
from core.api.task_handler import TaskApiHandler
from core.api.state_hub_handler import StateHubApiHandler
from core.console_ui.handler import ConsoleUiHandler
from core.utils.util import get_local_ip

TAG = __name__


class SimpleHttpServer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        self.ota_handler = OTAHandler(config)
        self.vision_handler = VisionHandler(config)
        self.task_handler = TaskApiHandler(config)
        self.state_hub_handler = StateHubApiHandler(config)
        self.console_ui = ConsoleUiHandler()

    def _get_websocket_url(self, local_ip: str, port: int) -> str:
        """获取websocket地址

        Args:
            local_ip: 本地IP地址
            port: 端口号

        Returns:
            str: websocket地址
        """
        server_config = self.config["server"]
        websocket_config = server_config.get("websocket")

        if websocket_config and "你" not in websocket_config:
            return websocket_config
        else:
            return f"ws://{local_ip}:{port}/xiaozhi/v1/"

    async def start(self):
        try:
            server_config = self.config["server"]
            read_config_from_api = self.config.get("read_config_from_api", False)
            host = server_config.get("ip", "0.0.0.0")
            port = int(server_config.get("http_port", 8003))

            if port:
                app = web.Application()

                if not read_config_from_api:
                    # 如果没有开启智控台，只是单模块运行，就需要再添加简单OTA接口，用于下发websocket接口
                    app.add_routes(
                        [
                            web.get("/xiaozhi/ota/", self.ota_handler.handle_get),
                            web.post("/xiaozhi/ota/", self.ota_handler.handle_post),
                            web.options(
                                "/xiaozhi/ota/", self.ota_handler.handle_options
                            ),
                            # 下载接口，仅提供 data/bin/*.bin 下载
                            web.get(
                                "/xiaozhi/ota/download/{filename}",
                                self.ota_handler.handle_download,
                            ),
                            web.options(
                                "/xiaozhi/ota/download/{filename}",
                                self.ota_handler.handle_options,
                            ),
                        ]
                    )
                # 添加路由
                app.add_routes(
                    [
                        # Console UI (static, no auth) - decoupled from task engine internals.
                        web.get("/console", self.console_ui.handle_redirect),
                        web.get("/console/", self.console_ui.handle_index),

                        web.get("/mcp/vision/explain", self.vision_handler.handle_get),
                        web.post(
                            "/mcp/vision/explain", self.vision_handler.handle_post
                        ),
                        web.options(
                            "/mcp/vision/explain", self.vision_handler.handle_options
                        ),
                        # Task engine debug APIs (no auth in MVP)
                        web.get("/tasks", self.task_handler.handle_list),
                        web.post("/tasks", self.task_handler.handle_upsert),
                        web.options("/tasks", self.task_handler.handle_options),
                        web.get("/tasks/{account_id}/all", self.task_handler.handle_list_account),
                        web.get("/tasks/{account_id}/instances", self.task_handler.handle_list_instances),
                        web.get("/tasks/{account_id}/{task_type}", self.task_handler.handle_get),
                        web.delete("/tasks/{account_id}/{task_type}", self.task_handler.handle_delete_task),
                        web.post(
                            "/tasks/{account_id}/{task_type}/kickoff",
                            self.task_handler.handle_kickoff,
                        ),
                        web.post("/tasks/{account_id}/{task_type}/run", self.task_handler.handle_run),
                        web.post("/tasks/{account_id}/{task_type}/pause", self.task_handler.handle_pause),
                        web.post("/tasks/{account_id}/{task_type}/cancel", self.task_handler.handle_cancel),
                        web.get(
                            "/tasks/{account_id}/{task_type}/attempts",
                            self.task_handler.handle_attempts,
                        ),
                        web.delete(
                            "/tasks/{account_id}/{task_type}/instances/{instance_key}",
                            self.task_handler.handle_delete_instance,
                        ),

                        # Legacy endpoints: require ?task_type=... to avoid ambiguous defaults.
                        web.get("/tasks/{account_id}", self.task_handler.handle_get),
                        web.post("/tasks/{account_id}/kickoff", self.task_handler.handle_kickoff),
                        web.post("/tasks/{account_id}/run", self.task_handler.handle_run),
                        web.post("/tasks/{account_id}/pause", self.task_handler.handle_pause),
                        web.post("/tasks/{account_id}/cancel", self.task_handler.handle_cancel),
                        web.get("/tasks/{account_id}/attempts", self.task_handler.handle_attempts),

                        # State Hub APIs (no auth in MVP)
                        web.get("/state_hub/view", self.state_hub_handler.handle_view),
                        web.get("/state_hub/status", self.state_hub_handler.handle_status),
                        web.post("/state_hub/reconnect", self.state_hub_handler.handle_reconnect),
                        web.post("/state_hub/refresh_target", self.state_hub_handler.handle_refresh_target),
                        web.post("/state_hub/exposure", self.state_hub_handler.handle_exposure),
                        web.get("/state_hub/entities", self.state_hub_handler.handle_entities),
                        web.options("/state_hub/view", self.state_hub_handler.handle_options),
                        web.options("/state_hub/status", self.state_hub_handler.handle_options),
                        web.options("/state_hub/reconnect", self.state_hub_handler.handle_options),
                        web.options("/state_hub/refresh_target", self.state_hub_handler.handle_options),
                        web.options("/state_hub/exposure", self.state_hub_handler.handle_options),
                        web.options("/state_hub/entities", self.state_hub_handler.handle_options),
                    ]
                )

                console_dir = Path(__file__).resolve().parent / "console_ui"
                app.router.add_static(
                    "/console/static/",
                    path=str(console_dir),
                    show_index=False,
                )

                # 运行服务
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, host, port)
                await site.start()

                try:
                    urls = []
                    if host in ("0.0.0.0", "::"):
                        urls.append(f"http://127.0.0.1:{port}/console/")
                        local_ip = ""
                        try:
                            local_ip = str(get_local_ip() or "").strip()
                        except Exception:
                            local_ip = ""
                        if local_ip:
                            urls.append(f"http://{local_ip}:{port}/console/")
                    else:
                        urls.append(f"http://{host}:{port}/console/")

                    self.logger.bind(tag=TAG).info(
                        f"ASSIST CONTROL BOARD UI: {', '.join(urls)}"
                    )
                except Exception:
                    pass

                # 保持服务运行
                while True:
                    await asyncio.sleep(3600)  # 每隔 1 小时检查一次
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"HTTP服务器启动失败: {e}")
            import traceback

            self.logger.bind(tag=TAG).error(f"错误堆栈: {traceback.format_exc()}")
            raise
