"""服务端插件工具执行器"""

import inspect
from typing import Dict, Any
from ..base import ToolType, ToolDefinition, ToolExecutor
from plugins_func.register import all_function_registry, Action, ActionResponse


class ServerPluginExecutor(ToolExecutor):
    """服务端插件工具执行器"""

    def __init__(self, conn):
        self.conn = conn
        self.config = conn.config

    async def execute(
        self, conn, tool_name: str, arguments: Dict[str, Any]
    ) -> ActionResponse:
        """执行服务端插件工具"""
        func_item = all_function_registry.get(tool_name)
        if not func_item:
            return ActionResponse(
                action=Action.NOTFOUND, response=f"插件函数 {tool_name} 不存在"
            )

        try:
            # 根据工具类型决定如何调用
            if hasattr(func_item, "type"):
                func_type = func_item.type
                if func_type.code in [4, 5]:  # SYSTEM_CTL, IOT_CTL (需要conn参数)
                    result = func_item.func(conn, **arguments)
                elif func_type.code == 2:  # WAIT
                    result = func_item.func(**arguments)
                elif func_type.code == 3:  # CHANGE_SYS_PROMPT
                    result = func_item.func(conn, **arguments)
                else:
                    result = func_item.func(**arguments)
            else:
                # 默认不传conn参数
                result = func_item.func(**arguments)

            if inspect.isawaitable(result):
                result = await result
            return result

        except Exception as e:
            return ActionResponse(
                action=Action.ERROR,
                response=str(e),
            )

    def get_tools(self) -> Dict[str, ToolDefinition]:
        """获取所有注册的服务端插件工具"""
        tools = {}

        # 获取必要的函数
        necessary_functions = ["handle_exit_intent", "get_lunar"]

        # 获取配置中的函数
        config_functions = self.config["Intent"][
            self.config["selected_module"]["Intent"]
        ].get("functions", [])

        # 转换为列表
        if not isinstance(config_functions, list):
            try:
                config_functions = list(config_functions)
            except TypeError:
                config_functions = []

        # 合并所有需要的函数
        plugins_cfg = self.config.get("plugins", {}) if isinstance(self.config, dict) else {}
        exposed_functions = []
        if isinstance(plugins_cfg, dict):
            for name, cfg in plugins_cfg.items():
                if not isinstance(cfg, dict):
                    continue
                if cfg.get("expose_to_llm") is True:
                    exposed_functions.append(str(name))

        # confirm_event group exposure (preferred): keep wrappers under plugins.confirm_event.
        # Two knobs:
        # - plugins.confirm_event.expose_to_llm: expose confirm_event itself
        # - plugins.confirm_event.expose_wrappers_to_llm: expose all wrappers (default: True if present but unset)
        confirm_cfg = plugins_cfg.get("confirm_event", {}) if isinstance(plugins_cfg, dict) else {}
        wrappers_cfg = (
            confirm_cfg.get("wrappers", {})
            if isinstance(confirm_cfg, dict)
            else {}
        )
        if not isinstance(wrappers_cfg, dict):
            wrappers_cfg = {}
        wrapper_names = [str(k) for k in wrappers_cfg.keys() if k]
        # Backward compatible default wrapper list when config doesn't declare wrappers.
        if not wrapper_names:
            wrapper_names = ["wake_check"]

        wrappers_expose_default = True
        if isinstance(confirm_cfg, dict) and "expose_wrappers_to_llm" in confirm_cfg:
            wrappers_expose_default = bool(confirm_cfg.get("expose_wrappers_to_llm"))

        if isinstance(confirm_cfg, dict):
            if confirm_cfg.get("disabled") is True:
                wrapper_names = []
            if confirm_cfg.get("expose_to_llm") is True:
                exposed_functions.append("confirm_event")
            for w in wrapper_names:
                w_cfg = wrappers_cfg.get(w, {}) if isinstance(wrappers_cfg, dict) else {}
                if isinstance(w_cfg, dict) and w_cfg.get("disabled") is True:
                    continue
                expose = (
                    bool(w_cfg.get("expose_to_llm"))
                    if isinstance(w_cfg, dict) and "expose_to_llm" in w_cfg
                    else wrappers_expose_default
                )
                if expose:
                    exposed_functions.append(str(w))

        all_required_functions = list(set(necessary_functions + config_functions + exposed_functions))

        for func_name in all_required_functions:
            plugins_cfg = self.config.get("plugins", {}) if isinstance(self.config, dict) else {}
            # Wrapper config lives under plugins.confirm_event.wrappers.<name>
            if func_name in wrapper_names and isinstance(confirm_cfg, dict):
                w_cfg = wrappers_cfg.get(func_name, {}) if isinstance(wrappers_cfg, dict) else {}
                base = confirm_cfg.copy()
                if isinstance(w_cfg, dict):
                    base.update(w_cfg)
                func_cfg = base
            else:
                func_cfg = plugins_cfg.get(func_name, {}) if isinstance(plugins_cfg, dict) else {}
            # Enforce confirm_event exposure via plugins.confirm_event.* knobs, even if it appears
            # in remote intent/plugin lists.
            if func_name == "confirm_event":
                if not (isinstance(confirm_cfg, dict) and confirm_cfg.get("expose_to_llm") is True):
                    continue
            if func_name in wrapper_names:
                w_cfg = wrappers_cfg.get(func_name, {}) if isinstance(wrappers_cfg, dict) else {}
                if isinstance(w_cfg, dict) and "expose_to_llm" in w_cfg:
                    should_expose = bool(w_cfg.get("expose_to_llm"))
                else:
                    should_expose = wrappers_expose_default
                if not should_expose:
                    continue
            if isinstance(func_cfg, dict) and func_cfg.get("disabled") is True:
                continue
            requires_mcp_tool = False
            if isinstance(func_cfg, dict):
                requires_mcp_tool = bool(func_cfg.get("requires_mcp_tool", False))
            # Backward-compatible defaults for existing wrappers
            if func_name == "confirm_event" or func_name in wrapper_names:
                requires_mcp_tool = True

            if requires_mcp_tool:
                vision_tool_name = None
                if isinstance(func_cfg, dict):
                    vision_tool_name = func_cfg.get("tool_name")
                if not vision_tool_name and func_name == "wake_check":
                    scheduled_cfg = (
                        plugins_cfg.get("scheduled_greeting", {})
                        if isinstance(plugins_cfg, dict)
                        else {}
                    )
                    wake_cfg = (
                        (scheduled_cfg.get("wake_check", {}) or {})
                        if isinstance(scheduled_cfg, dict)
                        else {}
                    )
                    if isinstance(wake_cfg, dict):
                        vision_tool_name = wake_cfg.get("tool_name")
                if not vision_tool_name:
                    confirm_cfg = (
                        plugins_cfg.get("confirm_event", {})
                        if isinstance(plugins_cfg, dict)
                        else {}
                    )
                    if isinstance(confirm_cfg, dict):
                        vision_tool_name = confirm_cfg.get("tool_name")
                vision_tool_name = str(vision_tool_name or "vision_assistant")

                mcp_client = getattr(self.conn, "mcp_endpoint_client", None)
                if (
                    not mcp_client
                    or not getattr(mcp_client, "ready", False)
                    or not mcp_client.has_tool(vision_tool_name)
                ):
                    continue

            func_item = all_function_registry.get(func_name)
            if func_item:
                # 从函数注册中获取描述
                fun_description = (
                    self.config.get("plugins", {})
                    .get(func_name, {})
                    .get("description", "")
                )
                if fun_description is not None and len(fun_description) > 0:
                    if "function" in func_item.description and isinstance(
                        func_item.description["function"], dict
                    ):
                        func_item.description["function"][
                            "description"
                        ] = fun_description
                tools[func_name] = ToolDefinition(
                    name=func_name,
                    description=func_item.description,
                    tool_type=ToolType.SERVER_PLUGIN,
                )

        return tools

    def has_tool(self, tool_name: str) -> bool:
        """检查是否有指定的服务端插件工具"""
        return tool_name in all_function_registry
