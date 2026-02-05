import os
import yaml
from collections.abc import Mapping
from config.manage_api_client import init_service, get_server_config, get_agent_models


def get_project_dir():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_config():
    """加载配置文件"""
    from core.utils.cache.manager import cache_manager, CacheType

    # 检查缓存
    cached_config = cache_manager.get(CacheType.CONFIG, "main_config")
    if cached_config is not None:
        return cached_config

    default_config_path = get_project_dir() + "config.yaml"
    custom_config_path = get_project_dir() + "data/.config.yaml"

    # 加载默认配置
    default_config = read_config(default_config_path)
    custom_config = read_config(custom_config_path)

    if custom_config.get("manager-api", {}).get("url"):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，使用异步版本
            config = asyncio.run_coroutine_threadsafe(
                get_config_from_api_async(custom_config), loop
            ).result()
        except RuntimeError:
            # 如果不在事件循环中（启动时），创建新的事件循环
            config = asyncio.run(get_config_from_api_async(custom_config))
    else:
        # 合并配置
        config = merge_configs(default_config, custom_config)
    # 初始化目录
    ensure_directories(config)

    # 缓存配置
    cache_manager.set(CacheType.CONFIG, "main_config", config)
    return config


async def get_config_from_api_async(config):
    """从Java API获取配置（异步版本）"""
    # 初始化API客户端
    init_service(config)

    # 获取服务器配置
    config_data = await get_server_config()
    if config_data is None:
        raise Exception("Failed to fetch server config from API")

    config_data["read_config_from_api"] = True
    config_data["manager-api"] = {
        "url": config["manager-api"].get("url", ""),
        "secret": config["manager-api"].get("secret", ""),
    }
    auth_enabled = config_data.get("server", {}).get("auth", {}).get("enabled", False)
    # server的配置以本地为准
    if config.get("server"):
        config_data["server"] = {
            "ip": config["server"].get("ip", ""),
            "port": config["server"].get("port", ""),
            "http_port": config["server"].get("http_port", ""),
            "vision_explain": config["server"].get("vision_explain", ""),
            "auth_key": config["server"].get("auth_key", ""),
        }
    config_data["server"]["auth"] = {"enabled": auth_enabled}
    # Merge local plugin configs from data/.config.yaml.
    # Keep scope minimal: only *add* missing keys, don't override manager settings.
    local_plugins = config.get("plugins", {}) if isinstance(config, dict) else {}
    if isinstance(local_plugins, dict) and local_plugins:
        if not isinstance(config_data.get("plugins"), dict):
            config_data["plugins"] = {}

        def _merge_missing(dst, src):
            if not isinstance(dst, dict) or not isinstance(src, dict):
                return dst
            for k, v in src.items():
                if k not in dst:
                    dst[k] = v
                    continue
                if isinstance(dst.get(k), dict) and isinstance(v, dict):
                    _merge_missing(dst[k], v)
            return dst

        for plugin_key, plugin_cfg in local_plugins.items():
            if plugin_key not in config_data["plugins"]:
                config_data["plugins"][plugin_key] = plugin_cfg
                continue
            if isinstance(config_data["plugins"].get(plugin_key), dict) and isinstance(plugin_cfg, dict):
                _merge_missing(config_data["plugins"][plugin_key], plugin_cfg)

    # Merge local debug switches from data/.config.yaml so operators can enable verbose logging
    # without requiring manager-api support for these fields.
    local_debug = config.get("debug", {}) if isinstance(config, dict) else {}
    if isinstance(local_debug, dict) and local_debug:
        config_data["debug"] = merge_configs(config_data.get("debug", {}) or {}, local_debug)

    # Merge local Intent functions (additive only) so operators can extend tool availability
    # without requiring manager-api support for newly added plugin tools.
    #
    # Note: manager-api configs may use module IDs like "Intent_function_call" as the key under "Intent",
    # while local examples often use "function_call". We map local functions into the active remote key.
    local_intent = config.get("Intent", {}) if isinstance(config, dict) else {}
    if isinstance(local_intent, dict) and local_intent:
        if not isinstance(config_data.get("Intent"), dict):
            config_data["Intent"] = {}

        active_intent_key = (config_data.get("selected_module") or {}).get("Intent")
        active_intent_key = str(active_intent_key or "").strip() or None

        def _get_functions_from_intent_key(key: str) -> list[str]:
            cfg = local_intent.get(key)
            if not isinstance(cfg, dict):
                return []
            fns = cfg.get("functions")
            if not isinstance(fns, list):
                return []
            return [str(x) for x in fns if x]

        local_functions: list[str] = []
        if active_intent_key:
            # Prefer exact match first.
            local_functions = _get_functions_from_intent_key(active_intent_key)
            # Then try common aliases.
            if not local_functions:
                if "function_call" in active_intent_key and "function_call" in local_intent:
                    local_functions = _get_functions_from_intent_key("function_call")
                elif "intent_llm" in active_intent_key and "intent_llm" in local_intent:
                    local_functions = _get_functions_from_intent_key("intent_llm")
                elif "nointent" in active_intent_key and "nointent" in local_intent:
                    local_functions = _get_functions_from_intent_key("nointent")

        # Fallback: union everything if we still couldn't determine.
        if not local_functions:
            union: list[str] = []
            for k in list(local_intent.keys()):
                for fn in _get_functions_from_intent_key(str(k)):
                    if fn not in union:
                        union.append(fn)
            local_functions = union

        if local_functions and active_intent_key:
            if active_intent_key not in config_data["Intent"] or not isinstance(config_data["Intent"].get(active_intent_key), dict):
                config_data["Intent"][active_intent_key] = {}
            remote_functions = config_data["Intent"][active_intent_key].get("functions")
            if not isinstance(remote_functions, list):
                remote_functions = []
            for fn in local_functions:
                if fn not in remote_functions:
                    remote_functions.append(fn)
            config_data["Intent"][active_intent_key]["functions"] = remote_functions
    # 如果服务器没有prompt_template，则从本地配置读取
    if not config_data.get("prompt_template"):
        config_data["prompt_template"] = config.get("prompt_template")
    return config_data


async def get_private_config_from_api(config, device_id, client_id):
    """从Java API获取私有配置"""
    return await get_agent_models(device_id, client_id, config["selected_module"])


def ensure_directories(config):
    """确保所有配置路径存在"""
    dirs_to_create = set()
    project_dir = get_project_dir()  # 获取项目根目录
    # 日志文件目录
    log_dir = config.get("log", {}).get("log_dir", "tmp")
    dirs_to_create.add(os.path.join(project_dir, log_dir))

    # ASR/TTS模块输出目录
    for module in ["ASR", "TTS"]:
        if config.get(module) is None:
            continue
        for provider in config.get(module, {}).values():
            output_dir = provider.get("output_dir", "")
            if output_dir:
                dirs_to_create.add(output_dir)

    # 根据selected_module创建模型目录
    selected_modules = config.get("selected_module", {})
    for module_type in ["ASR", "LLM", "TTS"]:
        selected_provider = selected_modules.get(module_type)
        if not selected_provider:
            continue
        if config.get(module) is None:
            continue
        if config.get(selected_provider) is None:
            continue
        provider_config = config.get(module_type, {}).get(selected_provider, {})
        output_dir = provider_config.get("output_dir")
        if output_dir:
            full_model_dir = os.path.join(project_dir, output_dir)
            dirs_to_create.add(full_model_dir)

    # 统一创建目录（保留原data目录创建）
    for dir_path in dirs_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except PermissionError:
            print(f"警告：无法创建目录 {dir_path}，请检查写入权限")


def merge_configs(default_config, custom_config):
    """
    递归合并配置，custom_config优先级更高

    Args:
        default_config: 默认配置
        custom_config: 用户自定义配置

    Returns:
        合并后的配置
    """
    if not isinstance(default_config, Mapping) or not isinstance(
        custom_config, Mapping
    ):
        return custom_config

    merged = dict(default_config)

    for key, value in custom_config.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
