"""
Model Configuration Loader
Centralized configuration for all agent and tool models.
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

# Default configuration path
DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parent / "resources" / "models.yaml"
)

# Global config cache
_config_cache: Optional[Dict[str, Any]] = None
_config_cache_by_path: Dict[str, tuple] = {}


def load_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default path.
    
    Returns:
        Dictionary with 'agents' and 'tools' keys.
    """
    global _config_cache
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model config file not found: {config_path}\n"
            f"Please create the config file or specify a valid path."
        )

    cache_key = str(Path(config_path).resolve())
    stat = os.stat(cache_key)
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _config_cache_by_path.get(cache_key)
    if cached and cached[0] == stamp:
        return cached[1]
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Validate structure
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a dictionary")
    
    # Set defaults if missing
    if "agents" not in config:
        config["agents"] = {}
    if "tools" not in config:
        config["tools"] = {}
    if "api" not in config:
        config["api"] = {}
    
    # Set default values for agents
    agents_defaults = {
        "analyst": "gpt-6.1-sol",
        "planner": "gpt-6.1-sol",
        "generator": "gpt-6.1-sol",
        "design": "gpt-6.1-sol",
        "grounding": "gpt-6.1-sol",
        "allocation": "gpt-6.1-sol",
    }
    for key, default in agents_defaults.items():
        if key not in config["agents"]:
            config["agents"][key] = default
    
    # Set default values for tools
    tools_defaults = {
        "parse_subtasks": "gpt-6.1-sol",
        "analyst_model": "gpt-6.1-sol",
        "transformability_check": "gpt-6.1-sol",
        "scoring": "gpt-6.1-sol",
        "importance_and_quota": "gpt-6.1-sol",
        "diagnose_allocation": "gpt-6.1-sol",
        "transform_stage1": "gpt-6.1-sol",
        "transform_stage2": "gpt-6.1-sol",
        "verify": "gpt-6.1-sol",
        "default": "gpt-6.1-sol",
    }
    for key, default in tools_defaults.items():
        if key not in config["tools"]:
            config["tools"][key] = default
    
    _config_cache = config
    _config_cache_by_path[cache_key] = (stamp, config)
    return config


def get_agent_model(agent_name: str, config_path: Optional[str] = None) -> str:
    """
    Get model for a specific agent.
    
    Args:
        agent_name: Name of the agent ('analyst', 'planner', 'generator')
        config_path: Optional path to config file
    
    Returns:
        Model name string
    """
    config = load_model_config(config_path)
    return config["agents"].get(agent_name, "gpt-6.1-sol")


def get_output_token_budget(requested: int, config_path: Optional[str] = None) -> int:
    """Allow an experiment to reserve output space for a reasoning model."""
    options = load_model_config(config_path).get('request_options') or {}
    minimum = options.get('min_output_tokens', 0)
    if type(minimum) is not int or minimum < 0:
        raise ValueError('request_options.min_output_tokens must be a nonnegative integer')
    return max(int(requested), minimum)


def get_tool_model(tool_name: str, config_path: Optional[str] = None) -> str:
    """
    Get model for a specific tool.
    
    Args:
        tool_name: Name of the tool (see models.yaml for available names)
        config_path: Optional path to config file
    
    Returns:
        Model name string
    """
    config = load_model_config(config_path)
    return config["tools"].get(tool_name, config["tools"].get("default", "gpt-6.1-sol"))


def get_design_model(config_path: Optional[str] = None) -> str:
    """Get Design Agent model."""
    return get_agent_model("design", config_path)


def get_grounding_model(config_path: Optional[str] = None) -> str:
    """Get Grounding Agent model."""
    return get_agent_model("grounding", config_path)


def get_allocation_model(config_path: Optional[str] = None) -> str:
    """Get Allocation Agent model."""
    return get_agent_model("allocation", config_path)


def _windows_environment_variable(name: str) -> str:
    """Read one persisted Windows variable when this process did not inherit it."""
    import winreg

    locations = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for hive, path in locations:
        try:
            with winreg.OpenKey(hive, path) as key:
                value, kind = winreg.QueryValueEx(key, name)
            if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) and isinstance(value, str) and value.strip():
                return value.strip()
        except OSError:
            continue
    return ""


def get_api_key(config_path: Optional[str] = None) -> str:
    """
    Read the configured key environment variable, or the default key setting.

    On Windows, a variable absent from the process is read from persisted user
    and system settings. An explicit process value (including empty) takes priority.
    """
    config = load_model_config(config_path)
    api = config.get("api") or {}
    key_env = api.get("api_key_env")
    if key_env:
        key = os.getenv(key_env)
        process_value_present = key is not None
        if key is None and os.name == "nt":
            key = _windows_environment_variable(key_env)
        # A provider may explicitly allow its standard environment-variable
        # spelling as an alias. Never fall back to another provider's key, or
        # override an intentionally empty primary process variable.
        if not key and not process_value_present:
            fallbacks = api.get("api_key_env_fallbacks") or []
            if (not isinstance(fallbacks, list)
                    or any(not isinstance(name, str) or not name.strip() for name in fallbacks)):
                raise ValueError("api_key_env_fallbacks must be a list of environment variable names.")
            for name in fallbacks:
                key = os.getenv(name)
                if key is None and os.name == "nt":
                    key = _windows_environment_variable(name)
                if key and key.strip():
                    break
        key = (key or "").strip()
        if not key:
            raise ValueError(
                f"API key environment variable {key_env} is empty or missing. "
                "Set it in the process environment or Windows user/system environment."
            )
        return key
    key = api.get("api_key") or os.getenv("YOUR_LLM_API_KEY_ENV", "")
    return (key if isinstance(key, str) else "") or ""


def get_api_base_url(config_path: Optional[str] = None) -> str:
    """
    Get the API base URL from the configuration or its environment override.
    """
    config = load_model_config(config_path)
    api = config.get("api") or {}
    placeholder = "https://your-llm-gateway.example/v1"
    configured_url = api.get("base_url")
    url = configured_url
    if not configured_url or configured_url == placeholder:
        url = os.getenv("YOUR_LLM_BASE_URL_ENV", "").strip()
    if url == placeholder or (configured_url == placeholder and not url):
        raise ValueError("Set api.base_url to your model API base URL before generation.")
    return (url if isinstance(url, str) else "")
