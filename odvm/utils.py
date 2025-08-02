from typing import Union, Optional
from pathlib import Path
import yaml
import json

from exceptions import ConfigurationError

def load_config(config_path: Optional[Union[str, Path]], default_config: dict) -> dict:
    """Load configuration from file and merge with defaults."""
    config = default_config.copy()
    
    if config_path is None:
        return config
    
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml':
                user_config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                user_config = json.load(f)
            else:
                raise ConfigurationError(f"Unsupported config format: {config_path.suffix}")
            
            # Merge with defaults
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
        return config
    except Exception as e:
        raise ConfigurationError(f"Error loading config: {str(e)}") from e