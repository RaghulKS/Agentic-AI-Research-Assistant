"""
Central configuration management for all API keys and settings.
All secrets and configuration should be managed through this file.
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv


class Settings:
    """
    Centralized settings management for the Agentic Research Assistant.
    All API keys, secrets, and configuration parameters are managed here.
    """
    
    def __init__(self):
        # Load environment variables from .env file
        self._load_environment()
        
        # Core API Keys (Required)
        self.openai_api_key = self._get_required_setting("OPENAI_API_KEY")
        
        # Optional API Keys
        self.copyleaks_api_key = self._get_optional_setting("COPYLEAKS_API_KEY")
        self.copyleaks_api_email = self._get_optional_setting("COPYLEAKS_API_EMAIL")
        
        # Model Configuration
        self.model_name = self._get_setting("MODEL_NAME", "gpt-4o")
        self.temperature = float(self._get_setting("TEMPERATURE", "0.3"))
        self.max_tokens = int(self._get_setting("MAX_TOKENS", "4000"))
        
        # Research Configuration
        self.default_max_results = int(self._get_setting("DEFAULT_MAX_RESULTS", "5"))
        self.timeout_seconds = int(self._get_setting("TIMEOUT_SECONDS", "20"))
        self.max_content_length = int(self._get_setting("MAX_CONTENT_LENGTH", "8000"))
        
        # Framework Configuration
        self.default_framework = self._get_setting("DEFAULT_FRAMEWORK", "hybrid")
        self.verbose_logging = self._get_setting("VERBOSE_LOGGING", "false").lower() == "true"
        self.enable_memory = self._get_setting("ENABLE_MEMORY", "false").lower() == "true"
        
        # Output Configuration
        self.reports_dir = self._get_setting("REPORTS_DIR", "reports")
        self.data_dir = self._get_setting("DATA_DIR", "data")
        self.logs_dir = self._get_setting("LOGS_DIR", "logs")
        
        # Plagiarism Detection Configuration
        self.originality_threshold = float(self._get_setting("ORIGINALITY_THRESHOLD", "0.8"))
        self.similarity_threshold = float(self._get_setting("SIMILARITY_THRESHOLD", "0.7"))
        
        # User Agent for Web Requests
        self.user_agent = self._get_setting(
            "USER_AGENT", 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
        
        # Validate critical settings
        self._validate_settings()
    
    def _load_environment(self):
        """Load environment variables from .env file with error handling"""
        env_path = Path(".env")
        if env_path.exists():
            try:
                load_dotenv(env_path)
            except Exception as e:
                print(f"Warning: Could not load .env file: {e}")
                print("Falling back to system environment variables")
    
    def _get_required_setting(self, key: str) -> str:
        """Get a required setting, raise error if not found"""
        value = os.getenv(key)
        if not value:
            raise ValueError(
                f"Required setting '{key}' not found. "
                f"Please set this in your .env file or environment variables."
            )
        return value.strip()
    
    def _get_optional_setting(self, key: str) -> Optional[str]:
        """Get an optional setting, return None if not found"""
        value = os.getenv(key)
        return value.strip() if value else None
    
    def _get_setting(self, key: str, default: str) -> str:
        """Get a setting with a default value"""
        value = os.getenv(key, default)
        return value.strip()
    
    def _validate_settings(self):
        """Validate critical settings"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required but not provided")
        
        if not self.openai_api_key.startswith("sk-"):
            raise ValueError("OpenAI API key appears to be invalid (should start with 'sk-')")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        if self.max_tokens < 100:
            raise ValueError("Max tokens must be at least 100")
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration dictionary"""
        return {
            "api_key": self.openai_api_key,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def get_copyleaks_config(self) -> dict:
        """Get Copyleaks configuration dictionary"""
        return {
            "api_key": self.copyleaks_api_key,
            "email": self.copyleaks_api_email,
            "enabled": bool(self.copyleaks_api_key and self.copyleaks_api_email)
        }
    
    def get_web_config(self) -> dict:
        """Get web scraping configuration"""
        return {
            "user_agent": self.user_agent,
            "timeout_seconds": self.timeout_seconds,
            "max_content_length": self.max_content_length
        }
    
    def get_research_config(self) -> dict:
        """Get research configuration"""
        return {
            "default_max_results": self.default_max_results,
            "originality_threshold": self.originality_threshold,
            "similarity_threshold": self.similarity_threshold
        }
    
    def get_paths_config(self) -> dict:
        """Get directory paths configuration"""
        return {
            "reports_dir": self.reports_dir,
            "data_dir": self.data_dir,
            "logs_dir": self.logs_dir
        }
    
    def is_copyleaks_enabled(self) -> bool:
        """Check if Copyleaks integration is enabled"""
        return bool(self.copyleaks_api_key and self.copyleaks_api_email)
    
    def display_config_status(self):
        """Display configuration status for debugging"""
        print("🔧 Configuration Status:")
        print(f"   ✅ OpenAI API Key: {'Set' if self.openai_api_key else '❌ Missing'}")
        print(f"   📊 Model: {self.model_name}")
        print(f"   🔍 Copyleaks: {'Enabled' if self.is_copyleaks_enabled() else 'Disabled'}")
        print(f"   🚀 Framework: {self.default_framework}")
        print(f"   📝 Verbose: {self.verbose_logging}")
        print(f"   🧠 Memory: {self.enable_memory}")


# Global settings instance
settings = Settings()


# Convenience functions for backward compatibility
def get_openai_api_key() -> str:
    """Get OpenAI API key"""
    return settings.openai_api_key


def get_model_name() -> str:
    """Get the model name"""
    return settings.model_name


def get_copyleaks_config() -> dict:
    """Get Copyleaks configuration"""
    return settings.get_copyleaks_config()


def get_web_headers() -> dict:
    """Get web request headers"""
    return {"User-Agent": settings.user_agent}


def validate_api_keys():
    """Validate that required API keys are present"""
    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key is required. Please set OPENAI_API_KEY in your .env file."
        )
