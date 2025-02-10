"""Cache manager for storing processed content and model outputs."""
import os
import json
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path
import pickle
from datetime import datetime, timedelta

class CacheManager:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.content_cache_dir = self.cache_dir / "content"
        self.content_cache_dir.mkdir(exist_ok=True)
        self.model_cache_dir = self.cache_dir / "model"
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Load cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"content": {}, "model": {}}

    def _save_metadata(self):
        """Save cache metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file based on its content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            print(f"Error reading file for hashing: {e}")
            # Fallback to filename-based hash if file can't be read
            return hashlib.md5(os.path.basename(file_path).encode()).hexdigest()

    def _get_content_cache_path(self, file_hash: str) -> Path:
        """Get the cache file path for content."""
        return self.content_cache_dir / f"{file_hash}.pkl"

    def _get_model_cache_path(self, prompt_hash: str) -> Path:
        """Get the cache file path for model output."""
        return self.model_cache_dir / f"{prompt_hash}.pkl"

    def get_cached_content(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached content for a file if it exists and is valid."""
        try:
            file_hash = self._get_file_hash(file_path)
            cache_path = self._get_content_cache_path(file_hash)
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error accessing cache for {file_path}: {e}")
        return None

    def cache_content(self, file_path: str, content: Dict[str, Any]):
        """Cache processed content for a file."""
        try:
            file_hash = self._get_file_hash(file_path)
            cache_path = self._get_content_cache_path(file_hash)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(content, f)
            
            self.metadata["content"][file_path] = {
                "hash": file_hash,
                "timestamp": datetime.now().isoformat()
            }
            self._save_metadata()
        except Exception as e:
            print(f"Error caching content for {file_path}: {e}")

    def get_cached_model_output(self, prompt: str, context: str) -> Optional[str]:
        """Get cached model output if it exists and is valid."""
        try:
            prompt_data = f"{prompt}_{context}"
            prompt_hash = hashlib.md5(prompt_data.encode()).hexdigest()
            cache_path = self._get_model_cache_path(prompt_hash)
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error accessing cache for {prompt} and {context}: {e}")
        return None

    def cache_model_output(self, prompt: str, context: str, output: str):
        """Cache model output for a prompt and context."""
        try:
            prompt_data = f"{prompt}_{context}"
            prompt_hash = hashlib.md5(prompt_data.encode()).hexdigest()
            cache_path = self._get_model_cache_path(prompt_hash)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(output, f)
            
            self.metadata["model"][prompt_hash] = {
                "timestamp": datetime.now().isoformat()
            }
            self._save_metadata()
        except Exception as e:
            print(f"Error caching model output for {prompt} and {context}: {e}")

    def clear_expired_cache(self):
        """Clear expired cache entries."""
        now = datetime.now()
        
        # Clear expired content cache (7 days)
        for file_path, metadata in list(self.metadata["content"].items()):
            cache_date = datetime.fromisoformat(metadata["timestamp"])
            if now - cache_date > timedelta(days=7):
                cache_path = self._get_content_cache_path(metadata["hash"])
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata["content"][file_path]
        
        # Clear expired model cache (1 day)
        for prompt_hash, metadata in list(self.metadata["model"].items()):
            cache_date = datetime.fromisoformat(metadata["timestamp"])
            if now - cache_date > timedelta(days=1):
                cache_path = self._get_model_cache_path(prompt_hash)
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata["model"][prompt_hash]
        
        self._save_metadata()
