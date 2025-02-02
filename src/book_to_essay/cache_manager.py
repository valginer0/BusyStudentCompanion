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
        """Generate a hash for a file based on its content and metadata."""
        file_stat = os.stat(file_path)
        metadata = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(metadata.encode()).hexdigest()

    def _get_content_cache_path(self, file_hash: str) -> Path:
        """Get the cache file path for content."""
        return self.content_cache_dir / f"{file_hash}.pkl"

    def _get_model_cache_path(self, prompt_hash: str) -> Path:
        """Get the cache file path for model output."""
        return self.model_cache_dir / f"{prompt_hash}.pkl"

    def get_cached_content(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached content for a file if it exists and is valid."""
        file_hash = self._get_file_hash(file_path)
        cache_path = self._get_content_cache_path(file_hash)
        
        if not cache_path.exists():
            return None
            
        metadata = self.metadata["content"].get(file_hash)
        if not metadata:
            return None
            
        # Check if cache is expired (7 days)
        cache_date = datetime.fromisoformat(metadata["cached_at"])
        if datetime.now() - cache_date > timedelta(days=7):
            return None
            
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    def cache_content(self, file_path: str, content: Dict[str, Any]):
        """Cache processed content from a file."""
        file_hash = self._get_file_hash(file_path)
        cache_path = self._get_content_cache_path(file_hash)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(content, f)
            
        self.metadata["content"][file_hash] = {
            "file_path": file_path,
            "cached_at": datetime.now().isoformat()
        }
        self._save_metadata()

    def get_cached_model_output(self, prompt: str, context: str) -> Optional[str]:
        """Get cached model output if it exists and is valid."""
        prompt_data = f"{prompt}_{context}"
        prompt_hash = hashlib.md5(prompt_data.encode()).hexdigest()
        cache_path = self._get_model_cache_path(prompt_hash)
        
        if not cache_path.exists():
            return None
            
        metadata = self.metadata["model"].get(prompt_hash)
        if not metadata:
            return None
            
        # Check if cache is expired (1 day for model outputs)
        cache_date = datetime.fromisoformat(metadata["cached_at"])
        if datetime.now() - cache_date > timedelta(days=1):
            return None
            
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    def cache_model_output(self, prompt: str, context: str, output: str):
        """Cache model output for a prompt and context."""
        prompt_data = f"{prompt}_{context}"
        prompt_hash = hashlib.md5(prompt_data.encode()).hexdigest()
        cache_path = self._get_model_cache_path(prompt_hash)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(output, f)
            
        self.metadata["model"][prompt_hash] = {
            "cached_at": datetime.now().isoformat()
        }
        self._save_metadata()

    def clear_expired_cache(self):
        """Clear expired cache entries."""
        now = datetime.now()
        
        # Clear expired content cache (7 days)
        for file_hash, metadata in list(self.metadata["content"].items()):
            cache_date = datetime.fromisoformat(metadata["cached_at"])
            if now - cache_date > timedelta(days=7):
                cache_path = self._get_content_cache_path(file_hash)
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata["content"][file_hash]
        
        # Clear expired model cache (1 day)
        for prompt_hash, metadata in list(self.metadata["model"].items()):
            cache_date = datetime.fromisoformat(metadata["cached_at"])
            if now - cache_date > timedelta(days=1):
                cache_path = self._get_model_cache_path(prompt_hash)
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata["model"][prompt_hash]
        
        self._save_metadata()
