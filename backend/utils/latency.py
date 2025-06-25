"""
Latency logging utilities for performance monitoring.

This module provides:
- LatencyLogger: Simple latency logging utility
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LatencyLogger:
    """Simple latency logging utility"""
    
    def __init__(self):
        self.log_file = Path("logs") / f"latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log_event(self, event_type: str, data: Dict[str, Any], latency_ms: float = None):
        """Log an event with optional latency"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "latency_ms": latency_ms,
            "data": data
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        logger.info(f"{event_type}: {latency_ms}ms" if latency_ms else f"{event_type}")