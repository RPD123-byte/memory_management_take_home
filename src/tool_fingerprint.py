"""
Tool fingerprinting module for deduplication of exact repeated tool calls.

This module provides functionality to create unique fingerprints for tool executions
based on their name and parameters, enabling detection of exact duplicate calls.
"""

import hashlib
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolFingerprint:
    """Represents a unique fingerprint for a tool execution"""
    fingerprint: str
    tool_name: str
    parameters: Dict[str, Any]
    normalized_params: str
    created_at: str
    
    
class ToolFingerprintService:
    """Service for generating and managing tool fingerprints"""
    
    def __init__(self):
        """Initialize the fingerprint service"""
        self.fingerprint_cache = {}  # fingerprint -> tool_id mapping
        
    def generate_fingerprint(self, tool_name: str, parameters: Dict[str, Any]) -> ToolFingerprint:
        """
        Generate a unique fingerprint for a tool execution.
        
        The fingerprint is based on:
        - Tool name (action_type)
        - Normalized parameters (sorted, whitespace-stripped)
        
        Args:
            tool_name: Name of the tool/action
            parameters: Tool parameters/arguments
            
        Returns:
            ToolFingerprint object with unique hash
        """
        # Normalize parameters for consistent hashing
        normalized = self._normalize_parameters(parameters)
        
        # Create fingerprint string
        fingerprint_str = f"{tool_name}::{normalized}"
        
        # Generate hash
        fingerprint_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
        
        return ToolFingerprint(
            fingerprint=fingerprint_hash,
            tool_name=tool_name,
            parameters=parameters,
            normalized_params=normalized,
            created_at=datetime.now().isoformat()
        )
    
    def _normalize_parameters(self, params: Dict[str, Any]) -> str:
        """
        Normalize parameters for consistent fingerprinting.
        
        - Sorts keys alphabetically
        - Strips whitespace from string values
        - Handles nested dictionaries and lists
        - Ignores timestamp/metadata fields
        
        Args:
            params: Raw parameters
            
        Returns:
            Normalized JSON string
        """
        # Fields to ignore in fingerprinting (temporal/metadata)
        ignore_fields = {'timestamp', 'created_at', 'updated_at', 'workflow_id'}
        
        def normalize_value(value: Any) -> Any:
            """Recursively normalize values"""
            if isinstance(value, dict):
                return {k: normalize_value(v) for k, v in sorted(value.items()) 
                       if k not in ignore_fields}
            elif isinstance(value, list):
                return [normalize_value(item) for item in value]
            elif isinstance(value, str):
                return value.strip()
            else:
                return value
        
        normalized = normalize_value(params)
        return json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    
    def check_duplicate(self, fingerprint: str) -> Optional[str]:
        """
        Check if a fingerprint has been seen before.
        
        Args:
            fingerprint: Fingerprint hash to check
            
        Returns:
            Tool ID if duplicate found, None otherwise
        """
        return self.fingerprint_cache.get(fingerprint)
    
    def register_fingerprint(self, fingerprint: str, tool_id: str):
        """
        Register a fingerprint with its tool ID.
        
        Args:
            fingerprint: Fingerprint hash
            tool_id: Associated tool result ID
        """
        self.fingerprint_cache[fingerprint] = tool_id
    
    def is_duplicate_call(self, tool_name: str, parameters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Check if this tool call is a duplicate of a previous execution.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            
        Returns:
            Tuple of (is_duplicate, previous_tool_id)
        """
        fp = self.generate_fingerprint(tool_name, parameters)
        previous_tool_id = self.check_duplicate(fp.fingerprint)
        
        return (previous_tool_id is not None, previous_tool_id)
    
    def clear_cache(self):
        """Clear the fingerprint cache"""
        self.fingerprint_cache.clear()
