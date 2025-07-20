"""Token budget management for dynamic compression control."""

from dataclasses import dataclass
from enum import Enum


class PressureLevel(Enum):
    """Memory pressure levels for adaptive compression."""
    LOW = 0.0       # < 50% usage
    MEDIUM = 0.3    # 50-75% usage  
    HIGH = 0.6      # 75-90% usage
    CRITICAL = 0.9  # > 90% usage


@dataclass
class TokenBudget:
    """Current token usage and pressure metrics."""
    max_tokens: int
    current_usage: int
    pressure_level: PressureLevel
    compression_multiplier: float
    
    @property
    def usage_ratio(self) -> float:
        """Calculate current usage percentage."""
        return self.current_usage / self.max_tokens
    
    @property
    def available_tokens(self) -> int:
        """Calculate remaining token budget."""
        return self.max_tokens - self.current_usage


class TokenBudgetManager:
    """Manages token budget and calculates compression pressure."""
    
    def __init__(self, max_tokens: int, safety_buffer: float = 0.05):
        """
        Initialize budget manager.
        
        Args:
            max_tokens: Maximum token limit
            safety_buffer: Reserve percentage (default 5%)
        """
        self.max_tokens = max_tokens
        self.safety_buffer = safety_buffer
        
    def calculate_budget(self, current_usage: int) -> TokenBudget:
        """
        Calculate current budget and pressure level.
        
        Args:
            current_usage: Current token count
            
        Returns:
            TokenBudget with pressure metrics
        """
        usage_ratio = current_usage / self.max_tokens
        pressure_level = self._determine_pressure(usage_ratio)
        
        return TokenBudget(
            max_tokens=self.max_tokens,
            current_usage=current_usage,
            pressure_level=pressure_level,
            compression_multiplier=pressure_level.value
        )
    
    def _determine_pressure(self, usage_ratio: float) -> PressureLevel:
        """Determine pressure level from usage ratio."""
        if usage_ratio >= 0.9:
            return PressureLevel.CRITICAL
        elif usage_ratio >= 0.75:
            return PressureLevel.HIGH
        elif usage_ratio >= 0.5:
            return PressureLevel.MEDIUM
        return PressureLevel.LOW
    
    def should_compress(self, budget: TokenBudget) -> bool:
        """Check if compression is recommended."""
        return budget.pressure_level.value >= PressureLevel.MEDIUM.value
    