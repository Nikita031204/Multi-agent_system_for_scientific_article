"""
LLM utilities for safe API calls with retry logic.
"""

import asyncio
import re
import json
from typing import List, Dict, Any, Optional, Callable
from langchain_core.messages import AIMessage


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 4,
    base_delay: float = 2.0
) -> Any:
    """
    Execute function with exponential backoff on rate limit errors.
    
    Args:
        func: Async function to call
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
    
    Returns:
        Function result
    
    Raises:
        Exception: If all retries exhausted or non-rate-limit error
    """
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit errors
            is_rate_limit = (
                "429" in error_str or 
                "rate-limited" in error_str or 
                "rate limit" in error_str or
                "rate_limit" in error_str or
                "too many requests" in error_str
            )
            
            # Check nested API errors
            if not is_rate_limit:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    is_rate_limit = True
                elif hasattr(e, 'code') and e.code == 429:
                    is_rate_limit = True
            
            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (1.5 ** attempt)
                print(f"    [RETRY] Rate limit, waiting {delay:.1f}s "
                      f"(attempt {attempt+1}/{max_retries})", flush=True)
                await asyncio.sleep(delay)
            elif is_rate_limit:
                print(f"    [RETRY] Rate limit persists after {max_retries} attempts", 
                      flush=True)
                raise
            else:
                raise


def parse_json_response(content: str) -> Optional[Dict]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        content: Raw LLM response text
    
    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        cleaned = content.strip()
        
        # Remove markdown code blocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1].strip()
        
        # Replace single quotes with double quotes for Python dicts
        cleaned = re.sub(r"(?<!\w)'(?!\w)", '"', cleaned)
        
        parsed = json.loads(cleaned)
        
        # Handle list responses
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        
        return parsed
        
    except (json.JSONDecodeError, IndexError):
        return None


def extract_tokens(response) -> int:
    """
    Extract token count from LLM response.
    
    Args:
        response: LLM response object
    
    Returns:
        Token count
    """
    meta = getattr(response, 'response_metadata', {}).get("token_usage", {})
    tokens = meta.get("total_tokens", 0)
    
    if tokens == 0:
        tokens = meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0)
    
    # Fallback: estimate from content length
    if tokens == 0 and hasattr(response, 'content'):
        tokens = len(response.content) // 4
    
    return tokens


def format_action_response(
    parsed_action: Dict,
    agent_name: str,
    tokens_used: int = 0
) -> AIMessage:
    """
    Format action response as AIMessage.
    
    Args:
        parsed_action: Parsed action dictionary
        agent_name: Name of the agent
        tokens_used: Token count (optional)
    
    Returns:
        AIMessage with action content
    """
    action_type = parsed_action.get("type", "wait")
    content = f"Action: {action_type}"
    
    if "order_id" in parsed_action:
        content += f"(order_id={parsed_action['order_id']})"
    if "direction" in parsed_action:
        content += f" -> {parsed_action['direction']}"
    
    return AIMessage(content=content, name=agent_name)
