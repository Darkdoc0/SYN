"""
S.Y.N. Brain — LLM Engine
=========================
Orchestrates natural language understanding and text generation for S.Y.N.
Supports rolling chat memory and switches between local Ollama inference and
cloud fallback systems (OpenAI/Anthropic) if local nodes are unavailable.

DESIGN PATTERNS:
1. Adapter/Wrapper: Standardizes interface calls across Ollama, OpenAI, and Anthropic.
2. Chain of Responsibility (Fallback): Tries local offline models first, then falls back to cloud APIs.
3. State Management: Maintains a rolling conversation history queue.
"""

import os
from typing import List, Dict, Any, Generator, Optional
from backend.utils.logger import get_logger
from backend.brain.personality import load_system_prompt
import config

logger = get_logger("LLM_ENGINE")


class ConversationContext:
    """Manages a rolling window of recent conversation turns to conserve memory."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Append user or assistant message to context."""
        self.history.append({"role": role, "content": content})
        # Keep only the last N turns (1 turn = 1 user + 1 assistant message)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        """Compiles system prompt + conversation history for the LLM."""
        return [{"role": "system", "content": system_prompt}] + self.history

    def clear(self):
        """Reset conversation context."""
        self.history.clear()
        logger.info("Conversation history cleared.")


class LLMEngine:
    """
    Language translation and generation engine. Handles API interfaces and
    unreachable host routing.
    """

    def __init__(self):
        self.context = ConversationContext(max_turns=10)
        self._ollama_client = None
        self._openai_client = None
        self._anthropic_client = None

    def _get_openai_client(self):
        """Lazy initializer for OpenAI client."""
        if not self._openai_client:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY environment variable is not set.")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _get_anthropic_client(self):
        """Lazy initializer for Anthropic client."""
        if not self._anthropic_client:
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY environment variable is not set.")
            self._anthropic_client = Anthropic(api_key=api_key)
        return self._anthropic_client

    def generate(self, prompt: str, system_override: Optional[str] = None) -> str:
        """
        Executes a single non-streaming generation query.
        Automatically updates context memory if it's running inside a chat loop.
        """
        # Load system instructions
        system_prompt = system_override if system_override else load_system_prompt()
        
        # Add user message to rolling memory
        self.context.add_message("user", prompt)
        messages = self.context.get_messages(system_prompt)
        
        provider = config.LLM_PROVIDER.lower()
        model = config.LLM_MODEL
        
        reply = ""
        try:
            reply = self._dispatch_generation(provider, model, messages)
        except Exception as e:
            logger.error(f"Provider '{provider}' failed with error: {e}")
            
            # Trigger Fallback Chain
            if provider == "ollama":
                logger.warning("Local Ollama node failed. Attempting cloud fallback (OpenAI/Anthropic)...")
                # Try OpenAI fallback
                try:
                    reply = self._dispatch_generation("openai", "gpt-4o-mini", messages)
                    logger.info("Successfully recovered using OpenAI Cloud Fallback.")
                except Exception as openai_err:
                    logger.error(f"OpenAI fallback also failed: {openai_err}")
                    
                    # Try Anthropic fallback
                    try:
                        reply = self._dispatch_generation("anthropic", "claude-3-5-haiku-20241022", messages)
                        logger.info("Successfully recovered using Anthropic Cloud Fallback.")
                    except Exception as anthropic_err:
                        logger.critical("All language provider nodes failed.")
                        reply = "I'm sorry Boss, but my cognitive nodes are completely offline at the moment."
            else:
                reply = "I encountered an error trying to process that query, Sir."

        # Save assistant response to context memory
        self.context.add_message("assistant", reply)
        return reply

    def _dispatch_generation(self, provider: str, model: str, messages: List[Dict[str, str]]) -> str:
        """Helper to invoke client calls based on provider."""
        if provider == "ollama":
            import ollama
            logger.debug(f"Invoking Ollama inference using model '{model}'")
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS
                }
            )
            return response["message"]["content"].strip()
            
        elif provider == "openai":
            logger.debug(f"Invoking OpenAI Cloud API using model '{model}'")
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
            
        elif provider == "anthropic":
            logger.debug(f"Invoking Anthropic Cloud API using model '{model}'")
            client = self._get_anthropic_client()
            # Extract system prompt from list of messages (Anthropic format constraint)
            system = messages[0]["content"] if messages[0]["role"] == "system" else ""
            anthropic_messages = [m for m in messages if m["role"] != "system"]
            
            response = client.messages.create(
                model=model,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                system=system,
                messages=anthropic_messages
            )
            return response.content[0].text.strip()
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def generate_stream(self, prompt: str, system_override: Optional[str] = None) -> Generator[str, None, None]:
        """
        Executes a streaming generation query.
        Yields chunk words incrementally as they are synthesized.
        Updates context memory after stream completes.
        """
        system_prompt = system_override if system_override else load_system_prompt()
        self.context.add_message("user", prompt)
        messages = self.context.get_messages(system_prompt)
        
        provider = config.LLM_PROVIDER.lower()
        model = config.LLM_MODEL
        
        full_reply = []
        
        # Since we're doing streaming, if the primary fails we catch and attempt fallback
        try:
            for chunk in self._dispatch_stream(provider, model, messages):
                full_reply.append(chunk)
                yield chunk
        except Exception as e:
            logger.error(f"Streaming provider '{provider}' failed: {e}")
            
            # Simple fallback routing (non-streaming fallback for reliability)
            logger.warning("Attempting non-streaming backup response.")
            backup_reply = self.generate(prompt, system_override=system_override)
            yield backup_reply
            return
            
        reply_string = "".join(full_reply).strip()
        self.context.add_message("assistant", reply_string)

    def _dispatch_stream(self, provider: str, model: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Helper to invoke streaming API requests."""
        if provider == "ollama":
            import ollama
            stream = ollama.chat(
                model=model,
                messages=messages,
                stream=True,
                options={
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS
                }
            )
            for chunk in stream:
                yield chunk["message"]["content"]
                
        elif provider == "openai":
            client = self._get_openai_client()
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        elif provider == "anthropic":
            client = self._get_anthropic_client()
            system = messages[0]["content"] if messages[0]["role"] == "system" else ""
            anthropic_messages = [m for m in messages if m["role"] != "system"]
            
            with client.messages.stream(
                model=model,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                system=system,
                messages=anthropic_messages
            ) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            raise ValueError(f"Unknown streaming provider: {provider}")
