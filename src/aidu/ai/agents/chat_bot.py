"""
Math tutor agent
"""

import logging
import textwrap

from aidu.ai.llm.agent import LLMAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChatBot(LLMAgent):
    """A chat bot agent which talks to user."""

    # System prompt with flexible placeholders that can be filled via prompt_args
    # Unfilled placeholders will remain as {placeholder} for later customization

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient chat bot.
                  
        """).strip()

    id: str = "chat_bot"
    target: str = "input"
