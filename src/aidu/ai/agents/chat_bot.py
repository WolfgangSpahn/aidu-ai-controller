"""
Math tutor agent
"""

import logging
import re
import textwrap
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from aidu.ai.llm.agent import LLMAgent
from aidu.ai.core.context import Context, Message
from aidu.support.regex.validate import assert_valid_sympy_problem


class ChatBot(LLMAgent):
    """A chat bot agent which talks to user."""

    # System prompt with flexible placeholders that can be filled via prompt_args
    # Unfilled placeholders will remain as {placeholder} for later customization

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient chat bot.
                  
        """).strip()

    id: str = "chat_bot"
    target: str = "input"