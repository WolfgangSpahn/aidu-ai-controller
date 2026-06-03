import logging
from collections import deque
from uuid import uuid4

from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler


from aidu.ai.core.artifacts import Artifact, SymbolicArtifact
from aidu.ai.core.context import Context
from aidu.ai.controller.processor import DummyProcessor, EchoProcessor, Processor, UserInputProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------
# Events
# ------------------------------------------------------------------------------


class Event(BaseModel):
    pass


class ProcessorRequested(Event):
    target: str

    artifact: Artifact


class Stop(Event):
    pass


# ------------------------------------------------------------------------------
# Controller
# ------------------------------------------------------------------------------


class Controller:
    """
    Simple event-driven controller.

    Current policy:

        Select recommendation with highest utility.
    """

    def __init__(self, context: Context, processors: dict[str, Processor], show_trace: bool = False):
        self.context = context
        self.mailbox = deque()
        self.show_trace = show_trace
        self.processors = processors

    def select(self, rs):

        if not rs:
            return None

        return max(rs, key=lambda r: r.utility)

    def build_agent_context(self, processor: Processor) -> Context:

        ctx = self.context.model_copy(deep=True)

        if not hasattr(processor, "agent"):
            return ctx

        system_messages = processor.agent.build_system_prompt()

        # `build_system_prompt` returns a list of message dicts; merge
        # them into the trace instead of nesting them under a single
        # system message's `content` field (which breaks provider APIs).
        if isinstance(system_messages, dict):
            system_messages = [system_messages]

        ctx.trace.messages = [
            *system_messages,
            *ctx.trace.messages,
        ]

        return ctx

    def start(self, start: str, artifact: Artifact) -> None:
        self.mailbox.clear()
        self.mailbox.append(ProcessorRequested(target=start, artifact=artifact))
        self.context.artifacts[artifact.id] = artifact
        logger.debug(f"[{self.context.step}] [contr context] {self.context}")

    def step_once(self, processors: dict, max_step: int = 10, console: Console | None = None) -> Event | None:
        if not self.mailbox:
            return None

        event = self.mailbox.popleft()

        if isinstance(event, Stop):
            logger.debug("\n[green]Controller stopped[/green]")
            return event

        if isinstance(event, ProcessorRequested):
            processor = processors[event.target]
            agent_context = self.build_agent_context(processor)

            logger.debug(f"[{self.context.step}] [agent context] {agent_context}")

            step, res = processor.run(
                self.context.step,
                event.artifact,
                context=agent_context,
                console=console,
            )

            logger.debug(f"[{step}] [result] {res}")

            if res.artifacts:
                artifact = res.artifacts[0]
            else:
                logger.warning(f"Processor {processor.id} did not return any artifacts; keeping previous artifact")
                artifact = event.artifact

            recommendation = self.select(res.recommendations)

            self.context.step = step
            self.context.artifacts[artifact.id] = artifact

            if step >= max_step or recommendation is None:
                self.mailbox.append(Stop())
            else:
                self.mailbox.append(
                    ProcessorRequested(
                        target=recommendation.target,
                        artifact=artifact,
                    )
                )

            if console and artifact.producer != "input":
                console.print(f"[bold blue]{artifact.producer}> [/bold blue]{artifact.content}")

            return event

    def run(self, start: str, artifact: Artifact, max_step: int = 10, console: Console = None) -> Context:
        self.start(start, artifact)

        while self.mailbox:
            self.step_once(self.processors, max_step=max_step, console=console)

        return self.context



# ------------------------------------------------------------------------------
# Smoke Test
# ------------------------------------------------------------------------------


def _smoke_test(console: Console):

    from aidu.ai.agents.math_tutor import MathTutor
    from aidu.ai.agents.chat_bot import ChatBot
    from aidu.ai.agents.symbolic_solver import SymbolicSolver
    from aidu.ai.controller.processor import AgentProcessor
    from aidu.ai.llm.clients.openai import OpenAIClient

    context = Context()
    context.control.data["input_mode"] = "interactive"



    client = OpenAIClient(model="gpt-4o-mini")

    processors = {
        "input": UserInputProcessor("math_tutor"),
        "echo": EchoProcessor(),
        "dummy": DummyProcessor(),
        "chat_bot": AgentProcessor(ChatBot(client)),
        "math_tutor": AgentProcessor(MathTutor(client)),
        "symbolic_solver": AgentProcessor(SymbolicSolver()),
    }

    controller = Controller(context=context, processors=processors, show_trace=True)

    starting_artifact = SymbolicArtifact(
        id=str(uuid4()),
        type="text",
        producer="starter",
        step=0,
        content="Hi!",
    )

    controller.run(
        start="input",
        artifact=starting_artifact,
        max_step=10,
        console=console,
    )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    console = Console()
    logging.basicConfig(
        level="WARNING",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
    )

    _smoke_test(console)
