import logging
from collections import deque
from contextlib import nullcontext
from uuid import uuid4

from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.live import Live
from rich.rule import Rule

from aidu.ai.core.processor_result import ProcessorResult
from aidu.ai.core.artifacts import Artifact, SymbolicArtifact
from aidu.ai.core.recommendation import Recommendation
from aidu.ai.core.context import Context
from aidu.ai.controller.monitor import Monitor
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

    def __init__(self, context: Context = None, show_trace: bool = False):
        self.context = context
        self.monitor = Monitor(context, show_trace)
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

    def run(self, processors: dict, start: str, artifact: Artifact, max_step: int = 10, console: Console = None, cockpit: bool = False) -> Context:
        logger.debug(f"[context] {self.context}")
        mailbox = deque()
        step = 0
        mailbox.append(ProcessorRequested(target=start, artifact=artifact))
        self.context.artifacts[artifact.id] = artifact

        while mailbox:
            event = mailbox.popleft()

            # logger.debug(f"\n[event] {event}")

            if isinstance(event, Stop):
                logger.debug("\n[green]Controller stopped[/green]")
                break

            if isinstance(event, ProcessorRequested):
                processor = processors[event.target]
                logger.debug(f"[event] {event}")

                # ------------------------------------------------------
                # Execute processor
                # ------------------------------------------------------

                agent_context = self.build_agent_context(processor)

                logger.debug(f"[{step}] [agent context] {agent_context}")

                step, res = processor.run(
                    step,
                    event.artifact,
                    context=agent_context,
                    console=console,
                )
                logger.debug(f"[{step}] [result] {res}")

                # ------------------------------------------------------
                # Update current artifact
                # ------------------------------------------------------

                prev_artifact = artifact

                if res.artifacts:
                    artifact = res.artifacts[0]
                else:
                    logger.warning(f"Processor {processor.id} did not return any artifacts; keeping previous artifact")

                # ------------------------------------------------------
                # Select next recommendation
                # ------------------------------------------------------

                r = self.select(res.recommendations)

                # ------------------------------------------------------
                # Update context and mailbox
                # ------------------------------------------------------

                self.context.step = step
                self.context.artifacts[artifact.id] = artifact

                # When having reached max step or having no recommendation, we stop the controller by sending a Stop event to the mailbox.
                if step >= max_step or r is None:
                    mailbox.append(Stop())
                    continue

                mailbox.append(
                    ProcessorRequested(
                        target=r.target,
                        artifact=artifact,
                    )
                )

                if artifact.producer != 'input':
                    console.print(f"[bold blue]{artifact.producer}> [/bold blue]{artifact.content}")

        # Return final artifact via StopIteration.value when iterated to completion.
        return self.context

# ------------------------------------------------------------------------------
# Smoke Test 1
# ------------------------------------------------------------------------------


def _smoke_test_1():

    result = ProcessorResult(
        artifacts=[],
        recommendations=[
            Recommendation(
                target="belief_updater",
                utility=0.9,
            ),
            Recommendation(
                target="tutor",
                utility=0.4,
            ),
        ],
    )

    controller = Controller()

    r = controller.select(result.recommendations)

    assert r.target == "belief_updater"

    print("\n[green]✓ Smoke Test 1 Passed[/green]")


# ------------------------------------------------------------------------------
# Smoke Test 2
# ------------------------------------------------------------------------------


def _smoke_test_2(console: Console):

    controller = Controller()

    processors = {
        "dummy": DummyProcessor(),
    }

    artifact = SymbolicArtifact(
        id="counter_0",
        content=0,
    )

    artifact = controller.run(
        processors=processors,
        start="dummy",
        artifact=artifact,
        max_step=10,
    )

    console.print(Rule("\n[final artifact]"))
    console.print(artifact)

    assert artifact.content == 10

    console.print("\n[green]✓ Smoke Test 2 Passed[/green]")


# ------------------------------------------------------------------------------
# Smoke Test 3
# ------------------------------------------------------------------------------


def _smoke_test_3(console: Console):

    from aidu.ai.agents.math_tutor import MathTutor
    from aidu.ai.agents.symbolic_solver import SymbolicSolver
    from aidu.ai.core.artifacts import TextArtifact
    from aidu.ai.controller.processor import AgentProcessor
    from aidu.ai.llm.clients.openai import OpenAIClient

    context = Context()
    
    controller = Controller(context=context)

    client = OpenAIClient(model="gpt-4o")

    agents = {
        "math_tutor": AgentProcessor(MathTutor(client)),
        "symbolic_solver": AgentProcessor(SymbolicSolver()),
    }

    starting_artifact = TextArtifact(
        id=str(uuid4()),
        type="text",
        producer="starter",
        step=0,
        content="Differentiate 4*x**3",
    )

    # console.print(Rule(f"start: target='math_tutor' receives the problem artifact"))
    # console.print(starting_artifact)

    artifacts = controller.run(
        processors=agents,
        start="math_tutor",
        artifact=starting_artifact,
        max_step=10,
        console=console,
    )

    # console.print(Rule(f"final context"))
    # console.print(artifacts)

    # console.print("\n[green]✓ Smoke Test 3 Passed[/green]")


def _smoke_test_4(console: Console):

    from aidu.ai.agents.math_tutor import MathTutor
    from aidu.ai.agents.chat_bot import ChatBot
    from aidu.ai.agents.symbolic_solver import SymbolicSolver
    from aidu.ai.core.artifacts import TextArtifact
    from aidu.ai.controller.processor import AgentProcessor
    from aidu.ai.llm.clients.openai import OpenAIClient

    context = Context()
    context.control.data["input_mode"] = "interactive"

    controller = Controller(context=context)

    client = OpenAIClient(model="gpt-4o")

    processors = {
        "input": UserInputProcessor("math_tutor"),
        "echo": EchoProcessor(),
        "dummy": DummyProcessor(),
        "chat_bot": AgentProcessor(ChatBot(client)),
        "math_tutor": AgentProcessor(MathTutor(client)),
        "symbolic_solver": AgentProcessor(SymbolicSolver()),
    }

    starting_artifact = SymbolicArtifact(
        id=str(uuid4()),
        type="text",
        producer="starter",
        step=0,
        content="Hi!",
    )

    artifacts = controller.run(
        processors=processors,
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
    # _smoke_test_1()

    # print()

    # _smoke_test_2()

    # print()

    _smoke_test_4(console)
