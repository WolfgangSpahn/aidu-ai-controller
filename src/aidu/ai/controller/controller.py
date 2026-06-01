import logging
from collections import deque
from uuid import uuid4

from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule

from aidu.ai.core.processor_result import ProcessorResult
from aidu.ai.core.artifacts import Artifact, SymbolicArtifact
from aidu.ai.core.recommendation import Recommendation
from aidu.ai.core.context import Context
from aidu.ai.controller.processor import Processor

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
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

    def __init__(self, context: Context = None):
        self.context = context

    def select(self, rs):

        if not rs:
            return None

        return max(rs, key=lambda r: r.utility)

    def build_agent_context(self, processor: Processor) -> Context:

        ctx = self.context.model_copy(deep=True)

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

    def run(self, processors: dict, start: str, artifact: Artifact, max_steps: int = 10, console: Console = None):

        mailbox = deque()

        mailbox.append(ProcessorRequested(target=start, artifact=artifact))

        steps = 0

        # This method is a generator: yield per-step results so callers can
        # pretty-print or inspect each processing step (used by smoke tests).
        while mailbox:
            event = mailbox.popleft()

            logger.debug(f"\n[event] {event}")

            if isinstance(event, Stop):
                logger.debug("\n[green]Controller stopped[/green]")
                break

            if isinstance(event, ProcessorRequested):
                processor = processors[event.target]

                # ------------------------------------------------------
                # Execute processor
                # ------------------------------------------------------
                agent_context = self.build_agent_context(processor)

                res = processor.run(
                    event.artifact,
                    context=agent_context,
                    console=console,
                )
                logger.debug(f"\n[result] {res}")

                # ------------------------------------------------------
                # Update current artifact
                # ------------------------------------------------------

                prev_artifact = artifact

                if res.artifacts:
                    artifact = res.artifacts[0]

                # yield the previous artifact and the ProcessorResult so callers
                # can compute and pretty-print diffs between before/after.

                # ------------------------------------------------------
                # Select next recommendation
                # ------------------------------------------------------

                r = self.select(res.recommendations)
                yield (steps, r, res)

                steps += 1

                if steps >= max_steps:
                    mailbox.append(Stop())

                    continue

                if r is None:
                    mailbox.append(Stop())

                    continue

                mailbox.append(
                    ProcessorRequested(
                        target=r.target,
                        artifact=artifact,
                    )
                )

                # ---------------------------------------------------------------
                # Update self.context
                # ---------------------------------------------------------------

                # context.trace

                self.context.trace.message.append({'role':'','content':''})
                # context.state

                # context.control

        # Return final artifact via StopIteration.value when iterated to completion.
        return artifact


# ------------------------------------------------------------------------------
# Dummy Processor
# ------------------------------------------------------------------------------


class DummyProcessor:
    consumes = [SymbolicArtifact]

    produces = [SymbolicArtifact]

    def run(
        self,
        artifact: SymbolicArtifact,
    ) -> ProcessorResult:

        value = artifact.content + 1

        out = SymbolicArtifact(
            id=f"counter_{value}",
            content=value,
        )

        return ProcessorResult(
            artifacts=[out],
            recommendations=[
                Recommendation(
                    target="dummy",
                    utility=1.0,
                    rationale="increment again",
                )
            ],
        )


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
        max_steps=10,
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
        content="Differentiate 4*x**3",
    )

    console.print(Rule(f"start: target='math_tutor' receives the problem artifact"))
    console.print(starting_artifact)

    for count, r, step in controller.run(
        processors=agents,
        start="math_tutor",
        artifact=starting_artifact,
        max_steps=3,
        #console=console,
    ):
        pass
        # console.print(Rule(f"step {count}: {r.target if r else 'no recommendation'}"))
        # console.print(step)

    console.print("\n[green]✓ Smoke Test 3 Passed[/green]")


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

    _smoke_test_3(console)
