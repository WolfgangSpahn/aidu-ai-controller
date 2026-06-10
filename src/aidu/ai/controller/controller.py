import logging
import sys
from collections import deque
from uuid import uuid4

from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler


from aidu.ai.core.artifacts import Artifact, SymbolicArtifact
from aidu.ai.core.artifacts import Artifacts
from aidu.ai.core.context import Context

from aidu.ai.llm.agent import Agent, TextArtifact

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------------------
# Events
# ------------------------------------------------------------------------------


class Event(BaseModel):
    pass


class AgentRequested(Event):
    target: type[Agent]
    artifact: Artifacts


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

    def __init__(self, name: str, context: Context, agents: list[type[Agent]]):
        self.name = name
        self.context = context
        self.mailbox = deque()
        self.agents = {agent.__class__: agent() for agent in agents}

    def start(self, start: type[Agent], artifact: Artifact, console: Console | None = None) -> None:
        """
        Initialize a new controller execution.

        This method clears the controller mailbox, stores the initial
        artifact in the global context, and schedules the first
        ``AgentRequested`` event.

        Parameters
        ----------
        start:
            The agent class that should receive the initial artifact.

        artifact:
            The initial artifact that starts the workflow.

        console:
            Optional Rich console used for interactive output.

        Raises
        ------
        AssertionError
            If the specified starting agent is not registered with
            the controller.

        Notes
        -----
        The controller does not execute the starting agent directly.
        Instead, it places an ``AgentRequested`` event into the mailbox.
        Actual execution begins when ``step_once()`` processes that
        event.
        """
        assert start in self.agents, f"Starting agent '{start.__name__}' not found"

        self.mailbox.clear()

        self.mailbox.append(AgentRequested(target=start, artifact=artifact))

        self.context.artifacts[artifact.id] = artifact

        if console:
            console.print(f"[bold red]from {artifact.producer}> [/bold red]{artifact.content}")

    def step_once(self, max_step: int = 10, console: Console | None = None) -> Event | None:
        """
        Execute a single controller step.

        The controller processes the next event from its mailbox and
        updates the global execution context.

        For an ``AgentRequested`` event, the controller:

        1. Retrieves the target agent.
        2. Invokes the agent's ``run()`` method.
        3. Stores any returned artifacts in the context.
        4. Selects the recommendation with the highest utility.
        5. Enqueues the next ``AgentRequested`` event or stops execution.

        Processing terminates when:

        - a ``Stop`` event is encountered,
        - no recommendation is produced,
        - the maximum number of steps is reached,
        - or the recommended target agent is unavailable.

        Parameters
        ----------
        max_step:
            Maximum number of controller steps before execution is
            terminated.

        console:
            Optional Rich console used for interactive output.

        Returns
        -------
        Event | None
            The processed event, or ``None`` if the mailbox is empty.

        Notes
        -----
        The current controller policy selects the recommendation with
        the highest utility value. More advanced routing policies may
        be introduced in future implementations.
        """

        if not self.mailbox:
            return None

        event = self.mailbox.popleft()

        logger.debug(f"[event][{self.name}] [{self.context.step}] {type(event).__name__}: {event}")

        # Handle events

        if isinstance(event, Stop):
            logger.debug(f"[{self.name}] Controller stopped")
            return event

        if isinstance(event, AgentRequested):
            if event.target not in self.agents:
                logger.error(f"[{self.name}] Agent '{event.target.__name__}' not found in {[agent.__name__ for agent in self.agents.keys()]}; stopping")
                self.mailbox.append(Stop())
                return event

            agent = self.agents[event.target]

            # ----------------------------------------------------------------------
            # Run agent
            # ----------------------------------------------------------------------

            result, self.context = agent.run(
                artifact=event.artifact,
                context=self.context,
                agents=list(self.agents.values()),
            )

            # ----------------------------------------------------------------------
            # Process result
            # ----------------------------------------------------------------------

            logger.debug(f"[{self.name}] [{self.context.step}] [result] {result}")

            if result.artifacts:
                artifact = result.artifacts[0]
            else:
                logger.warning(f"Agent {agent.id} did not return any artifacts; keeping previous artifact")
                artifact = event.artifact

            self.context.artifacts[artifact.id] = artifact

            logger.debug(f"[{self.name}] [{self.context.step}] [recommendations] {result.recommendations}")

            recommendation = max(result.recommendations, key=lambda r: r.utility) if result.recommendations else None

            # ----------------------------------------------------------------------
            # Process recommendation and update mailbox
            # ----------------------------------------------------------------------

            if recommendation is None:
                self.mailbox.append(Stop())

            elif recommendation.target not in self.agents:
                logger.error(
                    f"Recommended agent '{recommendation.target.__name__}' "
                    f"of agent '{agent.id}' not found in "
                    f"{[agent.__name__ for agent in self.agents.keys()]} "
                    f"of {self.name}; stopping"
                )
                self.mailbox.append(Stop())

            elif self.context.step >= max_step:
                self.mailbox.append(Stop())

            else:
                self.mailbox.append(
                    AgentRequested(
                        target=recommendation.target,
                        artifact=artifact,
                    )
                )

            if console and artifact.producer != "input":
                console.print(f"[bold blue]{artifact.producer}> [/bold blue]{artifact.content}")

            return event

        logger.warning(f"[{self.name}] Unknown event type: {type(event).__name__}")
        return event

    def run(self, start: type[Agent], artifact: Artifact, max_step: int = 10, console: Console | None = None) -> Context:
        """
        Execute an agent workflow until completion.

        The controller initializes execution with the supplied starting
        agent and artifact, then repeatedly processes mailbox events
        until no further work remains.

        During execution, agents may emit recommendations describing
        which agent should be invoked next. The controller follows the
        recommendation with the highest utility and continues routing
        artifacts between agents until a termination condition is reached.

        Termination occurs when:

        - a ``Stop`` event is encountered,
        - no recommendation is produced,
        - the maximum number of steps is reached,
        - or execution cannot be routed to a valid agent.

        Parameters
        ----------
        start:
            The agent class that should receive the initial artifact.

        artifact:
            The initial artifact that starts the workflow.

        max_step:
            Maximum number of controller steps before execution is
            terminated.

        console:
            Optional Rich console used for interactive output.

        Returns
        -------
        Context
            The final execution context containing the complete trace,
            artifacts, state, and control information accumulated during
            the workflow.

        Notes
        -----
        The controller itself performs no domain reasoning. Its role is
        to manage execution, maintain context, and route artifacts between
        agents according to their recommendations.
        """
        assert start in self.agents, f"Starting agent '{start.__name__}' not found in {[agent.__name__ for agent in self.agents.keys()]}"

        self.start(
            start=start,
            artifact=artifact,
            console=console,
        )

        while self.mailbox:
            self.step_once(
                max_step=max_step,
                console=console,
            )

        return self.context


# ------------------------------------------------------------------------------
# Smoke Test
# ------------------------------------------------------------------------------


def _smoke_test(console: Console):

    from aidu.ai.llm.agent import UserInput
    from aidu.ai.agents.math_tutor import MathTutor
    from aidu.ai.agents.chat_bot import ChatBot
    from aidu.ai.agents.symbolic_solver import SymbolicSolver
    from aidu.ai.llm.clients.openai import OpenAIClient

    context = Context()

    client = OpenAIClient(model="gpt-4o-mini")

    agents = [
        UserInput(),
        ChatBot(client),
        MathTutor(client),
        SymbolicSolver(),
    ]

    controller = Controller(
        name="main_controller",
        context=context,
        agents=agents,
        show_trace=True,
    )

    starting_artifact = TextArtifact(
        producer="starter",
        step=0,
        content="",
    )

    controller.run(
        start=UserInput,
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
