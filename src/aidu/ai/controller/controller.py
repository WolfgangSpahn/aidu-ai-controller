from copy import deepcopy
import logging
import sys
from collections import deque
from uuid import uuid4

from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler


from aidu.ai.core.artifacts import Artifact, SymbolicArtifact
from aidu.ai.core.context import Context, State

from aidu.ai.llm.agent import Agent, EndArtifact, TextArtifact

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------------------
# Events
# ------------------------------------------------------------------------------

class Event(BaseModel):
    pass


class AgentRequested(Event):
    target: type[Agent]
    artifact: Artifact
    continuations: list[type[Agent]] = []

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

    def __init__(self, name: str, agents: list[type[Agent]]):
        self.name = name
        mailbox = deque()
        self.agents_map = {agent.__class__: agent for agent in agents}

    def build_agent_context(self, agent: Agent, context) -> Context:
        """
        Build the local context perceived by an agent.

        The initial implementation exposes only:

        - the global trace of messages (as context.trace.messages)
        - the global artifacts (as context.artifacts)
        - the global state (as context.state)
        - the global control variables (as context.control)

        Later versions may expose/filter out artifacts, beliefs,
        observations, state variables, or other contextual information.
        """
        logger.debug(f"Global Messages: {context.trace.messages}")
        new_context = deepcopy(context)

        if not new_context.trace.messages:
            logger.debug(f"Global trace is empty; build system prompt for agent {agent.__class__.__name__} without global messages")
            if hasattr(agent, "build_system_prompt"):
                prompt_params = new_context.state.data[agent.__class__.__name__]
                new_context.trace.messages = agent.build_system_prompt(prompt_params=prompt_params)
            else:
                new_context.trace.messages = [{"role": "system", "content": f"Dummy system prompt for {agent.__class__.__name__}"}]
            return new_context


        logger.debug(new_context.trace.messages)
        # check if context.trace.messages[] has role system
        old_system_message = new_context.trace.messages[0]
        assert old_system_message["role"] == "system", "First message in trace is not a system message; agent may not receive expected system prompt"


        if hasattr(agent, "build_system_prompt"):
            prompt_params = new_context.state.data[agent.__class__.__name__]
            new_context.trace.messages[0] = agent.build_system_prompt(prompt_params=prompt_params)[0]

        return new_context

    def merge_context(self, global_context: Context, local_context: Context) -> Context:
        """
        Merge an agent-local context back into the controller context.

        The initial policy is deliberately simple: the local context becomes
        the new global context. Later versions may selectively merge trace,
        artifacts, state, control, evidence, and beliefs.
        """
        return deepcopy(local_context)

    def start(self, start: type[Agent], mailbox, context, artifact: Artifact, console: Console | None = None) -> None:
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
        assert start in self.agents_map, f"Starting agent '{start.__name__}' not found"
        context.check_agents_have_state(self.agents_map.values())
        mailbox.clear()

        mailbox.append(AgentRequested(target=start, artifact=artifact))

        context.artifacts[artifact.id] = artifact

        if console:
            console.print(f"[bold red]from {artifact.producer}> [/bold red]{artifact.content}")

        return mailbox, context

    def step_once(self, mailbox, context, max_step: int = 10, console: Console | None = None) -> tuple[Event | None, Context]:
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
        context.check_agents_have_state(self.agents_map.values())
        if not mailbox:
            return None, context

        event = mailbox.popleft()

        logger.debug(f"[event][{self.name}] [{context.step}] {type(event).__name__}: {event}")

        # Handle events

        if isinstance(event, Stop):
            logger.debug(f"[{self.name}] Controller stopped: {event}")
            return event, context

        if isinstance(event, AgentRequested):
            if event.target not in self.agents_map:
                logger.error(f"[{self.name}] Agent '{event.target.__name__}' not found in {[agent.__name__ for agent in self.agents_map.keys()]}; stopping")
                mailbox.append(Stop())
                return event, context

            agent = self.agents_map[event.target]

            # ----------------------------------------------------------------------
            # Run agent, manage context, and store results
            # ----------------------------------------------------------------------


            # build a selected world view for the agent
            agent_context = self.build_agent_context(agent, context=context)
            agent_context.create_messages_trace()
            agent_context.check_agents_have_state(self.agents_map.values())
            result, agent_context = agent.run(
                artifact=event.artifact,
                context=agent_context,
                agents=list(self.agents_map.values()),
            )

            logger.debug(f"Messages: {agent_context.trace.messages}")

            context = self.merge_context(
                global_context=context,
                local_context=agent_context,
            )
            # ----------------------------------------------------------------------
            # Process result
            # ----------------------------------------------------------------------

            logger.debug(f"[{self.name}] [{context.step}] [result] {result}")

            if result.artifacts:
                artifact = result.artifacts[0] #TODO: Handle multiple artifacts
            else:
                logger.debug(f"Agent {agent.id} did not return any artifacts; keeping previous artifact")
                artifact = event.artifact

            # Store artifact in global context
            if isinstance(artifact, EndArtifact):
                pass
            context.artifacts[artifact.id] = artifact

            logger.debug(f"[{self.name}] [{context.step}] [recommendations] {result.recommendations}")

            recommendation = max(result.recommendations, key=lambda r: r.utility) if result.recommendations else None

            # ----------------------------------------------------------------------
            # Determine next step
            # ----------------------------------------------------------------------

            recommendation = (
                max(result.recommendations, key=lambda r: r.utility)
                if result.recommendations
                else None
            )

            # Workflow agent produced a new plan
            if recommendation is not None:

                if recommendation.target not in self.agents_map:
                    logger.error(
                        f"Recommended agent '{recommendation.target.__name__ if hasattr(recommendation.target, '__name__') else str(recommendation.target)}' "
                        f"not found; stopping"
                    )
                    mailbox.append(Stop())

                elif context.step >= max_step:
                    mailbox.append(Stop())

                else:
                    mailbox.append(
                        AgentRequested(
                            target=recommendation.target,
                            artifact=artifact,
                            continuations=list(recommendation.continuations),
                        )
                    )

            # Utility agent produced no recommendation:
            # continue the existing plan
            elif event.continuations:

                next_target = event.continuations.pop(0)

                mailbox.append(
                    AgentRequested(
                        target=next_target,
                        artifact=artifact,
                        continuations=event.continuations,
                    )
               )

            # No recommendation and no remaining continuation
            else:
                mailbox.append(Stop())

            # -------------------------------------------------------------------
            # Optional console output for debugging and visualization
            # -------------------------------------------------------------------§

            if console and "Input" not in artifact.producer:
                console.print(f"[bold blue]{artifact.producer}> [/bold blue]{artifact.content}")

            return event, context

        logger.debug(f"[{self.name}] Unknown event type: {type(event).__name__}")
        return event, context

    def run(self, start: type[Agent], artifact: Artifact, mailbox: deque, context: Context, max_step: int = 10, console: Console | None = None) -> Context:
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
        assert start in self.agents_map, f"Starting agent '{start.__name__}' not found in {[agent.__name__ for agent in self.agents_map.keys()]}"
        context.check_agents_have_state(self.agents_map.values())

        self.start(
            mailbox=mailbox,
            context=context,
            start=start,
            artifact=artifact,
            console=console,
        )

        while mailbox:
            _, context = self.step_once(
                mailbox=mailbox,
                context=context,
                max_step=max_step,
                console=console,
            )

        return context


# ------------------------------------------------------------------------------
# Smoke Test
# ------------------------------------------------------------------------------


def _smoke_test(console: Console):

    from aidu.ai.llm.agent import UserInput, EchoAgent, DebugAgent

    from aidu.ai.agents.math_tutor import MathTutor, MathUserInput
    from aidu.ai.agents.chem_tutor import ChemTutor, ChemUserInput
    from aidu.ai.agents.chat_bot import ChatBot
    from aidu.ai.agents.symbolic_solver import SymbolicSolver
    from aidu.ai.llm.clients.openai import OpenAIClient
    from aidu.ai.core.belief import StudentBelief

    context = Context()

    client = OpenAIClient(model="gpt-5-mini")
    # Initialize belief state
    belief = StudentBelief()
    belief.engagement = 0.8
    belief.confusion = 0.6
    context.state.data["StudentBelief"] = belief
    # ------------------------------------------------------------------------------
    # Define agents and controller
    # ------------------------------------------------------------------------------

    agents = [
        ChemTutor(client, prompt_args={
            "tutor_name": "Marie", 
            "focus_area": "What is an atom?",
            "level": "beginner",
            "history": " - Student has mentioned that atoms have protons, you asked for more details.",
            "student_progress": " - mentioned protons, not mentioned yet core, neutrons, electrons, or nucleus.",
            "student_belief": " - " +context.state.data["StudentBelief"].to_tutor_text(),
            **ChemTutor.default_state}
            ),
        DebugAgent(),
        ChemUserInput(),
    ]

    # Initialize state for each agent

    context.create_agent_states(agents)

    controller = Controller(
        name="main_controller",
        agents=agents,
        # show_trace=True,
    )

    starting_artifact = TextArtifact(
        producer="starter",
        step=0,
        content="An Atom has Neutrons.",
    )

    controller.run(
        mailbox=deque(),
        context=context,
        start=ChemTutor,
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
