# src/aidu/ai/core/processor.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from uuid import uuid4

from aidu.ai.core.processor_result import ProcessorResult
from aidu.ai.core.artifacts import Artifact, SymbolicArtifact, TextArtifact, create_artifact
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context
from aidu.ai.core.recommendation import Recommendation
from aidu.support.typing.analyse.nested import infer_schema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(ABC):
    """
    Base class for all processing units.

    A processor consumes an artifact and produces:

        - artifacts
        - recommendations

    The controller owns execution flow.
    Processors never invoke each other directly.
    """

    name: str = "processor"

    consumes: list[type[Artifact]] = []

    produces: list[type[Artifact]] = []

    @abstractmethod
    def run(self, step: int, artifact: Artifact, context: Context = None, console=None) -> ProcessorResult:
        """
        Process a single artifact.

        Parameters
        ----------
        artifact:
            Input artifact.

        Returns
        -------
        ProcessorResult
            Produced artifacts and recommendations.
        """
        raise NotImplementedError

    def _to_agent_result(self, response: dict, context: Context, producer: str, step: int) -> ProcessorResult:
        """
        Convert an agent response to a ProcessorResult, handling both direct content and route messages.
        """

        logger.debug(f"Processing response: {infer_schema(response)}")
        # we should see '_fc_message' when we see 'function_call'
        assert "function_call" not in response or "_fc_message" in response, "Response contains 'function_call' but no '_fc_message'"

        # check if response['content'] contains already the keys 'artifacts' and 'recommendations'
        if "artifacts" in response.get("content", {}):
            logger.debug("Response content contains 'artifacts'. Checking for 'recommendations'.")
            if "recommendations" in response.get("content", {}):
                logger.debug("Response content contains 'artifacts' and 'recommendations'. Using it directly.")
                return ProcessorResult(
                    artifacts=[
                        create_artifact(
                            a["type"],
                            id=a["id"],
                            producer=a["producer"],
                            step=a["step"],
                            content=a["content"],
                        )
                        for a in response["content"].get("artifacts", [])
                    ],
                    recommendations=[Recommendation.model_validate(r) for r in response["content"].get("recommendations", [])],
                )
            else:
                logger.debug("Response content contains 'artifacts' but no 'recommendations'. Using artifacts and empty recommendations.")
                return ProcessorResult(
                    artifacts=[
                        create_artifact(
                            a["type"],
                            id=a["id"],
                            producer=a["producer"],
                            step=a["step"],
                            content=a["content"],
                        )
                        for a in response["content"].get("artifacts", [])
                    ],
                    recommendations=[],
                )

        # check if response is a fc_message and contains a route message and process it if present
        fc_message = response.get("_fc_message")
        if fc_message and fc_message.get("type") == "route":
            logger.debug("Response contains a route message. Processing route content.")
            content = fc_message["content"]

            return ProcessorResult(
                artifacts=[
                    create_artifact(
                        a["type"],
                        id=a["id"],
                        producer=a["producer"],
                        step=a["step"],
                        content=a["content"],
                    )
                    for a in content.get("artifacts", [])
                ],
                recommendations=[Recommendation.model_validate(r) for r in content.get("recommendations", [])],
            )

        logger.debug(f"Response does not contain a route message.")
        return ProcessorResult(
            artifacts=[
                TextArtifact(
                    id="response",
                    producer=producer,
                    step=step,
                    content=response["content"],
                )
            ],
            recommendations=[Recommendation.model_validate(r) for r in content.get("recommendations", [])],
        )


class AgentProcessor(Processor):
    def __init__(self, agent):
        self.agent = agent

    def run(self, step: int, artifact: TextArtifact, context: Context, console=None) -> ProcessorResult:

        message = {"role": "user", "content": artifact.content}

        logger.debug(f"Agent prior context: {context}")

        response, context = self.agent.ask(message, context, ask_config=AskConfig(route_mode=True))
        logger.debug(f"Agent response: {response}")
        next_step = step + 1
        context.step = next_step
        logger.debug(f"Agent post context: {context}")

        if "function_call" in response:
            logger.debug(f"Agent function call: {response['function_call']['name']} with arguments {response['function_call']['arguments']}")

        result = self._to_agent_result(response, context, producer=self.agent.id, step=step)

        logger.debug(f"AgentProcessor result: {result}")

        return next_step, result


class DummyProcessor:
    id = "dummy"

    def run(self, step: int, artifact: SymbolicArtifact, context: Context, console=None) -> ProcessorResult:
        logger.debug(f"DummyProcessor received artifact: {artifact}")

        value = int(artifact.content) + 1

        next_step = step + 1

        context.step = next_step

        result = ProcessorResult(
            artifacts=[
                SymbolicArtifact(
                    id=f"{uuid4()}",
                    producer=self.id,
                    step=next_step,
                    content=value,
                )
            ],
            recommendations=[
                Recommendation(
                    target="input",
                    utility=1.0,
                    rationale="increment again",
                )
            ],
        )

        logger.debug(f"DummyProcessor result: {result}")
        

        return next_step, result


class EchoProcessor(Processor):
    id = "echo"

    def run(self, step: int, artifact: TextArtifact, context: Context, console=None) -> ProcessorResult:
        logger.debug(f"EchoProcessor received artifact: {artifact}")

        value = "you said, " + artifact.content

        next_step = step + 1        

        result = ProcessorResult(
            artifacts=[
                SymbolicArtifact(
                    id=f"{uuid4()}",
                    producer=self.id,
                    step=next_step,
                    content=value,
                )
            ],
            recommendations=[
                Recommendation(
                    target="input",
                    utility=1.0,
                    rationale="echo input",
                )
            ],
        )

        logger.debug(f"EchoProcessor result: {result}")
        

        return next_step, result


class UserInputProcessor(Processor):
    id = "input"

    def __init__(self, target: str = "chat_bot"):
        self.target = target

    def run(self, step: int, artifact: TextArtifact, context: Context, console=None) -> ProcessorResult:
        logger.debug(f"UserInputProcessor received artifact: {artifact}")

        # ----------------------------------------------------------
        # Get user input
        # ----------------------------------------------------------
        user_input = context.control.data.pop("user_input", None)
        if user_input is None:
            if console is None:
                raise ValueError("UserInputProcessor requires input in context.control.data['user_input'] when no console is provided")
            console.print(f"[bold green]user>[/bold green] ", end="")
            user_input = console.input()

        next_step = step + 1

        context.step = next_step

        result = ProcessorResult(
            artifacts=[
                SymbolicArtifact(
                    id=f"{uuid4()}",
                    producer=self.id,
                    step=next_step,
                    content=user_input,
                )
            ],
            recommendations=[
                Recommendation(
                    target=self.target,
                    utility=1.0,
                    rationale="getting user input",
                )
            ],
        )

        logger.debug(f"UserInputProcessor result: {result}")
        

        return next_step, result     