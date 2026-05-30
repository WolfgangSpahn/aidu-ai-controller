# src/aidu/ai/core/processor.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import Artifact, TextArtifact
from aidu.ai.core.context import Context
from aidu.ai.core.recommendation import Recommendation
from aidu.support.typing.analyse.nested import infer_schema

logger = logging.getLogger(__name__)


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
    def run(self, artifact: Artifact, context: Context = None, console=None) -> AgentResult:
        """
        Process a single artifact.

        Parameters
        ----------
        artifact:
            Input artifact.

        Returns
        -------
        AgentResult
            Produced artifacts and recommendations.
        """
        raise NotImplementedError

    def _to_agent_result(self, response: dict, context: Context) -> AgentResult:
        
        logger.debug(f"Processing response: {infer_schema(response)}")

        fc_message = response.get("_fc_message")

        # check if response['content'] contains already the keys 'artifacts' and 'recommendations'
        if "artifacts" in response.get("content", {}) and "recommendations" in response.get("content", {}):
            logger.debug("Response content contains 'artifacts' and 'recommendations'. Using it directly.")
            return AgentResult(
                artifacts=[Artifact.model_validate(a) for a in response["content"].get("artifacts", [])],
                recommendations=[Recommendation.model_validate(r) for r in response["content"].get("recommendations", [])],
            )

        if fc_message and fc_message.get("type") == "route":
            content = fc_message["content"]

            return AgentResult(
                artifacts=[Artifact.model_validate(a) for a in content.get("artifacts", [])],
                recommendations=[Recommendation.model_validate(r) for r in content.get("recommendations", [])],
            )
        logger.warning(f"Response does not contain a route message. Returning empty AgentResult using {response}")
        return AgentResult(artifacts=[TextArtifact(id="response", content=response["content"])])


class AgentProcessor(Processor):
    def __init__(self, agent):
        self.agent = agent

    def run(self, artifact: TextArtifact, context: Context, console=None) -> AgentResult:

        message = {"role": "user", "content": artifact.content}

        response, context = self.agent.ask(message, context)

        logger.debug(f"Agent response: {response}")

        if "function_call" in response:
            logger.debug(f"Agent function call: {response['function_call']['name']} with arguments {response['function_call']['arguments']}")

        if console:
            context.pretty(console)

        return self._to_agent_result(response, context)
