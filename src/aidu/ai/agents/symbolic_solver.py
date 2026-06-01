import logging

from json import dumps
from uuid import uuid4

from aidu.ai.symbolic.engine import Engine
from aidu.ai.symbolic.engines.SymbolicSolver import solve_math_problem_with_sympy

logger = logging.getLogger(__name__)


class SymbolicSolver(Engine):
    process = staticmethod(solve_math_problem_with_sympy)

    def ask(
        self,
        message,
        context,
        config=None,
    ):
        logger.debug(f"SymbolicSolver received message: {dumps(message, indent=2)}")
        result = self.process(message["content"])
        logger.debug(f"SymbolicSolver result: {result.keys()}")
        logger.debug(f"SymbolicSolver result: {dumps(result, indent=2)}")
        # ----------------------------------------------------------
        # Harmonized route message
        # ----------------------------------------------------------

        message = {
            "role": "assistant",
            "type": "route",
            "content": {
                "artifacts": [
                    {
                        "id": str(uuid4()),
                        "type": "symbolic",
                        "content": result["result"],
                    }
                ],
                "recommendations": [
                    {
                        "target": "math_tutor",
                        "utility": 1.0,
                        "rationale": "comment on the solution requested",
                    }
                ],
            },
        }

        return message, context
