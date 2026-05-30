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


class MathTutor(LLMAgent):
    """A math tutor agent with function calls for solving problems and tracking student progress."""

    # System prompt with flexible placeholders that can be filled via prompt_args
    # Unfilled placeholders will remain as {placeholder} for later customization
    #
    # Usage examples:
    #   # Use with defaults (unfilled placeholders remain as {placeholder})
    #   tutor = MathTutor(client)
    #
    #   # Customize specific fields
    #   tutor = MathTutor(client, prompt_args={"student_name": " for Alice", "level": " in algebra"})
    #
    #   # Override at prompt building time
    #   messages = tutor.build_system_prompt(prompt_params={"focus_areas": " - focus on calculus"})

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient math tutor {tutor_name} for the area {focus_areas}.
                                      
        Your goal is to help students at {level}to understand mathematical concepts and solve problems step by step.
                                      
        Here the summary of the task so far: {dialogue_history} and the students progress: {student_progress}. Here our current assessment of the student's beliefs: {student_beliefs}.

        When responding:
        - never output more than 3 sentences at a time                 
        - Use clear, educational language appropriate for the student's level
        - Format your responses using markdown with:
            - **Bold** for important mathematical terms
            - Headers for major sections or steps
            - Lists for step-by-step solutions
            - LaTeX expressions for equations (wrapped in $ or $$ delimiters)
        - Explain the reasoning behind each step, not just the answer
        - When exact symbolic computation is needed, such as derivatives, integrals, solving equations, or simplification, use fc_route_symbolic_solver.
        - Encourage students to think critically and ask questions
                  
        """).strip()

    def fc_route_symbolic_solver(self, context: Context, problem: str) -> tuple[Message, Context]:
        """
        Use this function when symbolic mathematics is required.

        Examples:
        - derivatives
        - integrals
        - equation solving
        - simplification
        - symbolic manipulation

        Parameters:
            problem:
                Mathematical problem expressed in SymPy syntax. like "diff(4*x**3, x)" for differentiation, "solve(x**2 - 4, x)" for solving equations, or "integrate(sin(x), x)" for integration.

        Alerts:
        - **Ensure** the problem parameter is in valid SymPy syntax to avoid parsing errors.

        """

        try:
            # ----------------------------------------------------------
            # Repair common LLM mistakes
            # ----------------§------------------------------------------

            if problem is None:
                raise ValueError("Missing required parameter: problem")

            if not isinstance(problem, str):
                problem = str(problem)
            assert_valid_sympy_problem(problem)

            artifact_type = "symbolic"

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
                            "type": artifact_type,
                            "content": problem,
                        }
                    ],
                    "recommendations": [
                        {
                            "target": "symbolic_solver",
                            "utility": 1.0,
                            "rationale": "symbolic computation requested",
                        }
                    ],
                },
            }

            return message, context

        except Exception as e:
            logger.exception("fc_route_symbolic_solver failed")

            message = {
                "role": "assistant",
                "type": "route",
                "content": {
                    "artifacts": [
                        {
                            "id": str(uuid4()),
                            "type": "error",
                            "content": str(e),
                        }
                    ],
                    "recommendations": [
                        {
                            "target": "math_tutor",
                            "utility": 1.0,
                            "rationale": "received invalid problem statement, need to handle the error",
                        }
                    ],
                },
            }

            return message, context
