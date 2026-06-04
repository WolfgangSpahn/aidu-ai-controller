"""
Math tutor agent
"""

import logging
import textwrap
from uuid import uuid4


from aidu.ai.llm.agent import LLMAgent
from aidu.ai.core.context import Context, Message
from aidu.ai.core.recommendation import Recommendation
from aidu.ai.core.artifacts import Artifact, SymbolicArtifact
from aidu.support.regex.validate import assert_valid_sympy_problem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    id: str = "math_tutor"

    def fc_route_symbolic_solver(self, context: Context, problem: str) -> tuple[Message, Context]:
        """
        Use this function when symbolic mathematics is required. Use SymPy syntax for the problem statement,
        like "diff(4*x**3, x)" for differentiation, "solve(x**2 - 4, x)" for solving equations,
        or "integrate(sin(x), x)" for integration.

        Examples:
        - derivatives
        - integrals
        - equation solving
        - simplification
        - symbolic manipulation

        Args:
            problem (str): Mathematical problem expressed in SymPy syntax, e.g. "diff(4*x**3, x)".


        Alerts:
        - **Ensure** the problem parameter is in valid SymPy syntax to avoid parsing errors.

        """

        producer = f"{self.id}:fc_route_symbolic_solver"

        try:
            # ----------------------------------------------------------
            # Process LLM function call with validation and error handling
            # ----------------------------------------------------------

            if problem is None:
                raise ValueError("Missing required parameter: problem")

            if not isinstance(problem, str):
                problem = str(problem)
            assert_valid_sympy_problem(problem)

            default_target = "symbolic_solver"

            # ----------------------------------------------------------
            # Build Artifact and Recommendation for routing to the symbolic solver
            # ----------------------------------------------------------

            artifact     = SymbolicArtifact(producer=producer, step=context.step, content=problem)
            recommendation = Recommendation(target=default_target, utility=1.0, rationale="symbolic computation requested")

            # ----------------------------------------------------------
            # Return route message
            # ----------------------------------------------------------

            message = {
                "role": self.role,
                "type": "route",
                "content": {"artifacts": [artifact.model_dump()], "recommendations": [recommendation.model_dump()]},
            }
            logger.debug(f"fc_route_symbolic_solver produced message: {message}")

            return message, context

        except Exception as e:
            # ----------------------------------------------------------
            # Handle errors gracefully and route to an error target
            # ----------------------------------------------------------

            logger.exception("fc_route_symbolic_solver failed")
            error_target = "math_tutor"

            artifact = SymbolicArtifact(producer=producer, step=context.step, content=str(e))
            recommendation = Recommendation(target=error_target, utility=1.0, rationale="error in processing symbolic problem")

            message = {
                "role": self.role,
                "type": "route",
                "content": {"artifacts": [artifact.model_dump()], "recommendations": [recommendation.model_dump()]},
            }
            logger.debug(f"fc_route_symbolic_solver produced error message: {message}")
            return message, context
