from collections import deque

from aidu.ai.controller.controller import Controller
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.context import Context, Trace
from aidu.ai.llm.agent import WorkflowAgent


class TraceInspectingAgent(WorkflowAgent):
    def run(self, artifact, context=None, agents=None):
        return self.result(artifacts=[artifact], recommendations=[]), context


def test_controller_preserves_dialog_only_trace_for_agent_context():
    agent = TraceInspectingAgent()
    controller = Controller("test-controller", agents=[agent])
    context = Context(
        trace=Trace(
            messages=[
                {
                    "role": "assistant",
                    "content": "Welcome Anonymous to our Chemistry Periodic Table session.",
                },
                {
                    "role": "user",
                    "content": "Applet event: applet-periodic-table with elementName=Hydrogen",
                },
            ]
        )
    )
    context.create_agent_states([agent])

    agent_context = controller.build_agent_context(agent, context)

    assert agent_context.trace.messages == context.trace.messages
    assert agent_context.trace.messages[0]["role"] == "assistant"


def test_controller_step_does_not_rebuild_trace_around_system_message():
    agent = TraceInspectingAgent()
    controller = Controller("test-controller", agents=[agent])
    context = Context(
        trace=Trace(
            messages=[
                {
                    "role": "assistant",
                    "content": "Welcome Anonymous to our Chemistry Periodic Table session.",
                },
                {
                    "role": "user",
                    "content": "Applet event: applet-periodic-table with elementName=Hydrogen",
                },
            ]
        )
    )
    context.create_agent_states([agent])
    mailbox = deque()
    artifact = TextArtifact(producer="user", step=0, content="current turn")
    mailbox, context = controller.start(
        start=TraceInspectingAgent,
        mailbox=mailbox,
        context=context,
        artifact=artifact,
    )

    _, context = controller.step_once(mailbox, context)

    assert context.trace.messages == [
        {
            "role": "assistant",
            "content": "Welcome Anonymous to our Chemistry Periodic Table session.",
        },
        {
            "role": "user",
            "content": "Applet event: applet-periodic-table with elementName=Hydrogen",
        },
    ]
