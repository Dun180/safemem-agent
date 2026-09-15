from types import SimpleNamespace

import main


def test_run_cli_processes_message_and_exits(monkeypatch):
    user_inputs = iter(
        [
            "What is 6 * 7?",
            "exit",
        ]
    )

    outputs = []
    agent_inputs = []

    def fake_input(prompt):
        outputs.append(prompt)
        return next(user_inputs)

    def fake_output(message=""):
        outputs.append(str(message))

    def fake_run_agent(user_input):
        agent_inputs.append(user_input)

        return SimpleNamespace(
            answer="42",
            tool_events=[],
        )

    monkeypatch.setattr(
        main,
        "run_agent",
        fake_run_agent,
    )

    main.run_cli(
        input_fn=fake_input,
        output_fn=fake_output,
    )

    assert agent_inputs == [
        "What is 6 * 7?"
    ]

    assert any(
        "SafeMem Agent" in output
        for output in outputs
    )

    assert any(
        "Agent: 42" in output
        for output in outputs
    )