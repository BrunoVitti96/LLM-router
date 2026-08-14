"""Router input formatting shared by data preparation and online inference."""


def format_router_input(
    prompt: str,
    task: str,
    subject: str,
    num_choices: int,
    length_bin: str,
) -> str:
    return (
        f"[TASK={task}] [SUBJECT={subject}] [CHOICES={num_choices}] "
        f"[LENGTH_BIN={length_bin}] {prompt}"
    )
