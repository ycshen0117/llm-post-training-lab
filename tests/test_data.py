from llm_post_training_lab.data import hello


def test_hello() -> None:
    assert hello() == "hello"
