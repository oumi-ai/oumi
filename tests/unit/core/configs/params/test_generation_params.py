from oumi.core.configs.params.generation_params import GenerationParams


def test_chat_template_kwargs_field_default():
    gen_params = GenerationParams()
    assert isinstance(gen_params.chat_template_kwargs, dict)
    assert gen_params.chat_template_kwargs == {}


def test_chat_template_kwargs_custom_assignment():
    gen_params = GenerationParams(chat_template_kwargs={"enable_thinking": False})
    assert gen_params.chat_template_kwargs["enable_thinking"] is False
