import pytest
from pytest_mock import MockerFixture

from vocode.streaming.agent.anthropic_agent import AnthropicAgent
from vocode.streaming.models.agent import AnthropicAgentConfig
from vocode.streaming.models.transcript import Transcript


@pytest.mark.asyncio
async def test_a_claude_agent_with_the_livekit_splitter_sends_whole_sentences(
    mocker: MockerFixture,
):
    mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"})
    agent = AnthropicAgent(
        AnthropicAgentConfig(prompt_preamble="", use_livekit_sentence_splitter=True)
    )
    agent.attach_transcript(Transcript())
    state_manager = mocker.MagicMock()
    state_manager.using_input_streaming_synthesizer.return_value = True
    agent.attach_conversation_state_manager(state_manager)
    mocker.patch.object(agent, "_get_anthropic_stream", mocker.AsyncMock())

    async def tokens(stream):
        for token in ["Great", ", thanks", " for letting", " me know.", " Is Tuesday", " good?"]:
            yield token

    mocker.patch.object(agent, "token_generator", tokens)

    replies = [r.message.text async for r in agent.generate_response("Hi", "conversation")]

    assert replies == ["Great, thanks for letting me know.", "Is Tuesday good?"]
