import asyncio
import dataclasses

import pytest
from pytest_mock import MockerFixture

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BotBackchannel
from vocode.streaming.models.synthesizer import LiveKitSynthesizerConfig
from vocode.streaming.synthesizer.livekit_synthesizer import (
    LiveKitSynthesizer,
    _Stream,
    _WholeSentenceTokenizer,
    build_plugin_tts,
)
from vocode.streaming.utils.async_requester import AsyncRequestor


@pytest.fixture(autouse=True)
def provider_api_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("RIME_API_KEY", "test")
    monkeypatch.setenv("ELEVEN_API_KEY", "test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,options",
    [
        ("cartesia", {"base_url": "https://attacker.example"}),
        ("cartesia", {"base_url": "https://api.cartesia.ai.attacker.example"}),
        ("cartesia", {"base_url": "https://evilcartesia.ai"}),
        ("cartesia", {"base_url": "http://api.cartesia.ai"}),
        ("rime", {"base_url": "wss://attacker.example", "allow_custom_endpoint": True}),
    ],
)
async def test_options_cannot_send_the_api_key_outside_the_provider(provider, options):
    with pytest.raises(ValueError):
        build_plugin_tts(provider, options, True, 8000)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["rime.tts", "elevenlabs.tts", "cartesia.tts"])
async def test_a_plugin_can_only_be_chosen_by_its_plain_name(provider):
    with pytest.raises(ValueError, match="No LiveKit"):
        build_plugin_tts(provider, {}, True, 8000)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,options",
    [
        ("cartesia", {"base_url": "https://api.cartesia.ai"}),
        ("rime", {"base_url": "wss://users-ws.rime.ai"}),
    ],
)
async def test_options_can_choose_another_endpoint_of_the_provider(provider, options):
    plugin_tts = build_plugin_tts(provider, options, True, 8000)
    await plugin_tts.aclose()


@pytest.mark.asyncio
async def test_closing_the_shared_http_session_does_not_close_the_plugins_session():
    synthesizer = LiveKitSynthesizer(
        LiveKitSynthesizerConfig(
            provider="cartesia", sampling_rate=8000, audio_encoding=AudioEncoding.MULAW
        )
    )

    # what the transfer action does with the shared session once it has sent its request
    async with AsyncRequestor().get_session():
        pass

    assert synthesizer.http_session is not None
    assert not synthesizer.http_session.closed
    await synthesizer.tear_down()
    assert synthesizer.http_session.closed


# Rime's v1 WebSocket gives no word timings through the plugin, while its legacy /ws3 does
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,uses_websocket_v1",
    [
        ({"model": "coda", "speaker": "lyra"}, True),
        ({"model": "mistv3", "speaker": "cove"}, True),
        ({"speaker": "lyra"}, True),
        ({"model": "mistv2", "speaker": "cove"}, False),
        ({"model": "mist", "speaker": "cove"}, False),
        ({"model": "coda", "speaker": "lyra", "base_url": "wss://users-ws.rime.ai"}, False),
    ],
)
async def test_rime_streams_current_models_over_websocket_v1_and_legacy_models_over_ws3(
    options, uses_websocket_v1
):
    plugin_tts = build_plugin_tts("rime", options, True, 8000)

    assert plugin_tts.capabilities.streaming
    assert plugin_tts.capabilities.aligned_transcript is not uses_websocket_v1
    await plugin_tts.aclose()


@pytest.mark.asyncio
async def test_rime_without_streaming_uses_http_for_current_models():
    plugin_tts = build_plugin_tts("rime", {"model": "coda", "speaker": "lyra"}, False, 8000)

    assert not plugin_tts.capabilities.streaming
    await plugin_tts.aclose()


def test_cut_off_without_word_timings_follows_the_turns_real_audio_length():
    synthesizer = LiveKitSynthesizer(
        LiveKitSynthesizerConfig(
            provider="cartesia", sampling_rate=8000, audio_encoding=AudioEncoding.MULAW
        )
    )
    stream = _Stream("turn", 160)
    stream.add_text("one two three four five six seven eight")
    # a slow voice: eight words over 8 seconds, half the default 150 words per minute
    stream.audio_bytes = 8 * synthesizer.bytes_per_second
    stream.audio_complete = True

    assert synthesizer.get_message_up_to(stream, 3.5) == "one two three four"


@pytest.mark.asyncio
async def test_cut_off_inside_a_cached_opening_keeps_the_words_heard_before_the_live_ones():
    synthesizer = LiveKitSynthesizer(
        LiveKitSynthesizerConfig(
            provider="cartesia", sampling_rate=8000, audio_encoding=AudioEncoding.MULAW
        )
    )
    assert synthesizer.word_timestamps
    stream = _Stream("turn", 160)
    synthesizer.push_cached(
        stream, "Hi, this is Alex.", b"\xff" * (2 * synthesizer.bytes_per_second)
    )
    # Cartesia's word timings for the live sentence that follows the 2 seconds of cached audio
    stream.word_start_times += [("Are", 2.1), ("you", 2.4), ("there?", 2.7)]

    assert synthesizer.get_message_up_to(stream, 1.2) == "Hi, this is"
    assert synthesizer.get_message_up_to(stream, 2.5) == "Hi, this is Alex. Are you"
    await synthesizer.tear_down()


@pytest.mark.asyncio
async def test_elevenlabs_sends_each_sentence_as_soon_as_it_is_pushed():
    plugin_tts = build_plugin_tts("elevenlabs", {"voice_id": "abc"}, True, 8000)

    # its default tokenizer holds a sentence back until the next one starts
    assert isinstance(plugin_tts._opts.word_tokenizer, _WholeSentenceTokenizer)
    with pytest.raises(ValueError, match="word_tokenizer"):
        build_plugin_tts("elevenlabs", {"word_tokenizer": None}, True, 8000)
    await plugin_tts.aclose()


@pytest.mark.asyncio
async def test_elevenlabs_voice_settings_and_dictionaries_saved_as_json_work_on_a_call():
    plugin_tts = build_plugin_tts(
        "elevenlabs",
        {
            "voice_id": "abc",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            "pronunciation_dictionary_locators": [
                {"pronunciation_dictionary_id": "d", "version_id": "v"}
            ],
        },
        True,
        8000,
    )

    # what the plugin does with them on every request
    assert dataclasses.asdict(plugin_tts._opts.voice_settings)["stability"] == 0.5
    assert plugin_tts._opts.pronunciation_dictionary_locators[0].pronunciation_dictionary_id == "d"
    with pytest.raises(ValueError, match="Invalid elevenlabs options"):
        build_plugin_tts("elevenlabs", {"voice_settings": {"loudness": 1}}, True, 8000)
    await plugin_tts.aclose()


@pytest.mark.asyncio
async def test_cartesia_only_accepts_the_encoding_its_audio_is_decoded_as():
    with pytest.raises(ValueError, match="pcm_s16le"):
        build_plugin_tts("cartesia", {"encoding": "pcm_mulaw"}, True, 8000)
    plugin_tts = build_plugin_tts("cartesia", {"encoding": "pcm_s16le"}, True, 8000)
    await plugin_tts.aclose()


def test_audio_cached_at_one_sample_rate_is_not_reused_at_another():
    def identifier(sampling_rate: int) -> str:
        return LiveKitSynthesizer.get_voice_identifier(
            LiveKitSynthesizerConfig(
                provider="cartesia",
                sampling_rate=sampling_rate,
                audio_encoding=AudioEncoding.LINEAR16,
            )
        )

    assert identifier(16000) != identifier(24000)


def _http_synthesizer() -> LiveKitSynthesizer:
    # without streaming there is no connection to prewarm when the caller finishes speaking
    return LiveKitSynthesizer(
        LiveKitSynthesizerConfig(
            provider="cartesia",
            streaming=False,
            sampling_rate=8000,
            audio_encoding=AudioEncoding.MULAW,
        )
    )


@pytest.mark.asyncio
async def test_an_interrupted_turn_stops_generating_and_the_next_reply_opens_a_new_one():
    synthesizer = _http_synthesizer()
    turn = synthesizer.current_turn(160)
    turn.reader = asyncio.ensure_future(asyncio.sleep(10))

    synthesizer.cancel_turn()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(turn.reader, timeout=1)
    assert synthesizer.current_turn(160) is not turn
    await synthesizer.tear_down()


@pytest.mark.asyncio
async def test_a_stream_that_stops_producing_audio_is_cancelled_and_recorded_as_unheard(
    mocker: MockerFixture,
):
    mocker.patch("vocode.streaming.synthesizer.livekit_synthesizer.AUDIO_TIMEOUT_SECONDS", 0.05)
    synthesizer = _http_synthesizer()
    turn = synthesizer.current_turn(160)
    turn.add_text("Hello there.")
    turn.plugin_stream = mocker.AsyncMock()
    turn.reader = asyncio.ensure_future(asyncio.sleep(10))
    result = synthesizer.synthesis_result(turn)

    chunks = [chunk async for chunk in result.chunk_generator]

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(turn.reader, timeout=1)
    turn.plugin_stream.aclose.assert_awaited()
    assert chunks == []
    assert result.get_message_up_to(None) == ""
    await synthesizer.tear_down()


@pytest.mark.asyncio
async def test_an_interrupted_cached_opening_is_not_reused_by_the_next_reply():
    synthesizer = _http_synthesizer()
    turn = synthesizer.current_turn(160)
    synthesizer.push_cached(turn, "Hey, is this Faisal?", b"\xff" * 1600)

    synthesizer.cancel_turn()
    next_turn = synthesizer.current_turn(160)

    assert next_turn is not turn
    assert next_turn.text == ""
    await synthesizer.tear_down()


@pytest.mark.asyncio
async def test_a_filler_is_synthesized_on_its_own_instead_of_opening_the_reply_turn(
    mocker: MockerFixture,
):
    synthesizer = _http_synthesizer()
    filler_audio = mocker.MagicMock()
    mocker.patch.object(synthesizer, "get_cached_audio", mocker.AsyncMock(return_value=None))
    mocker.patch.object(
        synthesizer, "create_speech_uncached", mocker.AsyncMock(return_value=filler_audio)
    )

    result = await synthesizer.create_speech(
        BotBackchannel(text="Got it, one moment."), 160, is_first_text_chunk=True
    )

    assert result is filler_audio
    assert synthesizer.turn is None
    await synthesizer.tear_down()
