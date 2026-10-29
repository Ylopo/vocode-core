import pytest
from pytest_mock import MockerFixture

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.synthesizer import PlayHtSynthesizerConfig
from vocode.streaming.synthesizer.default_factory import DefaultSynthesizerFactory

DEFAULT_PARAMS = {
    "sampling_rate": 16000,
    "audio_encoding": AudioEncoding.LINEAR16,
    "voice_id": "s3://manifest.json",
}


def test_get_play_ht_synthesizer_v2_or_v1(mocker: MockerFixture):
    factory = DefaultSynthesizerFactory()

    v1_constructor = mocker.patch("vocode.streaming.synthesizer.default_factory.PlayHtSynthesizer")
    v2_constructor = mocker.patch(
        "vocode.streaming.synthesizer.default_factory.PlayHtSynthesizerV2"
    )

    factory.create_synthesizer(
        PlayHtSynthesizerConfig(version="2", **DEFAULT_PARAMS.copy()),
    ),
    v2_constructor.assert_called_once()
    v2_constructor.reset_mock()

    factory.create_synthesizer(
        PlayHtSynthesizerConfig(version="1", **DEFAULT_PARAMS.copy()),
    )
    v1_constructor.assert_called_once()
    v1_constructor.reset_mock()

    factory.create_synthesizer(
        PlayHtSynthesizerConfig(**DEFAULT_PARAMS.copy()),
    )
    v2_constructor.assert_called_once()


def test_play_ht_synthesizer_v2_fails_clearly_without_pyht():
    factory = DefaultSynthesizerFactory()

    with pytest.raises(ImportError, match="pyht"):
        factory.create_synthesizer(
            PlayHtSynthesizerConfig(version="2", **DEFAULT_PARAMS.copy()),
        )
