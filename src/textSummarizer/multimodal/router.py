from textSummarizer.models import ModelFactory
from textSummarizer.multimodal.audio import AudioSummarizer
from textSummarizer.multimodal.base import InputType, MultimodalInput
from textSummarizer.multimodal.image import ImageSummarizer
from textSummarizer.multimodal.video import VideoSummarizer


class MultimodalRouter:
    """Dispatch multimodal inputs to the appropriate summarizer."""

    def __init__(
        self,
        text_model: str = "extractive",
        image_caption_model: str | None = None,
        whisper_model: str | None = None,
    ):
        self.text_model = text_model
        self._image = ImageSummarizer(text_model=text_model)
        self._audio = AudioSummarizer(text_model=text_model)
        self._video = VideoSummarizer(text_model=text_model)
        if image_caption_model:
            self._image.caption_model = image_caption_model
            self._video._image.caption_model = image_caption_model
        if whisper_model:
            self._audio.whisper_model = whisper_model
            self._video._audio.whisper_model = whisper_model

    def summarize(
        self,
        input: MultimodalInput,
        max_length: int = 128,
        strategy: str = "stuff",
    ) -> dict:
        if input.input_type == InputType.TEXT:
            text = input.resolve_text()
            if not text:
                raise ValueError("Text input requires text, path, or base64_data")
            summarizer = ModelFactory.create(self.text_model)
            summary = summarizer.summarize(text, max_length=max_length, strategy=strategy)
            return {
                "input_type": input.input_type.value,
                "summary": summary,
                "model": self.text_model,
                "strategy": strategy,
            }

        if input.input_type == InputType.IMAGE:
            data = input.resolve_bytes()
            result = self._image.summarize(
                path=input.path,
                data=data,
                max_length=max_length,
                strategy=strategy,
            )
            return {
                "input_type": input.input_type.value,
                "caption": result["caption"],
                "summary": result["summary"],
                "model": result["model"],
                "strategy": strategy,
            }

        if input.input_type == InputType.AUDIO:
            data = input.resolve_bytes()
            result = self._audio.summarize(
                path=input.path,
                data=data,
                max_length=max_length,
                strategy=strategy,
            )
            return {
                "input_type": input.input_type.value,
                "transcript": result["transcript"],
                "summary": result["summary"],
                "model": result["model"],
                "strategy": strategy,
            }

        if input.input_type == InputType.VIDEO:
            data = input.resolve_bytes()
            result = self._video.summarize(
                path=input.path,
                data=data,
                max_length=max_length,
                strategy=strategy,
            )
            return {
                "input_type": input.input_type.value,
                "document": result["document"],
                "transcript": result["transcript"],
                "visual_captions": result["visual_captions"],
                "summary": result["summary"],
                "model": result["model"],
                "strategy": strategy,
                "chapters": result.get("chapters"),
            }

        raise ValueError(f"Unsupported input type: {input.input_type}")
