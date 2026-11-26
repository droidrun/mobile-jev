"""Prepare the image and retain the unpadded coverage for Jev's boxes."""

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from mobilerun_agents.config.user_config import DecisionConfig


def prepare_image(screenshot: bytes, config: DecisionConfig) -> tuple[bytes, tuple[float, float]]:
    try:
        with Image.open(BytesIO(screenshot)) as original:
            try:
                original.seek(1)
            except EOFError:
                original.seek(0)
            else:
                raise ValueError("TypeSafe screenshots must be still images")
            image = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("TypeSafe requires a valid screenshot image") from exc
    width, height = image.size
    scale = min(
        config.image_max_side / max(width, height),
        max(1.0, config.image_min_side / min(width, height)),
    )
    size = max(1, round(width * scale)), max(1, round(height * scale))
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    padded = max(config.image_min_side, size[0]), max(config.image_min_side, size[1])
    image = ImageOps.pad(image, padded, color="white", centering=(0, 0))
    output = BytesIO()
    image.save(output, format="JPEG", quality=config.jpeg_quality, optimize=True, progressive=False)
    return output.getvalue(), (size[0] / padded[0], size[1] / padded[1])
