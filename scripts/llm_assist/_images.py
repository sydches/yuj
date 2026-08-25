"""Bounded local-image input and session-owned attachment evidence."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


ATTACHMENT_SCHEMA = "yuj.assistant-attachments"
ATTACHMENT_SCHEMA_VERSION = 1
MAX_IMAGE_COUNT = 20
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8000
MAX_DISPLAY_NAME_CHARS = 96
_MAX_MANIFEST_BYTES = 128 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_MEDIA_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
_SHA256_HEX = frozenset("0123456789abcdef")


class ImageInputError(ValueError):
    """One image input or its saved evidence is unsafe or invalid."""


@dataclass(frozen=True)
class PendingImage:
    display_name: str
    media_type: str
    data: bytes
    size_bytes: int
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class ImageEvidence:
    segment_number: int
    image_number: int
    display_name: str
    media_type: str
    size_bytes: int
    sha256: str
    width: int
    height: int
    relative_path: str
    user_text_sha256: str
    user_text_utf8_bytes: int
    user_text_chars: int


@dataclass(frozen=True)
class SessionImage(ImageEvidence):
    data: bytes


def read_image_inputs(paths: Sequence[Path]) -> tuple[PendingImage, ...]:
    """Read explicit image paths once, with count and byte bounds."""
    selected = tuple(Path(path) for path in paths)
    if len(selected) > MAX_IMAGE_COUNT:
        raise ImageInputError(
            f"image input accepts at most {MAX_IMAGE_COUNT} images"
        )

    images: list[PendingImage] = []
    aggregate = 0
    for image_number, path in enumerate(selected, start=1):
        data = _read_explicit_path(path, MAX_IMAGE_BYTES)
        aggregate += len(data)
        if aggregate > MAX_IMAGE_TOTAL_BYTES:
            raise ImageInputError(
                "image input exceeds the aggregate limit of "
                f"{MAX_IMAGE_TOTAL_BYTES} bytes"
            )
        media_type, width, height = _detect_image(data)
        images.append(PendingImage(
            display_name=_bounded_display_name(path, image_number),
            media_type=media_type,
            data=data,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            width=width,
            height=height,
        ))
    return tuple(images)


def save_image_segment(
    artifact_dir: Path,
    *,
    segment_number: int,
    prompt_text: str,
    images: Sequence[PendingImage],
) -> tuple[SessionImage, ...]:
    """Save one image-bearing user segment without retaining source paths."""
    if (
        isinstance(segment_number, bool)
        or not isinstance(segment_number, int)
        or segment_number < 1
    ):
        raise ImageInputError("image segment number must be a positive integer")
    selected = tuple(images)
    if not selected:
        return ()
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ImageInputError("image input requires non-empty text")

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(artifact_dir)
    existing = next(
        (
            segment
            for segment in manifest["segments"]
            if segment["segment_number"] == segment_number
        ),
        None,
    )
    prompt_meta = _prompt_metadata(prompt_text)
    proposed = _segment_record(segment_number, prompt_meta, selected)
    if existing is not None:
        if existing != proposed:
            raise ImageInputError(
                f"image segment {segment_number} already has different evidence"
            )
        return tuple(
            image
            for image in load_session_images(artifact_dir)
            if image.segment_number == segment_number
        )

    existing_images = sum(
        len(segment["images"]) for segment in manifest["segments"]
    )
    existing_bytes = sum(
        image["size_bytes"]
        for segment in manifest["segments"]
        for image in segment["images"]
    )
    if existing_images + len(selected) > MAX_IMAGE_COUNT:
        raise ImageInputError(
            "saved session image input exceeds the lifetime count limit of "
            f"{MAX_IMAGE_COUNT}"
        )
    selected_bytes = sum(image.size_bytes for image in selected)
    if existing_bytes + selected_bytes > MAX_IMAGE_TOTAL_BYTES:
        raise ImageInputError(
            "saved session image input exceeds the lifetime aggregate limit of "
            f"{MAX_IMAGE_TOTAL_BYTES} bytes"
        )

    attachments_dir = artifact_dir / "attachments"
    try:
        attachments_dir.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _require_directory(attachments_dir, "attachment directory")
    segment_dir = attachments_dir / f"segment-{segment_number:04d}"
    try:
        segment_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise ImageInputError(
            f"incomplete saved image evidence exists for segment {segment_number}"
        ) from exc

    for image_number, image in enumerate(selected, start=1):
        extension = _MEDIA_EXTENSIONS[image.media_type]
        path = segment_dir / f"image-{image_number:04d}{extension}"
        _write_new_private_file(path, image.data)

    manifest["segments"].append(proposed)
    manifest["segments"].sort(key=lambda item: item["segment_number"])
    _write_manifest(artifact_dir, manifest)
    return tuple(
        image
        for image in load_session_images(artifact_dir)
        if image.segment_number == segment_number
    )


def image_evidence(artifact_dir: Path) -> tuple[ImageEvidence, ...]:
    """Return bounded manifest metadata without reading stored image bytes."""
    manifest = _load_manifest(Path(artifact_dir))
    evidence: list[ImageEvidence] = []
    for segment in manifest["segments"]:
        prompt = segment["user_text"]
        for image in segment["images"]:
            evidence.append(ImageEvidence(
                segment_number=segment["segment_number"],
                image_number=image["image_number"],
                display_name=image["display_name"],
                media_type=image["media_type"],
                size_bytes=image["size_bytes"],
                sha256=image["sha256"],
                width=image["width"],
                height=image["height"],
                relative_path=image["relative_path"],
                user_text_sha256=prompt["sha256"],
                user_text_utf8_bytes=prompt["utf8_bytes"],
                user_text_chars=prompt["chars"],
            ))
    return tuple(evidence)


def load_session_images(artifact_dir: Path) -> tuple[SessionImage, ...]:
    """Load only manifest-named session files and verify every saved byte."""
    artifact_dir = Path(artifact_dir)
    loaded: list[SessionImage] = []
    for evidence in image_evidence(artifact_dir):
        parts = PurePosixPath(evidence.relative_path).parts
        attachments_dir = artifact_dir / parts[0]
        segment_dir = attachments_dir / parts[1]
        _require_directory(
            attachments_dir,
            "saved image evidence attachment directory",
        )
        _require_directory(
            segment_dir,
            "saved image evidence segment directory",
        )
        path = segment_dir / parts[2]
        try:
            data = _read_regular_file(
                path,
                MAX_IMAGE_BYTES,
                missing_message="saved image evidence is missing or unreadable",
                non_regular_message="saved image evidence is not a regular file",
                symlink_message="saved image evidence cannot be a symbolic link",
                too_large_message="saved image evidence exceeds the per-file limit",
            )
            media_type, width, height = _detect_image(data)
        except ImageInputError as exc:
            if "saved image evidence" in str(exc):
                raise
            raise ImageInputError(f"saved image evidence is invalid: {exc}") from exc
        if (
            len(data) != evidence.size_bytes
            or hashlib.sha256(data).hexdigest() != evidence.sha256
            or media_type != evidence.media_type
            or width != evidence.width
            or height != evidence.height
        ):
            raise ImageInputError(
                "saved image evidence does not match attachments.json"
            )
        loaded.append(SessionImage(**evidence.__dict__, data=data))
    return tuple(loaded)


def _read_explicit_path(path: Path, limit: int) -> bytes:
    return _read_regular_file(
        path,
        limit,
        missing_message=f"image input is not readable: {path}",
        non_regular_message=f"image input is not a regular file: {path}",
        symlink_message=f"image input cannot be a symbolic link: {path}",
        too_large_message=(
            f"image input exceeds the per-file limit of {limit} bytes: {path}"
        ),
    )


def _read_regular_file(
    path: Path,
    limit: int,
    *,
    missing_message: str,
    non_regular_message: str,
    symlink_message: str,
    too_large_message: str,
) -> bytes:
    try:
        initial = os.lstat(path)
    except OSError as exc:
        raise ImageInputError(missing_message) from exc
    if stat.S_ISLNK(initial.st_mode):
        raise ImageInputError(symlink_message)
    if not stat.S_ISREG(initial.st_mode):
        raise ImageInputError(non_regular_message)
    if initial.st_size > limit:
        raise ImageInputError(too_large_message)

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ImageInputError(missing_message) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ImageInputError(non_regular_message)
        chunks: list[bytes] = []
        total = 0
        while total <= limit:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, limit + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > limit:
            raise ImageInputError(too_large_message)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_directory(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ImageInputError(f"{label} is missing or unreadable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ImageInputError(f"{label} cannot be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ImageInputError(f"{label} is not a directory")


def _detect_image(data: bytes) -> tuple[str, int, int]:
    detected: tuple[str, int, int] | None = None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = _png_dimensions(data)
    elif data.startswith(b"\xff\xd8"):
        detected = _jpeg_dimensions(data)
    elif data.startswith((b"GIF87a", b"GIF89a")):
        detected = _gif_dimensions(data)
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        detected = _webp_dimensions(data)
    if detected is None:
        raise ImageInputError(
            "unsupported image media type; use PNG, JPEG, GIF, or WebP"
        )
    media_type, width, height = detected
    if width < 1 or height < 1:
        raise ImageInputError("image dimensions must be positive")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ImageInputError(
            "image dimensions exceed the supported limit of "
            f"{MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
        )
    return media_type, width, height


def _png_dimensions(data: bytes) -> tuple[str, int, int] | None:
    if (
        len(data) < 33
        or data[8:12] != b"\x00\x00\x00\r"
        or data[12:16] != b"IHDR"
    ):
        return None
    return (
        "image/png",
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


def _jpeg_dimensions(data: bytes) -> tuple[str, int, int] | None:
    offset = 2
    start_of_frame = frozenset({
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    })
    while offset < len(data):
        if data[offset] != 0xFF:
            return None
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            return None
        marker = data[offset]
        offset += 1
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if marker in {0xD9, 0xDA} or offset + 2 > len(data):
            return None
        segment_length = int.from_bytes(data[offset:offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            return None
        if marker in start_of_frame:
            if segment_length < 7:
                return None
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            return "image/jpeg", width, height
        offset += segment_length
    return None


def _gif_dimensions(data: bytes) -> tuple[str, int, int] | None:
    if len(data) < 10:
        return None
    return (
        "image/gif",
        int.from_bytes(data[6:8], "little"),
        int.from_bytes(data[8:10], "little"),
    )


def _webp_dimensions(data: bytes) -> tuple[str, int, int] | None:
    if len(data) < 20:
        return None
    declared_size = int.from_bytes(data[4:8], "little") + 8
    if declared_size > len(data):
        return None
    chunk_type = data[12:16]
    chunk_size = int.from_bytes(data[16:20], "little")
    chunk = data[20:20 + chunk_size]
    if len(chunk) != chunk_size:
        return None
    if chunk_type == b"VP8X" and len(chunk) >= 10:
        width = int.from_bytes(chunk[4:7], "little") + 1
        height = int.from_bytes(chunk[7:10], "little") + 1
        return "image/webp", width, height
    if chunk_type == b"VP8L" and len(chunk) >= 5 and chunk[0] == 0x2F:
        bits = int.from_bytes(chunk[1:5], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return "image/webp", width, height
    if (
        chunk_type == b"VP8 "
        and len(chunk) >= 10
        and chunk[3:6] == b"\x9d\x01\x2a"
    ):
        width = int.from_bytes(chunk[6:8], "little") & 0x3FFF
        height = int.from_bytes(chunk[8:10], "little") & 0x3FFF
        return "image/webp", width, height
    return None


def _bounded_display_name(path: Path, image_number: int) -> str:
    raw = path.name or f"image-{image_number}"
    safe = "".join(
        "_"
        if not character.isprintable() or character in {"/", "\\"}
        else character
        for character in raw
    ).strip()
    if not safe:
        safe = f"image-{image_number}"
    return safe[:MAX_DISPLAY_NAME_CHARS]


def _prompt_metadata(prompt_text: str) -> dict[str, object]:
    encoded = prompt_text.encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "utf8_bytes": len(encoded),
        "chars": len(prompt_text),
    }


def _segment_record(
    segment_number: int,
    prompt_meta: dict[str, object],
    images: Sequence[PendingImage],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for image_number, image in enumerate(images, start=1):
        extension = _MEDIA_EXTENSIONS[image.media_type]
        relative_path = (
            f"attachments/segment-{segment_number:04d}/"
            f"image-{image_number:04d}{extension}"
        )
        records.append({
            "image_number": image_number,
            "display_name": image.display_name,
            "media_type": image.media_type,
            "size_bytes": image.size_bytes,
            "sha256": image.sha256,
            "width": image.width,
            "height": image.height,
            "relative_path": relative_path,
        })
    return {
        "segment_number": segment_number,
        "user_text": prompt_meta,
        "images": records,
    }


def _empty_manifest() -> dict[str, object]:
    return {
        "schema": ATTACHMENT_SCHEMA,
        "schema_version": ATTACHMENT_SCHEMA_VERSION,
        "segments": [],
    }


def _load_manifest(artifact_dir: Path) -> dict:
    path = artifact_dir / "attachments.json"
    try:
        os.lstat(path)
    except FileNotFoundError:
        return _empty_manifest()
    except OSError as exc:
        raise ImageInputError("attachment manifest is not readable") from exc
    raw = _read_regular_file(
        path,
        _MAX_MANIFEST_BYTES,
        missing_message="attachment manifest is not readable",
        non_regular_message="attachment manifest is not a regular file",
        symlink_message="attachment manifest cannot be a symbolic link",
        too_large_message="attachment manifest is too large",
    )
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImageInputError("attachment manifest is malformed") from exc
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise ImageInputError("attachment manifest is malformed")
    if (
        manifest.get("schema") != ATTACHMENT_SCHEMA
        or manifest.get("schema_version") != ATTACHMENT_SCHEMA_VERSION
        or not isinstance(manifest.get("segments"), list)
    ):
        raise ImageInputError("attachment manifest has an unsupported schema")

    seen_segments: set[int] = set()
    image_count = 0
    total_bytes = 0
    for segment in manifest["segments"]:
        if not isinstance(segment, dict):
            raise ImageInputError("attachment manifest segment is malformed")
        segment_number = segment.get("segment_number")
        if (
            isinstance(segment_number, bool)
            or not isinstance(segment_number, int)
            or segment_number < 1
            or segment_number in seen_segments
        ):
            raise ImageInputError("attachment manifest segment number is invalid")
        seen_segments.add(segment_number)
        prompt = segment.get("user_text")
        if not isinstance(prompt, dict):
            raise ImageInputError("attachment manifest user text is malformed")
        _validate_sha256(prompt.get("sha256"), "user text")
        for field in ("utf8_bytes", "chars"):
            value = prompt.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ImageInputError(
                    f"attachment manifest user text {field} is invalid"
                )
        images = segment.get("images")
        if not isinstance(images, list) or not images:
            raise ImageInputError("attachment manifest image list is malformed")
        for expected_number, image in enumerate(images, start=1):
            if not isinstance(image, dict):
                raise ImageInputError("attachment manifest image is malformed")
            if image.get("image_number") != expected_number:
                raise ImageInputError("attachment manifest image number is invalid")
            display_name = image.get("display_name")
            if (
                not isinstance(display_name, str)
                or not display_name
                or len(display_name) > MAX_DISPLAY_NAME_CHARS
                or any(
                    not character.isprintable()
                    for character in display_name
                )
                or "/" in display_name
                or "\\" in display_name
            ):
                raise ImageInputError("attachment display name is invalid")
            media_type = image.get("media_type")
            if media_type not in _MEDIA_EXTENSIONS:
                raise ImageInputError("attachment media type is invalid")
            size_bytes = image.get("size_bytes")
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 1
                or size_bytes > MAX_IMAGE_BYTES
            ):
                raise ImageInputError("attachment size is invalid")
            _validate_sha256(image.get("sha256"), "image")
            for field in ("width", "height"):
                value = image.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    or value > MAX_IMAGE_DIMENSION
                ):
                    raise ImageInputError(
                        f"attachment {field} is invalid"
                    )
            expected_path = (
                f"attachments/segment-{segment_number:04d}/"
                f"image-{expected_number:04d}{_MEDIA_EXTENSIONS[media_type]}"
            )
            relative_path = image.get("relative_path")
            if (
                relative_path != expected_path
                or not _safe_relative_path(relative_path)
            ):
                raise ImageInputError("attachment saved path is invalid")
            image_count += 1
            total_bytes += size_bytes
    if image_count > MAX_IMAGE_COUNT or total_bytes > MAX_IMAGE_TOTAL_BYTES:
        raise ImageInputError("attachment manifest exceeds saved-session limits")


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(
        part not in {"", ".", ".."} for part in path.parts
    )


def _validate_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ImageInputError(f"attachment manifest {label} digest is invalid")


def _write_new_private_file(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count < 1:
                raise ImageInputError("failed to save image evidence")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_manifest(artifact_dir: Path, manifest: dict) -> None:
    _validate_manifest(manifest)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ImageInputError("attachment manifest is too large")
    path = artifact_dir / "attachments.json"
    temporary = artifact_dir / f".attachments.{uuid.uuid4().hex}.tmp"
    _write_new_private_file(temporary, payload)
    try:
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "ATTACHMENT_SCHEMA",
    "ATTACHMENT_SCHEMA_VERSION",
    "ImageEvidence",
    "ImageInputError",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_COUNT",
    "MAX_IMAGE_DIMENSION",
    "MAX_IMAGE_TOTAL_BYTES",
    "PendingImage",
    "SessionImage",
    "image_evidence",
    "load_session_images",
    "read_image_inputs",
    "save_image_segment",
]
