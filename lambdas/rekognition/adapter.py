# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Adapter layer that maps the latest Amazon Rekognition Image API responses
(``DetectFaces`` and ``DetectModerationLabels``) into a stable, *versioned*
response contract.

Why an adapter?
---------------
The raw Rekognition response shape evolves over time (new facial attributes such
as ``FaceOccluded`` / ``EyeDirection``, richer moderation metadata such as
``TaxonomyLevel`` / ``ContentTypes`` / ``ProjectVersion``). Downstream consumers
in this project (``processFaces``) historically read a small subset of fields
directly off each Rekognition ``FaceDetail`` record:

    Confidence, Gender.Value, Emotions, AgeRange, BoundingBox

To let new consumers opt into the richer data *without breaking* the older
integrations, this module produces a single envelope that:

  * Keeps the legacy face-record fields byte-for-byte where possible. The face
    records under ``facerecords`` remain raw Rekognition ``FaceDetail`` dicts,
    so existing consumers keep working unchanged.
  * Adds a ``schema_version`` so consumers can branch on the contract version.
  * Adds a normalized ``analysis`` block (faces + moderation) that new consumers
    can read from, surfacing the newer attributes in a predictable structure.

Consumers should treat ``schema_version`` as the compatibility signal:
  * ``processFaces`` (legacy) ignores everything except ``facerecords`` and the
    top-level ``image_url`` / ``full_text`` / ``tweet_id`` fields.
  * New consumers read ``analysis`` and may rely on ``analysis.faces[*].occluded``,
    ``analysis.faces[*].eye_direction``, ``analysis.moderation.taxonomy_level``,
    etc.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Bump the MINOR version when adding backward-compatible fields to the contract.
# Bump the MAJOR version only for a breaking change to existing fields/semantics.
SCHEMA_VERSION = "1.1.0"


def _num(value: Any) -> Optional[float]:
    """Best-effort numeric coercion; returns None when not convertible."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_emotion(emotions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the highest-confidence emotion, or None."""
    if not emotions:
        return None
    ranked = sorted(emotions, key=lambda e: _num(e.get("Confidence")) or 0.0, reverse=True)
    top = ranked[0]
    return {"type": top.get("Type"), "confidence": _num(top.get("Confidence"))}


def normalize_face(face: Dict[str, Any]) -> Dict[str, Any]:
    """Map a raw Rekognition ``FaceDetail`` into the normalized ``analysis`` shape.

    This surfaces both legacy and newer attributes. Every field is optional and
    defaults to ``None`` when the attribute was not requested/returned, so a
    consumer reading this block never has to guard against missing keys.
    """
    gender = face.get("Gender") or {}
    age_range = face.get("AgeRange") or {}
    occluded = face.get("FaceOccluded") or {}
    eye_direction = face.get("EyeDirection") or {}
    quality = face.get("Quality") or {}

    return {
        "confidence": _num(face.get("Confidence")),
        "bounding_box": face.get("BoundingBox"),
        "gender": {
            "value": gender.get("Value"),
            "confidence": _num(gender.get("Confidence")),
        },
        "age_range": {
            "low": age_range.get("Low"),
            "high": age_range.get("High"),
        },
        "emotions": [
            {"type": e.get("Type"), "confidence": _num(e.get("Confidence"))}
            for e in (face.get("Emotions") or [])
        ],
        "primary_emotion": _top_emotion(face.get("Emotions") or []),
        # Newer DetectFaces attributes (opt-in for new consumers).
        "occluded": {
            "value": occluded.get("Value"),
            "confidence": _num(occluded.get("Confidence")),
        } if occluded else None,
        "eye_direction": {
            "yaw": _num(eye_direction.get("Yaw")),
            "pitch": _num(eye_direction.get("Pitch")),
            "confidence": _num(eye_direction.get("Confidence")),
        } if eye_direction else None,
        "quality": {
            "brightness": _num(quality.get("Brightness")),
            "sharpness": _num(quality.get("Sharpness")),
        } if quality else None,
        "smile": (face.get("Smile") or {}).get("Value"),
        "eyeglasses": (face.get("Eyeglasses") or {}).get("Value"),
        "sunglasses": (face.get("Sunglasses") or {}).get("Value"),
    }


def normalize_moderation(mod_response: Dict[str, Any]) -> Dict[str, Any]:
    """Map a raw ``DetectModerationLabels`` response into the normalized shape.

    Surfaces the newer moderation metadata: ``TaxonomyLevel`` per label,
    ``ContentTypes`` predictions, the base ``ModerationModelVersion``, and the
    custom-adapter ``ProjectVersion`` when present.
    """
    labels = mod_response.get("ModerationLabels") or []
    return {
        "flagged": len(labels) > 0,
        "labels": [
            {
                "name": lbl.get("Name"),
                "confidence": _num(lbl.get("Confidence")),
                "parent_name": lbl.get("ParentName"),
                "taxonomy_level": lbl.get("TaxonomyLevel"),
            }
            for lbl in labels
        ],
        "content_types": [
            {"name": ct.get("Name"), "confidence": _num(ct.get("Confidence"))}
            for ct in (mod_response.get("ContentTypes") or [])
        ],
        "model_version": mod_response.get("ModerationModelVersion"),
        "project_version": mod_response.get("ProjectVersion"),
    }


def build_moderated_contract(
    *,
    image_url: str,
    tweet_id: str,
    full_text: str,
    mod_response: Dict[str, Any],
) -> Dict[str, Any]:
    """Envelope returned when an image is moderated (blocked)."""
    return {
        "result": "Moderated",
        "schema_version": SCHEMA_VERSION,
        "image_url": image_url,
        "tweet_id": tweet_id,
        "full_text": full_text,
        "analysis": {
            "faces": [],
            "moderation": normalize_moderation(mod_response),
        },
    }


def build_success_contract(
    *,
    image_url: str,
    tweet_id: str,
    full_text: str,
    face_details: List[Dict[str, Any]],
    mod_response: Optional[Dict[str, Any]] = None,
    requested_attributes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the success envelope.

    Backward compatibility contract:
      * ``result`` == "Succeed" and ``count`` (stringified) are preserved.
      * ``data`` is a JSON-serializable dict containing the legacy keys
        ``image_url``, ``full_text``, ``facerecords`` (raw FaceDetail list) and
        ``tweet_id`` exactly as the previous implementation emitted them, so the
        existing ``processFaces`` consumer keeps working with no change.
      * ``schema_version`` and ``analysis`` are additive, for new consumers.
    """
    legacy_data: Dict[str, Any] = {
        "image_url": str(image_url),
        "full_text": full_text,
        "facerecords": face_details,
        "tweet_id": tweet_id,
        # Additive fields inside the legacy payload for opt-in consumers.
        "schema_version": SCHEMA_VERSION,
        "requested_attributes": requested_attributes or [],
        "analysis": {
            "faces": [normalize_face(f) for f in face_details],
            "moderation": normalize_moderation(mod_response) if mod_response else None,
        },
    }

    return {
        "result": "Succeed",
        "schema_version": SCHEMA_VERSION,
        "count": str(len(face_details)),
        "data": legacy_data,
    }
