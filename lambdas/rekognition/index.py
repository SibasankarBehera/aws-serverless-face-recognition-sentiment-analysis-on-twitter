# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import os
import logging
from time import sleep

import boto3
import requests
from aws_embedded_metrics import metric_scope
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

from adapter import (
    SCHEMA_VERSION,
    build_moderated_contract,
    build_success_contract,
)

patch_all()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
rek = boto3.client('rekognition')

S3Bucket = os.getenv('Bucket')
s3_bucket = S3Bucket

# --- Configuration ---------------------------------------------------------
# The set of facial attributes to request from DetectFaces is configurable via
# the REKOGNITION_FACE_ATTRIBUTES env var (comma-separated). This lets operators
# opt into richer attributes (e.g. FACE_OCCLUDED, EYE_DIRECTION) without a code
# change. Invalid tokens are dropped with a warning. The default preserves the
# previous behavior of requesting the full attribute set ("ALL").
#
# Valid values per the latest DetectFaces API:
#   DEFAULT | ALL | AGE_RANGE | BEARD | EMOTIONS | EYE_DIRECTION | EYEGLASSES |
#   EYES_OPEN | GENDER | MOUTH_OPEN | MUSTACHE | FACE_OCCLUDED | SMILE | SUNGLASSES
VALID_FACE_ATTRIBUTES = {
    "DEFAULT", "ALL", "AGE_RANGE", "BEARD", "EMOTIONS", "EYE_DIRECTION",
    "EYEGLASSES", "EYES_OPEN", "GENDER", "MOUTH_OPEN", "MUSTACHE",
    "FACE_OCCLUDED", "SMILE", "SUNGLASSES",
}
# Downstream processFaces requires GENDER, EMOTIONS, AGE_RANGE and the default
# BoundingBox/Confidence. "ALL" satisfies all of these, matching legacy output.
_DEFAULT_FACE_ATTRIBUTES = "ALL"

MAX_RETRIES = int(os.getenv("REKOGNITION_MAX_RETRIES", "5"))


def get_face_attributes():
    """Parse and validate the configured DetectFaces attribute set."""
    raw = os.getenv("REKOGNITION_FACE_ATTRIBUTES", _DEFAULT_FACE_ATTRIBUTES)
    requested = [tok.strip().upper() for tok in raw.split(",") if tok.strip()]

    valid = []
    for tok in requested:
        if tok in VALID_FACE_ATTRIBUTES:
            if tok not in valid:
                valid.append(tok)
        else:
            logger.warning("Ignoring invalid Rekognition face attribute: %s", tok)

    if not valid:
        logger.warning(
            "No valid face attributes configured; falling back to %s",
            _DEFAULT_FACE_ATTRIBUTES,
        )
        valid = [_DEFAULT_FACE_ATTRIBUTES]
    return valid


def get_moderation_min_confidence():
    """Minimum confidence for moderation labels (configurable, default 50)."""
    try:
        return float(os.getenv("MODERATION_MIN_CONFIDENCE", "50"))
    except ValueError:
        logger.warning("Invalid MODERATION_MIN_CONFIDENCE; using default 50")
        return 50.0


def _call_with_backoff(subsegment_name, fn, error_label):
    """Invoke a Rekognition call with exponential backoff.

    Returns a tuple (response, error_result). On success error_result is None;
    on exhaustion of retries response is None and error_result is the Fail dict.
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            xray_recorder.begin_subsegment(subsegment_name)
            response = fn()
            xray_recorder.end_subsegment()
            return response, None
        except Exception as e:  # noqa: BLE001 - preserve legacy broad handling
            xray_recorder.end_subsegment()
            logger.error(str(e))
            retries += 1
            backoff = pow(2, retries) * 0.15
            logger.info("sleeps: " + str(backoff))
            sleep(backoff)
            if retries == MAX_RETRIES:
                logger.error("Error: %s backoff limit", error_label)
                return None, {
                    'result': 'Fail',
                    'schema_version': SCHEMA_VERSION,
                    'msg': str(e) + " " + error_label + " backoff limit",
                }
    return None, {
        'result': 'Fail',
        'schema_version': SCHEMA_VERSION,
        'msg': error_label + " backoff limit",
    }


@metric_scope
def handler(event, context, metrics):
    image_url = event["image_url"]
    tweet_id = event["tweet_id"]
    full_text = event["full_text"]

    r = requests.get(image_url, allow_redirects=True)

    face_attributes = get_face_attributes()
    min_confidence = get_moderation_min_confidence()

    # --- Content moderation (latest DetectModerationLabels) ----------------
    mod_response, err = _call_with_backoff(
        '## Moderation',
        lambda: rek.detect_moderation_labels(
            Image={'Bytes': r.content},
            MinConfidence=min_confidence,
        ),
        "Moderation",
    )
    if err is not None:
        return err

    if len(mod_response.get("ModerationLabels", [])) != 0:
        metrics.set_namespace('TwitterRekognition')
        metrics.put_metric("ImagesModerated", 1, "Count")
        metrics.set_property("RequestId", context.aws_request_id)
        metrics.set_property("Labels", mod_response["ModerationLabels"])
        metrics.set_property("ModerationModelVersion",
                             mod_response.get("ModerationModelVersion"))
        return build_moderated_contract(
            image_url=image_url,
            tweet_id=tweet_id,
            full_text=full_text,
            mod_response=mod_response,
        )

    # --- Face analysis (latest DetectFaces) --------------------------------
    rek_response, err = _call_with_backoff(
        '## DetectFaces',
        lambda: rek.detect_faces(
            Image={"Bytes": r.content},
            Attributes=face_attributes,
        ),
        "FaceDetect",
    )
    if err is not None:
        return err

    face_details = rek_response.get("FaceDetails", [])
    logger.info("FaceDetails: " + str(len(face_details)))

    if len(face_details) > 0:
        faces_count = len(face_details)
        contract = build_success_contract(
            image_url=image_url,
            tweet_id=tweet_id,
            full_text=full_text,
            face_details=face_details,
            mod_response=mod_response,
            requested_attributes=face_attributes,
        )
        # Preserve the legacy wire contract: `data` is a JSON string that the
        # downstream processFaces consumer deserializes via json.loads().
        contract["data"] = json.dumps(contract["data"])
        return contract

    logger.error('Unable to rekognize any face')
    return {
        'result': 'Fail',
        'schema_version': SCHEMA_VERSION,
        'msg': 'Unable to rekognize face',
    }
