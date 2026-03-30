"""
AWS utility functions for S3 model upload/download.
Uses boto3 with environment-based credential handling.
"""

import logging
import os

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from config import AWS_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)


def _get_s3_client():
    """
    Create and return a boto3 S3 client.

    Returns:
        boto3 S3 client instance.

    Raises:
        NoCredentialsError: If AWS credentials are not configured.
    """
    return boto3.client(
        "s3",
        region_name=AWS_CONFIG.region,
        aws_access_key_id=AWS_CONFIG.aws_access_key or None,
        aws_secret_access_key=AWS_CONFIG.aws_secret_key or None,
    )


def upload_model_to_s3(local_path: str, s3_key: str = None) -> bool:
    """
    Upload a local model archive to S3.

    Args:
        local_path: Path to the local .tar.gz archive.
        s3_key: Optional S3 object key. Defaults to config value.

    Returns:
        True if successful, False otherwise.
    """
    s3_key = s3_key or MODEL_CONFIG.s3_model_key
    client = _get_s3_client()

    try:
        file_size = os.path.getsize(local_path)
        logger.info(
            "Uploading %s (%.2f MB) → s3://%s/%s",
            local_path,
            file_size / 1e6,
            AWS_CONFIG.s3_bucket,
            s3_key,
        )
        client.upload_file(
            local_path,
            AWS_CONFIG.s3_bucket,
            s3_key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        logger.info("Upload complete.")
        return True
    except FileNotFoundError:
        logger.error("Local file not found: %s", local_path)
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not configured.")
        return False
    except ClientError as exc:
        logger.error("S3 upload failed: %s", exc)
        return False


def download_model_from_s3(local_path: str, s3_key: str = None) -> bool:
    """
    Download model archive from S3 to a local path.

    Args:
        local_path: Destination local file path.
        s3_key: Optional S3 object key. Defaults to config value.

    Returns:
        True if successful, False otherwise.
    """
    s3_key = s3_key or MODEL_CONFIG.s3_model_key
    client = _get_s3_client()

    try:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        logger.info(
            "Downloading s3://%s/%s → %s",
            AWS_CONFIG.s3_bucket,
            s3_key,
            local_path,
        )
        client.download_file(AWS_CONFIG.s3_bucket, s3_key, local_path)
        logger.info("Download complete.")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not configured.")
        return False
    except ClientError as exc:
        logger.error("S3 download failed: %s", exc)
        return False


def list_s3_models(prefix: str = "models/") -> list:
    """
    List all model objects stored under a given S3 prefix.

    Args:
        prefix: S3 key prefix to filter results.

    Returns:
        List of S3 object key strings.
    """
    client = _get_s3_client()
    try:
        response = client.list_objects_v2(
            Bucket=AWS_CONFIG.s3_bucket, Prefix=prefix
        )
        return [obj["Key"] for obj in response.get("Contents", [])]
    except ClientError as exc:
        logger.error("Failed to list S3 objects: %s", exc)
        return []