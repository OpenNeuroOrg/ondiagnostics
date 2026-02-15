"""Tests for AWS configuration loading."""

import os

import pytest
import yaml

from ondiagnostics.awsconfig import AWSConfig


AWS_CONFIG = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_PUBLIC_BUCKET": "test-bucket",
    "AWS_REGION": "us-west-2",
}


def test_from_dict():
    """Test creating AWSConfig from a dictionary."""
    config = AWSConfig.from_dict(AWS_CONFIG)

    assert config.AWS_ACCESS_KEY_ID == "test_key"
    assert config.AWS_SECRET_ACCESS_KEY == "test_secret"
    assert config.AWS_S3_BUCKET_NAME == "test-bucket"
    assert config.AWS_REGION == "us-west-2"


def test_from_dict_with_defaults():
    """Test that from_dict uses default values."""
    data = {
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
    }
    config = AWSConfig.from_dict(data)

    assert config.AWS_S3_BUCKET_NAME == "openneuro.org"
    assert config.AWS_REGION == "us-east-1"


def test_from_env(monkeypatch):
    """Test creating AWSConfig from environment variables."""
    monkeypatch.setattr(os, "environ", AWS_CONFIG)

    config = AWSConfig.from_env()

    assert config.AWS_ACCESS_KEY_ID == "test_key"
    assert config.AWS_SECRET_ACCESS_KEY == "test_secret"
    assert config.AWS_S3_BUCKET_NAME == "test-bucket"
    assert config.AWS_REGION == "us-west-2"


def test_from_env_missing_required(monkeypatch):
    """Test that from_env raises when required vars are missing."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

    with pytest.raises(ValueError, match="AWS credentials are missing"):
        AWSConfig.from_env()


def test_from_file(tmp_path):
    """Test creating AWSConfig from a file."""
    config_file = tmp_path / "config.yaml"
    config_data = {"secrets": {"aws": AWS_CONFIG}}
    config_file.write_text(yaml.dump(config_data))

    config = AWSConfig.from_file(config_file)

    assert config.AWS_ACCESS_KEY_ID == "test_key"
    assert config.AWS_SECRET_ACCESS_KEY == "test_secret"
    assert config.AWS_S3_BUCKET_NAME == "test-bucket"
    assert config.AWS_REGION == "us-west-2"


def test_from_file_missing_secrets(tmp_path):
    """Test that from_file raises when config is malformed."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"not_secrets": {}}))

    with pytest.raises(ValueError, match="AWS credentials are missing"):
        AWSConfig.from_file(config_file)
