import os
from dataclasses import dataclass
from pathlib import Path

import aioboto3


@dataclass
class AWSConfig:
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_S3_BUCKET_NAME: str
    AWS_REGION: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> AWSConfig:
        return cls(
            AWS_ACCESS_KEY_ID=data["AWS_ACCESS_KEY_ID"],
            AWS_SECRET_ACCESS_KEY=data["AWS_SECRET_ACCESS_KEY"],
            AWS_S3_BUCKET_NAME=data.get("AWS_S3_PUBLIC_BUCKET", "openneuro.org"),
            AWS_REGION=data.get("AWS_REGION", "us-east-1"),
        )

    @classmethod
    def from_env(cls) -> AWSConfig:
        try:
            return cls.from_dict(dict(os.environ))
        except KeyError:
            raise ValueError("AWS credentials are missing from environment variables.")

    @classmethod
    def from_file(cls, config_path: Path) -> AWSConfig:
        import yaml

        config_data: dict[str, dict[str, dict[str, str]]] = yaml.safe_load(
            Path(config_path).read_text()
        )
        try:
            return cls.from_dict(config_data["secrets"]["aws"])
        except KeyError:
            raise ValueError("AWS credentials are missing in the config file.")

    def create_session(self) -> aioboto3.Session:
        """Create an aioboto3 Session with configured credentials."""
        return aioboto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_REGION,
        )
