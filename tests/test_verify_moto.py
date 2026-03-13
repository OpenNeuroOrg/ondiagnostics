"""Verification tests that aiomoto behaves like real S3."""

import pytest
import aioboto3
from aiomoto import mock_aws
from botocore import UNSIGNED
from aiobotocore.config import AioConfig

CONFIG = AioConfig(signature_version=UNSIGNED)


# @pytest.mark.integration
async def test_real_s3_openneuro_accessible() -> None:
    """Verify we can access real OpenNeuro S3 bucket (for test validity)."""
    session = aioboto3.Session()

    try:
        async with session.client("s3", region_name="us-east-1", config=CONFIG) as s3:
            # Just check that ds000001 prefix exists
            response = await s3.list_objects_v2(
                Bucket="openneuro.org", Prefix="ds000001/", MaxKeys=1
            )
            assert "Contents" in response
            assert len(response["Contents"]) > 0
    except Exception as e:
        pytest.xfail(f"Could not access OpenNeuro S3: {e}")


# @pytest.mark.integration
async def test_aiomoto_pagination_structure_matches_real() -> None:
    """Verify aiomoto's pagination structure matches real S3 structure."""

    # Get real S3 response structure
    real_session = aioboto3.Session()
    try:
        async with real_session.client(
            "s3", region_name="us-east-1", config=CONFIG
        ) as s3:
            real_response = await s3.list_objects_v2(
                Bucket="openneuro.org", Prefix="ds000001/", MaxKeys=5
            )
    except Exception as e:
        pytest.xfail(f"Could not access real S3: {e}")

    # Get aiomoto response structure
    async with mock_aws():
        aiomoto_session = aioboto3.Session()
        async with aiomoto_session.client(
            "s3", region_name="us-east-1", config=CONFIG
        ) as s3:
            await s3.create_bucket(Bucket="test-bucket")

            # Add some objects
            for i in range(10):
                await s3.put_object(
                    Bucket="test-bucket", Key=f"ds000001/file{i}.txt", Body=b"content"
                )

            aiomoto_response = await s3.list_objects_v2(
                Bucket="test-bucket", Prefix="ds000001/", MaxKeys=5
            )

    # Verify both have same response structure
    assert "Contents" in real_response
    assert "Contents" in aiomoto_response

    # Both should have pagination fields
    for response in [real_response, aiomoto_response]:
        assert "IsTruncated" in response
        assert "KeyCount" in response
        assert "MaxKeys" in response

    # Verify Contents structure
    real_obj = real_response["Contents"][0]
    aiomoto_obj = aiomoto_response["Contents"][0]

    # Both should have same keys
    # moto doesn't drop ChecksumAlgorithm for unsigned config, but that's okay
    assert set(real_obj.keys()) ^ set(aiomoto_obj.keys()) <= {"ChecksumAlgorithm"}


async def test_aiomoto_delete_behavior() -> None:
    """Verify aiomoto's delete_objects behaves sensibly."""

    async with mock_aws():
        session = aioboto3.Session()
        async with session.client("s3", region_name="us-east-1") as s3:
            await s3.create_bucket(Bucket="test-bucket")

            # Create objects
            for i in range(5):
                await s3.put_object(
                    Bucket="test-bucket", Key=f"file{i}.txt", Body=b"content"
                )

            # Delete some (including non-existent)
            response = await s3.delete_objects(
                Bucket="test-bucket",
                Delete={
                    "Objects": [
                        {"Key": "file0.txt"},
                        {"Key": "file1.txt"},
                        {"Key": "nonexistent.txt"},  # Shouldn't error
                    ]
                },
            )

            # Verify response structure
            assert "Deleted" in response
            assert len(response["Deleted"]) >= 2  # At least the 2 that existed

            # Verify deleted files have correct structure
            for deleted in response["Deleted"]:
                assert "Key" in deleted

            # Verify files are actually gone
            list_response = await s3.list_objects_v2(Bucket="test-bucket")
            remaining_keys = [obj["Key"] for obj in list_response.get("Contents", [])]

            assert "file0.txt" not in remaining_keys
            assert "file1.txt" not in remaining_keys
            assert "file2.txt" in remaining_keys


async def test_aiomoto_paginator_behavior() -> None:
    """Verify aiomoto's paginator works as expected."""

    async with mock_aws():
        session = aioboto3.Session()
        async with session.client("s3", region_name="us-east-1") as s3:
            await s3.create_bucket(Bucket="test-bucket")

            # Create 25 objects
            for i in range(25):
                await s3.put_object(
                    Bucket="test-bucket", Key=f"file{i:02d}.txt", Body=b"content"
                )

            # Test paginator
            paginator = s3.get_paginator("list_objects_v2")
            pages = []

            async for page in paginator.paginate(
                Bucket="test-bucket", PaginationConfig={"PageSize": 10}
            ):
                if "Contents" in page:
                    pages.append(page["Contents"])

            # Should have 3 pages (10 + 10 + 5)
            assert len(pages) == 3
            assert len(pages[0]) == 10
            assert len(pages[1]) == 10
            assert len(pages[2]) == 5

            # All keys should be present
            all_keys = [obj["Key"] for page in pages for obj in page]
            assert len(all_keys) == 25
