"""Tests for GraphQL client and data models."""

from unittest.mock import AsyncMock, Mock

import pytest
import gql
from gql.transport.exceptions import TransportQueryError

from ondiagnostics.graphql import (
    Dataset,
    GraphQLResponse,
    converter,
    get_dataset_count,
    datasets_generator,
)


def test_dataset_creation():
    """Test creating a Dataset."""
    dataset = Dataset(id="ds000001", tag="1.0.0", hexsha="abc123")
    assert dataset.id == "ds000001"
    assert dataset.tag == "1.0.0"
    assert dataset.hexsha == "abc123"

    dataset = Dataset("ds000003", "1.0.0")
    assert dataset.id == "ds000003"
    assert dataset.tag == "1.0.0"
    assert dataset.hexsha == "unknown"


def test_converter_structure():
    """Test cattrs converter structures nested data correctly."""
    data = {
        "datasets": {
            "edges": [
                {
                    "node": {
                        "id": "ds000001",
                        "latestSnapshot": {
                            "tag": "1.0.0",
                            "created": "2023-01-01",
                            "hexsha": "abc123",
                        },
                    }
                }
            ],
            "pageInfo": {"hasNextPage": False, "endCursor": None, "count": 1},
        }
    }

    response = converter.structure(data, GraphQLResponse)

    assert isinstance(response, GraphQLResponse)
    assert len(response.datasets.edges) == 1
    assert response.datasets.edges[0].node.id == "ds000001"
    assert response.datasets.edges[0].node.latestSnapshot.tag == "1.0.0"
    assert response.datasets.pageInfo.count == 1


@pytest.mark.asyncio
async def test_get_dataset_count():
    """Test getting total dataset count."""
    mock_client = Mock(
        spec_set=gql.Client,
        execute_async=AsyncMock(
            spec_set=dict,
            return_value={
                "datasets": {
                    "edges": [],
                    "pageInfo": {"hasNextPage": False, "endCursor": None, "count": 42},
                }
            },
        ),
    )

    count = await get_dataset_count(mock_client)

    assert count == 42
    mock_client.execute_async.assert_called_once()


@pytest.mark.asyncio
async def test_datasets_generator():
    """Test dataset generator yields all datasets."""

    pages = [
        {
            "datasets": {
                "edges": [
                    {
                        "node": {
                            "id": "ds000001",
                            "latestSnapshot": {
                                "tag": "1.0.0",
                                "created": "2023-01-01",
                                "hexsha": "abc123",
                            },
                        }
                    }
                ],
                "pageInfo": {"hasNextPage": True, "endCursor": "c1", "count": 5},
            }
        },
        TransportQueryError(
            "GraphQL query failed",
            data={
                "datasets": {
                    "edges": [
                        None,
                        {
                            "node": {
                                "id": "ds000002",
                                "latestSnapshot": {
                                    "tag": "2.0.0",
                                    "created": "2023-01-02",
                                    "hexsha": "def456",
                                },
                            }
                        },
                    ],
                    "pageInfo": {
                        "hasNextPage": True,
                        "endCursor": "c2",
                        "count": 5,
                    },
                }
            },
        ),
        TransportQueryError("Critical failure of some sort"),
        {
            "datasets": {
                "edges": [
                    {
                        "node": {
                            "id": "ds000003",
                            "latestSnapshot": {
                                "tag": "3.0.0",
                                "created": "2023-01-03",
                                "hexsha": "abcdef",
                            },
                        }
                    }
                ],
                "pageInfo": {"hasNextPage": False, "endCursor": None, "count": 5},
            }
        },
    ]

    mock_client = Mock(
        spec_set=gql.Client, execute_async=AsyncMock(spec_set=dict, side_effect=pages)
    )

    datasets = []
    async for dataset in datasets_generator(mock_client):
        datasets.append(dataset)

    # Nones are skipped over, and errors without data terminate
    assert len(datasets) == 2
    assert datasets[0].id == "ds000001"
    assert datasets[1].id == "ds000002"
