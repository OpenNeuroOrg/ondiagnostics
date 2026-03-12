"""GraphQL client for  OpenNeuro datasets."""

from dataclasses import dataclass

import cattrs
import httpx
import stamina
import gql
from gql.transport.httpx import HTTPXAsyncTransport
from gql.transport.exceptions import TransportQueryError

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import AsyncIterator

converter: cattrs.Converter = cattrs.Converter()


@dataclass
class Dataset:
    id: str
    tag: str
    hexsha: str = "unknown"


@dataclass
class LatestSnapshot:
    tag: str
    created: str
    hexsha: str


@dataclass
class DatasetNode:
    id: str
    latestSnapshot: LatestSnapshot


@dataclass
class DatasetEdge:
    node: DatasetNode


@dataclass
class PageInfo:
    hasNextPage: bool = True
    endCursor: str | None = None
    count: int = 0


@dataclass
class DatasetsResponse:
    edges: list[DatasetEdge | None]
    pageInfo: PageInfo


@dataclass
class GraphQLResponse:
    datasets: DatasetsResponse


@dataclass
class SingleDatasetResponse:
    dataset: DatasetNode | None


ENDPOINT = "https://openneuro.org/crn/graphql"
GET_DATASETS: gql.GraphQLRequest = gql.gql("""
query DatasetsWithLatestSnapshots($count: Int, $after: String) {
  datasets(
    first: $count,
    after: $after,
    orderBy: {created: ascending}
    filterBy: {public: true}
  ) {
    edges {
      node {
        id
        latestSnapshot {
          tag
          created
          hexsha
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
      count
    }
  }
}
""")
GET_SINGLE_DATASET: gql.GraphQLRequest = gql.gql("""
query GetDataset($id: ID!) {
  dataset(id: $id) {
    id
    latestSnapshot {
      tag
      created
      hexsha
    }
  }
}
""")


def create_client() -> gql.Client:
    return gql.Client(transport=HTTPXAsyncTransport(url=ENDPOINT))


@stamina.retry(on=httpx.HTTPError)
async def get_page(
    client: gql.Client, count: int, after: str | None
) -> GraphQLResponse:
    """Fetch a page of datasets from the GraphQL API."""
    result = await client.execute_async(
        GET_DATASETS, variable_values={"count": count, "after": after}
    )
    return converter.structure(result, GraphQLResponse)


async def get_dataset_count(client: gql.Client) -> int:
    """Get the total count of datasets."""
    response = await get_page(client, 0, None)
    return response.datasets.pageInfo.count


async def datasets_generator(client: gql.Client) -> AsyncIterator[Dataset]:
    """
    Async generator that yields datasets from the GraphQL API.

    Args:
        client: GraphQL client

    Yields:
        Dataset objects
    """
    page_info = PageInfo()

    while page_info.hasNextPage:
        try:
            result = await get_page(client, 100, page_info.endCursor)
        except TransportQueryError as e:
            import structlog

            logger = structlog.get_logger()
            if e.data is not None:
                logger.warning("GraphQL query error, missing dataset")
                result = converter.structure(e.data, GraphQLResponse)
            else:
                logger.critical("GraphQL query error, cannot continue")
                break

        page_info = result.datasets.pageInfo

        for edge in result.datasets.edges:
            if edge is None:
                continue

            yield Dataset(
                id=edge.node.id,
                tag=edge.node.latestSnapshot.tag,
                hexsha=edge.node.latestSnapshot.hexsha,
            )


@stamina.retry(on=httpx.HTTPError)
async def get_dataset(client: gql.Client, dataset_id: str) -> Dataset | None:
    """
    Fetch a single dataset by ID from the GraphQL API.

    Args:
        client: GraphQL client
        dataset_id: Dataset ID to fetch

    Returns:
        Dataset object or None if not found
    """
    try:
        result = await client.execute_async(
            GET_SINGLE_DATASET, variable_values={"id": dataset_id}
        )
    except TransportQueryError:
        return None

    response = converter.structure(result, SingleDatasetResponse)

    if response.dataset is None:
        return None

    return Dataset(
        id=response.dataset.id,
        tag=response.dataset.latestSnapshot.tag,
        hexsha=response.dataset.latestSnapshot.hexsha,
    )


async def datasets_by_ids_generator(
    client: gql.Client, dataset_ids: list[str]
) -> AsyncIterator[Dataset]:
    """
    Async generator that yields specific datasets by ID.

    Args:
        client: GraphQL client
        dataset_ids: List of dataset IDs to fetch

    Yields:
        Dataset objects (skips datasets that don't exist)
    """
    for dataset_id in dataset_ids:
        dataset = await get_dataset(client, dataset_id)
        if dataset is None:
            import structlog

            logger = structlog.get_logger()
            logger.error("Dataset not found or has no snapshots", dataset_id=dataset_id)
        else:
            yield dataset
