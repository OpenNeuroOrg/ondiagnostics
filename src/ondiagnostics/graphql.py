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


ENDPOINT = "https://openneuro.org/crn/graphql"
QUERY: gql.GraphQLRequest = gql.gql("""
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


def create_client() -> gql.Client:
    return gql.Client(transport=HTTPXAsyncTransport(url=ENDPOINT))


@stamina.retry(on=httpx.HTTPError)
async def get_page(
    client: gql.Client, count: int, after: str | None
) -> GraphQLResponse:
    """Fetch a page of datasets from the GraphQL API."""
    result = await client.execute_async(
        QUERY, variable_values={"count": count, "after": after}
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
