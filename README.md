# ONDiagnostics

This is a package for diagnosing and fixing issues on OpenNeuro.

## Installation

```console
pip install ondiagnostics
```

## Usage

### Verify synchronization with GitHub mirrors

OpenNeuro exports published datasets to GitHub as the final action in release.
This utility identifies datasets that do not contain the latest version,
which can be indicative of a larger export failure.

```console
ondiagnostics check-sync
```

### Verify and delete excess S3 objects

OpenNeuro's export process should ensure that fetching the dataset from S3
use `aws s3 sync` fetches the latest version.
In some cases, files from previous versions are not marked as deleted, so
excess files are downloaded.

```console
ondiagnostics clean-s3
```

This is an expensive operation that will potentially clone all OpenNeuro
datasets. It also requires access to OpenNeuro AWS credentials.

## Library

This package has utilities that may be useful for building tools to query
OpenNeuro.

### Async dataset iterator

```python
from ondiagnostics.graphql import create_client, datasets_generator

client = create_client()

async for dataset in datasets_generator(client):
    ...
```

### Clone an OpenNeuro dataset

```python
from ondiagnostics.graphql import Dataset
from ondiagnostics.tasks import clone_dataset

await clone_dataset(Dataset(id='ds000001', tag='1.0.0'))
```

Note that this only performs a shallow git clone and does not initialize the annex.
