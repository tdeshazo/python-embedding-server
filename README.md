# Python Embedding Server

## Install

```bash
pip install -e ".[json]"
pip install -e ".[grpc]"
```

## Run with uvicorn

```bash
uvicorn python_encoder_server.main:app --host 0.0.0.0 --port 8000
```

Factory mode is also supported:

```bash
uvicorn python_encoder_server.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Run gRPC Service

```bash
python-embeddings-grpc
```

Default bind is `0.0.0.0:50051`. Override with:

```bash
GRPC_BIND_ADDR=0.0.0.0:50052 python-embeddings-grpc
```

Proto contract:

```text
src/python_encoder_server/grpc/embeddings.proto
```

## Run with Docker

```bash
docker build -t python-embedding-server .
docker run --rm -p 8000:8000 python-embedding-server
```

## Run gRPC with Docker

```bash
docker build -f Dockerfile.grpc -t python-embedding-grpc-server .
docker run --rm -p 50051:50051 python-embedding-grpc-server
```

## Request Concurrency

Requests that set `max_length` run exclusively while temporarily overriding the
model sequence length. Requests without `max_length` still use the configured
`MAX_INFERENCE_CONCURRENCY`.

## Go Request Struct Example

```go
type EmbedRequest struct {
	Texts     []string `json:"texts"`
	Normalize bool     `json:"normalize"`
	BatchSize int      `json:"batch_size"`
	MaxLength *int     `json:"max_length,omitempty"`
}
```

## Go Response Struct Example

```go
type EmbedResponse struct {
	Model      string      `json:"model"`
	Dim        int         `json:"dim"`
	Embeddings [][]float32 `json:"embeddings"`
}
```
