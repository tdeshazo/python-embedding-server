# Python Embedding Server

## Run with uvicorn

```bash
uvicorn main:app --app-dir python-embeddings-service --host 0.0.0.0 --port 8000
```

Factory mode is also supported:

```bash
uvicorn main:create_app --factory --app-dir python-embeddings-service --host 0.0.0.0 --port 8000
```

## Run with Docker

```bash
docker build -t python-embedding-server .
docker run --rm -p 8000:8000 python-embedding-server
```

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
