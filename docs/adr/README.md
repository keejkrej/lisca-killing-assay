# Architecture Decision Records

Why a shipped choice looks the way it does. Agents write these during the change. The procedure is the `decision-record` skill.

| Status       | Meaning                                                    |
| ------------ | ---------------------------------------------------------- |
| `accepted`   | Current choice. Matching behavior is intended.             |
| `proposed`   | Not yet confirmed. Matching behavior is not protected.     |
| `superseded` | Replaced. The old file stays, and it links to the new ADR. |

## Index

| ID                                                     | Status   | Title                                                        |
| ------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| [0001](0001-frozen-reference-classification.md)        | accepted | Compare label-free viability with a frozen encoder           |
| [0002](0002-remote-embedding-service.md)               | superseded | Serve frozen viability inference from this package         |
| [0003](0003-assay-registers-its-model.md)              | accepted   | Register EmbeddingGemma and the viability classifier       |
