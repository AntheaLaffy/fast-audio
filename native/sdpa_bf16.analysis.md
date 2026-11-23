## Feature Index

| Lines | Feature |
| --- | --- |
| 1–151 | Tensor Core fragments, shared-memory swizzle and tile loads |
| 152–383 | Tensor Core forward |
| 384–569 | Tensor Core query gradient |
| 570–822 | Tensor Core key and value gradients |
| 823–885 | Backward row preparation and grouped-query gradient reduction |
| 886–1089 | Warp-cooperative signed-stride fallback, online output and fused backward row preparation |
| 1090–1333 | Tensor Core workspace preparation and route-specific CUDA launch dispatch |
