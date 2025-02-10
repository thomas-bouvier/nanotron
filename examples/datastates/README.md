# DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models

Paper: https://arxiv.org/abs/2406.10707

## Installation

DataStates currently on a C++ extension living in an external repository. Eventually, this extension could be integrated into the Nanotron codebase.

Install the DataStates C++ extension as follows:

```bash
git clone https://github.com/korovod/datastates.git
cd datastates
pip install . -v
```

To enable DataStates in Nanotron, just add the following key in your YAML config:

```yaml
checkpointing_engine: datastates
```

If you want to tweak the DataStates engine, edit the following config:

```yaml
datastates:
  host_cache_size: 16
  parser_threads: 2
  pin_host_cache: true
```
