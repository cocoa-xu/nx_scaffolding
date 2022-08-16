# NxScaffolding [WIP]

Benchmark numerical-elixir backend.

## Usage
Say the backend you're developing is `AAA.Backend`, then add an mix task file
```elixir
defmodule Mix.Tasks.Aaa.Benchmark do
  use Mix.Tasks.NxScaffolding.Benchmark
end
```

and run it with default configuration
```shell
mix aaa.benchmark --backend AAA.Backend

# or check for more details with
mix aaa.benchmark --help
```

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `nx_scaffolding` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:nx_scaffolding, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/nx_scaffolding>.

