defmodule Mix.Tasks.NxScaffolding.Benchmark do
  defmacro __using__(_opts) do
    quote do
      @moduledoc """
      usage:
          $ mix nx_scaffolding.benchmark --backend BACKEND [OPTIONS]

      options:
          --backend=Backend.ModuleName  Backend to benchmark.
          --warm-up=N                   Run each test `N` times before the real benchmark for warm-up.
                                        Default is `0`.
          --cooldown=C                  Wait `C` seconds between each test.
                                        Default is `0`.
          --repeat=R                    Repeat each test `R` times.
                                        Default is `5`.
          --time-unit                   Use time unit. [auto|us|ms|sec|min].
                                        Default is `auto`.
          --data-types                  Comma separated data types.
                                        Default is `u8,f32,f64`.
          --stop-on-error=[Ny]          Stop benchmarking if any runtime error is raised.
          --ignore-not-implemented=[Yn] Ignore not implemented callbacks.

      """

      use Mix.Task
      require Logger

      def run(args) do
        {opts, _, errors} = parse_args(args)
        case errors do
          [] ->
            backend = opts[:backend] || Mix.raise("not specified backend")
            backend = Elixir.Module.concat([backend])

            if Code.ensure_loaded?(backend) do
              # do this so that the backend can load its shared library
              # if applicable
              _ = Nx.tensor(0, backend: backend, type: :u8)
              do_benchmark(backend, opts)
            else
              Mix.raise("cannot find backend: #{inspect(backend)}")
            end

          _ ->
            IO.puts("Bad option:")
            IO.inspect(errors)
            IO.puts(@moduledoc)
        end
      end

      @available_benchmarks [
        "Nx.constant/3",
        "Nx.dot/2"
      ]
      defp do_benchmark(backend, opts) do
        repeat = opts[:repeat]
        stop_on_error = opts[:stop_on_error]
        warm_up = opts[:warm_up]
        time_unit = String.to_atom(opts[:time_unit])
        data_types =
          Enum.map(String.split(opts[:data_types], ",", trim: true), &String.to_atom(&1))

        data =
          Enum.reduce(@available_benchmarks, [], fn callback, acc ->
            current =
              Enum.map(data_types, fn data_type ->
                do_benchmark(
                  callback, data_type,
                  backend, warm_up, repeat,
                  stop_on_error, time_unit,
                  opts
                )
            end)
            [current | acc]
          end)
          |> List.flatten()

        Scribe.print(data, data: [:callback, :data_type, :succeeded, :mean, :std, :min, :max, :note])
      end

      defp to_time_unit(nanosecond, unit) do
        case unit do
          :us ->
            "#{Float.round(nanosecond / 1.0e3, 3)} µs"
          :ms ->
            "#{Float.round(nanosecond / 1.0e6, 3)} ms"
          :sec ->
            "#{Float.round(nanosecond / 1.0e9, 3)} sec"
          :min ->
            "#{Float.round(nanosecond / 1.0e9 / 60.0, 3)} minutes"
        end
      end

      defp use_proper_time_unit(nanosecond) when nanosecond > 1.0e9 * 60 do
        to_time_unit(nanosecond, :min)
      end

      defp use_proper_time_unit(nanosecond) when nanosecond > 1.0e9 do
        to_time_unit(nanosecond, :sec)
      end

      defp use_proper_time_unit(nanosecond) when nanosecond > 1.0e6 do
        to_time_unit(nanosecond, :ms)
      end

      defp use_proper_time_unit(nanosecond) when nanosecond > 1.0e3 do
        to_time_unit(nanosecond, :us)
      end

      defp do_benchmark(callback="Nx.dot/2", data_type, backend,
        warm_up, repeat, stop_on_error, time_unit, opts) do
        matrix_size = {100, 100}
        note = "100x100"
        previous = Nx.default_backend(backend)
        a = Nx.random_uniform(matrix_size, type: :f32, backend: backend)
        b = Nx.random_uniform(matrix_size, type: :f32, backend: backend)
        Nx.default_backend(previous)
        do_benchmark(callback, data_type, backend, warm_up, repeat,
          stop_on_error, time_unit, opts, {a, b},
          fn {a, b} ->
            time_func(fn ->
              Nx.dot(a, b)
            end)
          end,
          fn {a, b}, user_tensor ->
            bin_a = Nx.backend_copy(a, Nx.BinaryBackend)
            bin_b = Nx.backend_copy(b, Nx.BinaryBackend)
            expected_tensor = Nx.dot(bin_a, bin_b)
            verify_result(
              callback,
              data_type,
              user_tensor,
              expected_tensor,
              stop_on_error
            )
          end,
          note
        )
      end

      defp do_benchmark(callback="Nx.constant/3", data_type, backend,
        warm_up, repeat, stop_on_error, time_unit, opts) do
          do_benchmark(callback, data_type, backend, warm_up, repeat,
            stop_on_error, time_unit, opts, nil,
            fn _prepared ->
              time_func(fn ->
                Nx.tensor(0, type: data_type, backend: backend)
              end)
            end,
            fn _prepared, user_tensor ->
              verify_result(
                callback,
                data_type,
                user_tensor,
                Nx.tensor(0, type: data_type, backend: Nx.BinaryBackend),
                stop_on_error
              )
            end
          )
      end

      defp do_benchmark(callback, data_type, backend,
        warm_up, repeat, stop_on_error, time_unit, opts,
        prepared, benchmark_func, verify_func, note \\ nil) do
        results =
          Enum.map(1..repeat+warm_up, fn _ ->
            previous = Nx.default_backend(backend)
            user_tensor = benchmark_func.(prepared)
            Nx.default_backend(previous)
            user_tensor
          end)
          |> Enum.drop(warm_up)
          |> Enum.map(fn {time_elapsed, user_tensor} ->
            previous = Nx.default_backend(backend)
            close? = verify_func.(prepared, user_tensor)
            Nx.default_backend(previous)
            {time_elapsed*1.0, close?}
          end)

        time_elapsed = Nx.tensor(Enum.map(results, &elem(&1, 0)), backend: Nx.BinaryBackend)
        mean = Float.round(Nx.to_number(Nx.mean(time_elapsed)), 3)
        std = Float.round(Nx.to_number(Nx.standard_deviation(time_elapsed)), 3)
        max_run = Float.round(Nx.to_number(Nx.take(time_elapsed, Nx.argmax(time_elapsed))), 3)
        min_run = Float.round(Nx.to_number(Nx.take(time_elapsed, Nx.argmin(time_elapsed))), 3)
        statictics = [mean, std, max_run, min_run]
        [mean, std, max_run, min_run] =
          if time_unit == :auto do
            Enum.map(statictics, &use_proper_time_unit(&1))
          else
            Enum.map(statictics, &to_time_unit(&1, time_unit))
          end
        succeeded = Enum.count(Enum.reject(Enum.map(results, &elem(&1, 1)), fn b -> b == false end))
        %{
          callback: callback,
          data_type: data_type,
          mean: mean,
          std: std,
          max: max_run,
          min: min_run,
          succeeded: succeeded,
          note: note
        }
      end

      defp verify_result(callback, data_type, user_tensor, expected_tensor, stop_on_error) do
        Nx.default_backend(Nx.BinaryBackend)
        user_tensor = Nx.backend_transfer(user_tensor, Nx.BinaryBackend)
        close? = Nx.to_number(Nx.all_close(user_tensor, expected_tensor)) == 1
        if close? do
          close?
        else
          error_msg =
            "Failed: #{inspect(callback)}, data_type: #{inspect(data_type)}\n" <>
            "expecting: #{inspect(expected_tensor)}\n" <>
            "got: #{inspect(user_tensor)}"
          if stop_on_error do
            Mix.raise(error_msg)
          else
            Logger.error(error_msg)
            close?
          end
        end
      end

      defp time_func(func) do
        st = :erlang.monotonic_time(:nanosecond)
        ret = func.()
        et = :erlang.monotonic_time(:nanosecond)
        {et - st, ret}
      end

      defp parse_args(args) do
        {opts, cmd_and_args, errors} =
          args
          |> OptionParser.parse(
            strict: [
              backend: :string,
              warm_up: :integer,
              cooldown: :integer,
              repeat: :integer,
              time_unit: :string,
              stop_on_error: :boolean,
              data_types: :string,
              help: :boolean,
            ]
          )

        default_values = [
          cooldown: 0,
          warm_up: 0,
          repeat: 5,
          time_unit: "auto",
          stop_on_error: false,
          data_types: "f32,f64",
          help: false
        ]

        opts =
          Keyword.merge(opts, default_values, fn _k, user, default ->
            if user == nil do
              default
            else
              user
            end
          end)

        {opts, cmd_and_args, errors}
      end
    end
  end
end
