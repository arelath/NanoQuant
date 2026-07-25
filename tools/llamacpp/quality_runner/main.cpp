#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr char INPUT_MAGIC[8] = {'N', 'Q', 'Q', 'L', '0', '0', '0', '1'};
constexpr char OUTPUT_MAGIC[8] = {'N', 'Q', 'Q', 'O', '0', '0', '0', '1'};

struct Sequence {
    std::vector<llama_token> tokens;
    std::uint32_t score_start = 0;
};

struct Score {
    double negative_log_likelihood = 0.0;
    std::uint32_t token_count = 0;
};

struct Options {
    std::string model;
    std::string input;
    std::string output;
    int gpu_layers = -1;
    int parallel = 4;
    int threads = 0;
    int batch_threads = 0;
};

template <typename T>
T read_value(std::istream & stream) {
    T value{};
    stream.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!stream) {
        throw std::runtime_error("quality input ended unexpectedly");
    }
    return value;
}

template <typename T>
void write_value(std::ostream & stream, const T & value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
    if (!stream) {
        throw std::runtime_error("failed to write quality output");
    }
}

int parse_integer(const char * value, const char * name) {
    std::size_t consumed = 0;
    const std::string text(value);
    const long parsed = std::stol(text, &consumed);
    if (consumed != text.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char ** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        auto next = [&]() -> const char * {
            if (++index >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[index];
        };
        if (argument == "--model") {
            result.model = next();
        } else if (argument == "--input") {
            result.input = next();
        } else if (argument == "--output") {
            result.output = next();
        } else if (argument == "--gpu-layers") {
            result.gpu_layers = parse_integer(next(), "gpu layer count");
        } else if (argument == "--parallel") {
            result.parallel = parse_integer(next(), "parallel sequence count");
        } else if (argument == "--threads") {
            result.threads = parse_integer(next(), "thread count");
        } else if (argument == "--batch-threads") {
            result.batch_threads = parse_integer(next(), "batch thread count");
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: nanoquant-llamacpp-quality --model MODEL.gguf --input INPUT.bin "
                   "--output OUTPUT.bin [--gpu-layers -1] [--parallel 4]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (result.model.empty() || result.input.empty() || result.output.empty()) {
        throw std::runtime_error("--model, --input, and --output are required");
    }
    if (result.parallel <= 0 || result.threads < 0 || result.batch_threads < 0) {
        throw std::runtime_error("parallel and thread settings are invalid");
    }
    return result;
}

std::vector<Sequence> read_sequences(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open quality input: " + path);
    }
    char magic[8]{};
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, INPUT_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error("quality input has an unsupported header");
    }
    const auto count = read_value<std::uint32_t>(stream);
    if (count == 0) {
        throw std::runtime_error("quality input contains no sequences");
    }
    std::vector<Sequence> result;
    result.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto token_count = read_value<std::uint32_t>(stream);
        const auto score_start = read_value<std::uint32_t>(stream);
        if (token_count < 2 || score_start == 0 || score_start >= token_count) {
            throw std::runtime_error("quality input contains an invalid sequence");
        }
        Sequence sequence;
        sequence.tokens.resize(token_count);
        sequence.score_start = score_start;
        stream.read(
            reinterpret_cast<char *>(sequence.tokens.data()),
            static_cast<std::streamsize>(token_count * sizeof(llama_token)));
        if (!stream) {
            throw std::runtime_error("quality input token payload ended unexpectedly");
        }
        result.push_back(std::move(sequence));
    }
    if (stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("quality input contains trailing bytes");
    }
    return result;
}

double target_negative_log_likelihood(
    const float * logits,
    std::int32_t vocabulary_size,
    llama_token target) {
    if (logits == nullptr || target < 0 || target >= vocabulary_size) {
        throw std::runtime_error("llama.cpp returned invalid logits or a target exceeds its vocabulary");
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::int32_t token = 0; token < vocabulary_size; ++token) {
        maximum = std::max(maximum, logits[token]);
    }
    double exponential_sum = 0.0;
    for (std::int32_t token = 0; token < vocabulary_size; ++token) {
        exponential_sum += std::exp(static_cast<double>(logits[token] - maximum));
    }
    const double log_normalizer = static_cast<double>(maximum) + std::log(exponential_sum);
    return log_normalizer - static_cast<double>(logits[target]);
}

std::vector<double> score_outputs(
    const float * all_logits,
    std::int32_t vocabulary_size,
    const std::vector<std::pair<std::size_t, llama_token>> & outputs,
    int requested_threads) {
    if (all_logits == nullptr) {
        throw std::runtime_error("llama.cpp returned no requested logits");
    }
    for (const auto & output : outputs) {
        if (output.second < 0 || output.second >= vocabulary_size) {
            throw std::runtime_error("a quality target exceeds the llama.cpp vocabulary");
        }
    }
    std::vector<double> result(outputs.size());
    std::atomic<std::size_t> next{0};
    const auto available = std::max(1u, std::thread::hardware_concurrency());
    const auto requested = requested_threads > 0
        ? static_cast<unsigned int>(requested_threads)
        : available;
    const std::size_t worker_count =
        std::min<std::size_t>(std::max(1u, requested), outputs.size());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&]() {
            while (true) {
                const std::size_t index = next.fetch_add(1);
                if (index >= outputs.size()) {
                    return;
                }
                result[index] = target_negative_log_likelihood(
                    all_logits + index * static_cast<std::size_t>(vocabulary_size),
                    vocabulary_size,
                    outputs[index].second);
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    return result;
}

void write_scores(const std::string & path, const std::vector<Score> & scores) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("failed to create quality output: " + path);
    }
    stream.write(OUTPUT_MAGIC, sizeof(OUTPUT_MAGIC));
    write_value(stream, static_cast<std::uint32_t>(scores.size()));
    for (const auto & score : scores) {
        write_value(stream, score.negative_log_likelihood);
        write_value(stream, score.token_count);
    }
    stream.flush();
    if (!stream) {
        throw std::runtime_error("failed to finalize quality output");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_batch batch{};
    bool batch_allocated = false;
    bool backend_initialized = false;
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<Sequence> sequences = read_sequences(options.input);
        const std::size_t parallel = std::min<std::size_t>(
            static_cast<std::size_t>(options.parallel), sequences.size());
        std::size_t maximum_input_length = 0;
        std::size_t maximum_scored = 0;
        for (const auto & sequence : sequences) {
            maximum_input_length =
                std::max(maximum_input_length, sequence.tokens.size() - 1);
            maximum_scored = std::max(
                maximum_scored,
                sequence.tokens.size() - static_cast<std::size_t>(sequence.score_start));
        }
        if (maximum_input_length >
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
            maximum_input_length * parallel >
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("quality input exceeds llama.cpp batch limits");
        }

        llama_backend_init();
        backend_initialized = true;
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.gpu_layers;
        model = llama_model_load_from_file(options.model.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("llama.cpp failed to load the GGUF model");
        }
        const llama_vocab * vocabulary = llama_model_get_vocab(model);
        const std::int32_t vocabulary_size = llama_vocab_n_tokens(vocabulary);
        if (vocabulary_size <= 0) {
            throw std::runtime_error("llama.cpp model has an invalid vocabulary");
        }

        const auto batch_capacity =
            static_cast<std::uint32_t>(maximum_input_length * parallel);
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = batch_capacity;
        context_params.n_batch = batch_capacity;
        context_params.n_ubatch = std::min<std::uint32_t>(batch_capacity, 512);
        context_params.n_seq_max = static_cast<std::uint32_t>(parallel);
        context_params.n_outputs_max = static_cast<std::uint32_t>(maximum_scored * parallel);
        context_params.no_perf = false;
        context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            throw std::runtime_error("llama.cpp failed to create the quality context");
        }
        if (options.threads > 0 || options.batch_threads > 0) {
            const int threads = options.threads > 0 ? options.threads : llama_n_threads(context);
            const int batch_threads =
                options.batch_threads > 0 ? options.batch_threads : llama_n_threads_batch(context);
            llama_set_n_threads(context, threads, batch_threads);
        }

        batch = llama_batch_init(
            static_cast<std::int32_t>(batch_capacity),
            0,
            static_cast<std::int32_t>(parallel));
        batch_allocated = true;
        std::vector<Score> scores(sequences.size());
        const auto wall_started = std::chrono::steady_clock::now();

        auto decode_range = [&](std::size_t begin, std::size_t end) {
            llama_memory_clear(llama_get_memory(context), true);
            batch.n_tokens = 0;
            std::vector<std::pair<std::size_t, llama_token>> outputs;
            for (std::size_t sequence_index = begin; sequence_index < end; ++sequence_index) {
                const auto & sequence = sequences[sequence_index];
                const llama_seq_id local_sequence =
                    static_cast<llama_seq_id>(sequence_index - begin);
                for (
                    std::size_t position = 0;
                    position + 1 < sequence.tokens.size();
                    ++position) {
                    const std::int32_t batch_index = batch.n_tokens++;
                    batch.token[batch_index] = sequence.tokens[position];
                    batch.pos[batch_index] = static_cast<llama_pos>(position);
                    batch.n_seq_id[batch_index] = 1;
                    batch.seq_id[batch_index][0] = local_sequence;
                    const bool scored = position + 1 >= sequence.score_start;
                    batch.logits[batch_index] = scored ? 1 : 0;
                    if (scored) {
                        outputs.emplace_back(sequence_index, sequence.tokens[position + 1]);
                    }
                }
            }
            const std::int32_t status = llama_decode(context, batch);
            if (status != 0) {
                throw std::runtime_error(
                    "llama_decode failed with status " + std::to_string(status));
            }
            const float * all_logits = llama_get_logits(context);
            const auto output_scores = score_outputs(
                all_logits, vocabulary_size, outputs, options.threads);
            std::vector<Score> range_scores(end - begin);
            for (std::size_t output_index = 0; output_index < outputs.size(); ++output_index) {
                const auto sequence_index = outputs[output_index].first;
                auto & score = range_scores[sequence_index - begin];
                score.negative_log_likelihood += output_scores[output_index];
                ++score.token_count;
            }
            return range_scores;
        };

        auto valid_score = [&](std::size_t sequence_index, const Score & score) {
            const auto expected =
                sequences[sequence_index].tokens.size() -
                sequences[sequence_index].score_start;
            return score.token_count == expected &&
                std::isfinite(score.negative_log_likelihood);
        };

        auto invalid_score_message = [&](std::size_t sequence_index, const Score & score) {
            const auto expected =
                sequences[sequence_index].tokens.size() -
                sequences[sequence_index].score_start;
            return
                "sequence " + std::to_string(sequence_index) +
                " expected " + std::to_string(expected) +
                " scored tokens but received " + std::to_string(score.token_count) +
                "; negative_log_likelihood=" +
                std::to_string(score.negative_log_likelihood);
        };

        for (std::size_t begin = 0; begin < sequences.size(); begin += parallel) {
            const std::size_t end = std::min(begin + parallel, sequences.size());
            auto range_scores = decode_range(begin, end);
            bool range_valid = true;
            for (std::size_t sequence_index = begin; sequence_index < end; ++sequence_index) {
                if (!valid_score(sequence_index, range_scores[sequence_index - begin])) {
                    range_valid = false;
                    break;
                }
            }
            if (!range_valid && end - begin > 1) {
                std::cerr
                    << "llama.cpp quality batch " << (begin / parallel + 1)
                    << " produced an invalid parallel score; retrying sequences "
                    << begin << "-" << (end - 1) << " individually"
                    << std::endl;
                for (std::size_t sequence_index = begin; sequence_index < end; ++sequence_index) {
                    auto single_score = decode_range(sequence_index, sequence_index + 1).front();
                    if (!valid_score(sequence_index, single_score)) {
                        throw std::runtime_error(
                            "quality score remained invalid after single-sequence retry: " +
                            invalid_score_message(sequence_index, single_score));
                    }
                    scores[sequence_index] = single_score;
                }
            } else {
                for (std::size_t sequence_index = begin; sequence_index < end; ++sequence_index) {
                    const auto & score = range_scores[sequence_index - begin];
                    if (!valid_score(sequence_index, score)) {
                        throw std::runtime_error(
                            "quality score is invalid: " +
                            invalid_score_message(sequence_index, score));
                    }
                    scores[sequence_index] = score;
                }
            }
            std::cerr << "llama.cpp quality batch " << (begin / parallel + 1) << "/"
                      << ((sequences.size() + parallel - 1) / parallel) << " completed"
                      << std::endl;
        }
        llama_synchronize(context);
        for (std::size_t index = 0; index < sequences.size(); ++index) {
            if (!valid_score(index, scores[index])) {
                throw std::runtime_error(
                    "quality score failed final validation: " +
                    invalid_score_message(index, scores[index]));
            }
        }
        write_scores(options.output, scores);
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_started);
        std::cerr << "llama.cpp quality completed " << sequences.size() << " sequences in "
                  << elapsed.count() << " seconds" << std::endl;

        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama.cpp quality failed: " << error.what() << std::endl;
        if (batch_allocated) {
            llama_batch_free(batch);
        }
        if (context != nullptr) {
            llama_free(context);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }
}
