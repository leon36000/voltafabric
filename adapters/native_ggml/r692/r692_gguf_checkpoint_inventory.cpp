#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace {

void json_string(std::ostream & out, std::string_view value) {
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
}

std::optional<std::string> string_value(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return std::nullopt;
    const char * value = gguf_get_val_str(ctx, id);
    if (value == nullptr) return std::nullopt;
    return std::string(value);
}

std::optional<uint64_t> unsigned_value(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return std::nullopt;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT8: {
            const auto value = gguf_get_val_i8(ctx, id);
            if (value >= 0) return static_cast<uint64_t>(value);
            break;
        }
        case GGUF_TYPE_INT16: {
            const auto value = gguf_get_val_i16(ctx, id);
            if (value >= 0) return static_cast<uint64_t>(value);
            break;
        }
        case GGUF_TYPE_INT32: {
            const auto value = gguf_get_val_i32(ctx, id);
            if (value >= 0) return static_cast<uint64_t>(value);
            break;
        }
        case GGUF_TYPE_INT64: {
            const auto value = gguf_get_val_i64(ctx, id);
            if (value >= 0) return static_cast<uint64_t>(value);
            break;
        }
        default: break;
    }
    return std::nullopt;
}

void optional_string(std::ostream & out, const std::optional<std::string> & value) {
    if (value) json_string(out, *value);
    else out << "null";
}

void optional_u64(std::ostream & out, const std::optional<uint64_t> & value) {
    if (value) out << *value;
    else out << "null";
}

int inventory(const std::filesystem::path & path) {
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "GGUF inventory input is not a regular file\n";
        return 2;
    }

    ggml_context * tensor_context = nullptr;
    gguf_init_params params{true, &tensor_context};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> model(
        gguf_init_from_file(path.c_str(), params), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(tensor_context, ggml_free);
    if (!model) {
        std::cerr << "llama.cpp GGUF metadata parser rejected the input\n";
        return 3;
    }

    const auto architecture = string_value(model.get(), "general.architecture");
    const std::string prefix = architecture ? *architecture + "." : "";
    const auto file_bytes = std::filesystem::file_size(path);
    const int64_t tensor_count = gguf_get_n_tensors(model.get());
    if (tensor_count < 0) {
        std::cerr << "GGUF parser returned an invalid tensor count\n";
        return 3;
    }

    std::map<std::string, uint64_t> type_counts;
    for (int64_t i = 0; i < tensor_count; ++i) {
        const char * type_name = ggml_type_name(gguf_get_tensor_type(model.get(), i));
        if (type_name == nullptr) {
            std::cerr << "GGUF contains an unrecognized tensor type\n";
            return 3;
        }
        ++type_counts[type_name];
    }

    const auto license = string_value(model.get(), "general.license");
    const auto tokenizer_model = string_value(model.get(), "tokenizer.ggml.model");
    const auto tokenizer_pre = string_value(model.get(), "tokenizer.ggml.pre");
    const int64_t token_id = gguf_find_key(model.get(), "tokenizer.ggml.tokens");
    std::optional<uint64_t> tokenizer_tokens;
    if (token_id >= 0 && gguf_get_kv_type(model.get(), token_id) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(model.get(), token_id) == GGUF_TYPE_STRING) {
        tokenizer_tokens = static_cast<uint64_t>(gguf_get_arr_n(model.get(), token_id));
    }

    std::cout << "{\n"
              << "  \"format\": \"GGUF\",\n"
              << "  \"status\": \"METADATA_ONLY_NO_WEIGHTS_LOADED\",\n"
              << "  \"gguf_version\": " << gguf_get_version(model.get()) << ",\n"
              << "  \"file_bytes\": " << file_bytes << ",\n"
              << "  \"tensor_count\": " << tensor_count << ",\n"
              << "  \"tensor_type_counts\": {";
    bool first = true;
    for (const auto & [name, count] : type_counts) {
        if (!first) std::cout << ',';
        std::cout << "\n    ";
        json_string(std::cout, name);
        std::cout << ": " << count;
        first = false;
    }
    if (!type_counts.empty()) std::cout << '\n' << "  ";
    std::cout << "},\n"
              << "  \"metadata\": {\n"
              << "    \"general.name\": ";
    optional_string(std::cout, string_value(model.get(), "general.name"));
    std::cout << ",\n    \"general.architecture\": ";
    optional_string(std::cout, architecture);
    std::cout << ",\n    \"general.file_type\": ";
    optional_u64(std::cout, unsigned_value(model.get(), "general.file_type"));
    std::cout << ",\n    \"general.license\": ";
    optional_string(std::cout, license);
    std::cout << ",\n    \"license_verification_required\": true,\n"
              << "    \"tokenizer.ggml.model\": ";
    optional_string(std::cout, tokenizer_model);
    std::cout << ",\n    \"tokenizer.ggml.pre\": ";
    optional_string(std::cout, tokenizer_pre);
    std::cout << ",\n    \"tokenizer.ggml.tokens_count\": ";
    optional_u64(std::cout, tokenizer_tokens);
    std::cout << "\n  },\n"
              << "  \"architecture\": {\n"
              << "    \"embedding_length\": ";
    optional_u64(std::cout, unsigned_value(model.get(), prefix + "embedding_length"));
    std::cout << ",\n    \"block_count\": ";
    optional_u64(std::cout, unsigned_value(model.get(), prefix + "block_count"));
    std::cout << ",\n    \"expert_count\": ";
    optional_u64(std::cout, unsigned_value(model.get(), prefix + "expert_count"));
    std::cout << ",\n    \"expert_used_count\": ";
    optional_u64(std::cout, unsigned_value(model.get(), prefix + "expert_used_count"));
    std::cout << ",\n    \"expert_feed_forward_length\": ";
    optional_u64(std::cout, unsigned_value(model.get(), prefix + "expert_feed_forward_length"));
    std::cout << "\n  },\n"
              << "  \"compatibility_assessment\": \"NOT_ATTEMPTED\",\n"
              << "  \"license_note\": \"GGUF metadata is not independent license evidence; verify the model card and source terms.\"\n"
              << "}\n";
    return std::cout.good() ? 0 : 4;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: r692_gguf_checkpoint_inventory MODEL.gguf\n";
        return 2;
    }
    try {
        return inventory(argv[1]);
    } catch (const std::exception & error) {
        std::cerr << "GGUF inventory failed: " << error.what() << '\n';
        return 4;
    }
}
