#include "core/render/modules/world/shader_pack/shader_pack.hpp"

#include "core/render/renderer.hpp"
#include "core/util/parallel.hpp"
#include "core/vulkan/result.hpp"

#include "mz.h"
#include "mz_strm.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"
#include <tinyexpr.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::json;

namespace {

constexpr uint64_t FNV1A64_OFFSET_BASIS = 14695981039346656037ULL;
constexpr uint64_t FNV1A64_PRIME = 1099511628211ULL;

void updateFnv1a64(uint64_t &hash, const char *data, size_t size) {
    for (size_t i = 0; i < size; ++i) { hash = (hash ^ static_cast<unsigned char>(data[i])) * FNV1A64_PRIME; }
}

bool hashFileFnv1a64(const fs::path &path, uint64_t &hash, std::string &error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "failed to open shader pack for hashing: " + path.string();
        return false;
    }

    hash = FNV1A64_OFFSET_BASIS;
    std::array<char, 64 * 1024> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        updateFnv1a64(hash, buffer.data(), static_cast<size_t>(stream.gcount()));
    }
    if (!stream.eof()) {
        error = "failed to read shader pack for hashing: " + path.string();
        return false;
    }
    return true;
}

std::shared_ptr<std::mutex> shaderPackExtractionMutex(const fs::path &path) {
    static std::mutex registryMutex;
    static std::unordered_map<std::string, std::weak_ptr<std::mutex>> mutexes;

    std::error_code ec;
    fs::path normalizedPath = fs::weakly_canonical(path, ec);
    if (ec) { normalizedPath = path.lexically_normal(); }
    const std::string key = normalizedPath.string();

    std::lock_guard<std::mutex> registryLock(registryMutex);
    std::shared_ptr<std::mutex> extractionMutex = mutexes[key].lock();
    if (extractionMutex == nullptr) {
        extractionMutex = std::make_shared<std::mutex>();
        mutexes[key] = extractionMutex;
    }
    return extractionMutex;
}

}

std::string ShaderPack::trimString(std::string_view text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])) != 0) { start++; }

    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { end--; }

    return std::string(text.substr(start, end - start));
}

std::optional<std::string> ShaderPack::unwrapScalarConstructor(std::string_view expression,
                                                               std::string_view constructorName) {
    std::string_view trimmed = expression;
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front())) != 0) {
        trimmed.remove_prefix(1);
    }
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back())) != 0) {
        trimmed.remove_suffix(1);
    }

    if (trimmed.size() <= constructorName.size() + 2 || trimmed.substr(0, constructorName.size()) != constructorName) {
        return std::nullopt;
    }

    trimmed.remove_prefix(constructorName.size());
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front())) != 0) {
        trimmed.remove_prefix(1);
    }
    if (trimmed.empty() || trimmed.front() != '(' || trimmed.back() != ')') { return std::nullopt; }

    int depth = 0;
    for (size_t i = 0; i < trimmed.size(); ++i) {
        if (trimmed[i] == '(') {
            depth++;
        } else if (trimmed[i] == ')') {
            depth--;
            if (depth == 0 && i + 1 != trimmed.size()) { return std::nullopt; }
            if (depth < 0) { return std::nullopt; }
        }
    }
    if (depth != 0) { return std::nullopt; }

    return trimString(trimmed.substr(1, trimmed.size() - 2));
}

std::string ShaderPack::parseAttributeValueForDefineExpression(const ShaderPackLoader::AttributeConfig &attribute,
                                                               const std::string &currentValue) {
    if (attribute.type == "bool") {
        return currentValue == "render_pipeline.true" || currentValue == "true" || currentValue == "1" ? "1" : "0";
    }
    if (attribute.type.rfind("enum:", 0) == 0) {
        std::string options = attribute.type.substr(5);
        std::stringstream stream(options);
        std::string option;
        int index = 0;
        while (std::getline(stream, option, '-')) {
            if (option == currentValue) { return std::to_string(index); }
            index++;
        }
    }
    return currentValue;
}

std::string ShaderPack::substitutePlaceholderInDefineExpression(const std::string &expression,
                                                                const std::string &placeholderValue) {
    auto isSymbolChar = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_'; };
    std::string resolved;
    resolved.reserve(expression.size() + placeholderValue.size());
    for (size_t i = 0; i < expression.size(); i++) {
        if (expression[i] == 'X') {
            bool leftIsSymbol = i > 0 && isSymbolChar(expression[i - 1]);
            bool rightIsSymbol = i + 1 < expression.size() && isSymbolChar(expression[i + 1]);
            if (!leftIsSymbol && !rightIsSymbol) {
                resolved += placeholderValue;
                continue;
            }
        }
        resolved += expression[i];
    }
    return resolved;
}

std::optional<double> ShaderPack::tryEvaluateScalarDefineExpression(const ShaderPackLoader::AttributeConfig &attribute,
                                                                    const std::string &currentValue,
                                                                    const std::string &expression) {
    std::string resolved = substitutePlaceholderInDefineExpression(
        expression, parseAttributeValueForDefineExpression(attribute, currentValue));

    bool stripped = true;
    while (stripped) {
        stripped = false;
        for (std::string_view constructorName : {"uint", "int", "float", "double"}) {
            auto inner = unwrapScalarConstructor(resolved, constructorName);
            if (!inner.has_value()) { continue; }
            resolved = std::move(*inner);
            stripped = true;
            break;
        }
    }

    try {
        return ExpressionEvaluator::evaluate(resolved, {});
    } catch (...) { return std::nullopt; }
}

bool ShaderPack::parseExecutionBoolValue(const std::string &rawValue) {
    return rawValue == "true" || rawValue == "1" || rawValue == "render_pipeline.true";
}

bool ShaderPack::isExecutionIntType(const std::string &type) {
    return type == "int" || type.rfind("int_range:", 0) == 0;
}

bool ShaderPack::isExecutionFloatType(const std::string &type) {
    return type == "float" || type.rfind("float_range:", 0) == 0;
}

bool ShaderPack::isExecutionNumericExpressionType(const std::string &type) {
    return type == "bool" || isExecutionIntType(type) || isExecutionFloatType(type);
}

std::string ShaderPack::formatExecutionFloat(double numericValue) {
    std::ostringstream stream;
    stream << numericValue;
    return stream.str();
}

double ShaderPack::parseExecutionNumber(const std::string &name, const std::string &type, const std::string &rawValue) {
    if (type == "bool") { return parseExecutionBoolValue(rawValue) ? 1.0 : 0.0; }
    if (type.rfind("int_range:", 0) == 0) {
        std::string range = type.substr(10);
        size_t separator = range.find('-', 1);
        if (separator != std::string::npos) {
            int minValue = std::stoi(range.substr(0, separator));
            int maxValue = std::stoi(range.substr(separator + 1));
            if (minValue > maxValue) { std::swap(minValue, maxValue); }
            int value = std::stoi(rawValue);
            return static_cast<double>(std::clamp(value, minValue, maxValue));
        }
    }
    if (type.rfind("float_range:", 0) == 0) {
        std::string range = type.substr(12);
        size_t separator = range.find('-', 1);
        if (separator != std::string::npos) {
            double minValue = std::stod(range.substr(0, separator));
            double maxValue = std::stod(range.substr(separator + 1));
            if (minValue > maxValue) { std::swap(minValue, maxValue); }
            double value = std::stod(rawValue);
            return std::clamp(value, minValue, maxValue);
        }
    }
    if (type.rfind("enum:", 0) == 0) {
        std::string options = type.substr(5);
        std::stringstream stream(options);
        std::string option;
        int index = 0;
        while (std::getline(stream, option, '-')) {
            if (option == rawValue) { return static_cast<double>(index); }
            index++;
        }
    }

    size_t parsed = 0;
    double numericValue = std::stod(rawValue, &parsed);
    if (parsed != rawValue.size()) { throw std::runtime_error("invalid numeric execution variable: " + name); }
    return numericValue;
}

std::string ShaderPack::formatExecutionValue(const std::string &type, double numericValue) {
    if (type == "bool") { return numericValue != 0.0 ? "true" : "false"; }
    if (isExecutionIntType(type)) { return std::to_string(static_cast<int64_t>(numericValue)); }
    if (isExecutionFloatType(type)) { return formatExecutionFloat(numericValue); }
    return formatExecutionFloat(numericValue);
}

std::string ShaderPack::defaultExecutionValueForType(const std::string &type) {
    if (type == "bool") { return "false"; }
    if (isExecutionIntType(type) || type.rfind("enum:", 0) == 0) { return "0"; }
    return "0.0";
}

double ExpressionEvaluator::trueValue() {
    return 1.0;
}

double ExpressionEvaluator::falseValue() {
    return 0.0;
}

double ExpressionEvaluator::ifValue(double condition, double ifTrueValue, double ifFalseValue) {
    return condition != 0.0 ? ifTrueValue : ifFalseValue;
}

struct ExpressionEvaluator::CompiledExpression::CompiledExpressionState {
    std::vector<std::string> variableNames;
    std::vector<double> defaultVariableValues;
    mutable std::vector<double> variableValues;
    std::unordered_map<std::string, size_t> variableIndices;
    te_expr *compiledExpression = nullptr;
    mutable std::mutex evaluationMutex;

    ~CompiledExpressionState() {
        if (compiledExpression != nullptr) { te_free(compiledExpression); }
    }
};

ExpressionEvaluator::CompiledExpression::CompiledExpression() = default;

ExpressionEvaluator::CompiledExpression::CompiledExpression(const std::string &expression,
                                                            const std::vector<std::string> &variableNames,
                                                            const std::vector<Variable> &staticVariables)
    : state_(std::make_unique<CompiledExpressionState>()) {
    auto appendVariable = [&](const std::string &name, double defaultValue) -> size_t {
        if (!isValidIdentifier(name)) { throw std::runtime_error("invalid execution variable name: " + name); }
        if (isReservedIdentifier(name)) { throw std::runtime_error("reserved execution variable name: " + name); }
        auto iter = state_->variableIndices.find(name);
        if (iter != state_->variableIndices.end()) {
            state_->defaultVariableValues[iter->second] = defaultValue;
            state_->variableValues[iter->second] = defaultValue;
            return iter->second;
        }
        const size_t index = state_->variableNames.size();
        state_->variableIndices.emplace(name, index);
        state_->variableNames.push_back(name);
        state_->defaultVariableValues.push_back(defaultValue);
        state_->variableValues.push_back(defaultValue);
        return index;
    };

    state_->variableIndices.emplace("true", state_->variableNames.size());
    state_->variableNames.push_back("true");
    state_->defaultVariableValues.push_back(trueValue());
    state_->variableValues.push_back(trueValue());

    state_->variableIndices.emplace("false", state_->variableNames.size());
    state_->variableNames.push_back("false");
    state_->defaultVariableValues.push_back(falseValue());
    state_->variableValues.push_back(falseValue());

    for (const auto &name : variableNames) { appendVariable(name, 0.0); }
    for (const auto &variable : staticVariables) { appendVariable(variable.name, variable.value); }

    std::vector<te_variable> teVariables;
    teVariables.reserve(state_->variableNames.size() + 1);
    for (size_t i = 0; i < state_->variableNames.size(); ++i) {
        teVariables.push_back({
            .name = state_->variableNames[i].c_str(),
            .address = &state_->variableValues[i],
            .type = TE_VARIABLE,
            .context = nullptr,
        });
    }
    teVariables.push_back({
        .name = "if",
        .address = reinterpret_cast<const void *>(&ExpressionEvaluator::ifValue),
        .type = TE_FUNCTION3 | TE_FLAG_PURE,
        .context = nullptr,
    });

    int error = 0;
    state_->compiledExpression =
        te_compile(expression.c_str(), teVariables.data(), static_cast<int>(teVariables.size()), &error);
    if (state_->compiledExpression == nullptr) {
        throw std::runtime_error("invalid execution expression: " + expression + " at " + std::to_string(error));
    }
}

ExpressionEvaluator::CompiledExpression::~CompiledExpression() = default;

ExpressionEvaluator::CompiledExpression::CompiledExpression(CompiledExpression &&) noexcept = default;

ExpressionEvaluator::CompiledExpression &
ExpressionEvaluator::CompiledExpression::operator=(CompiledExpression &&) noexcept = default;

double ExpressionEvaluator::CompiledExpression::evaluate(const std::vector<Variable> &variables) const {
    if (state_ == nullptr || state_->compiledExpression == nullptr) {
        throw std::runtime_error("empty compiled expression");
    }

    std::lock_guard<std::mutex> lock(state_->evaluationMutex);
    std::copy(state_->defaultVariableValues.begin(), state_->defaultVariableValues.end(),
              state_->variableValues.begin());
    for (const auto &variable : variables) {
        auto iter = state_->variableIndices.find(variable.name);
        if (iter != state_->variableIndices.end()) { state_->variableValues[iter->second] = variable.value; }
    }
    return te_eval(state_->compiledExpression);
}

bool ExpressionEvaluator::isValidIdentifier(std::string_view name) {
    if (name.empty()) { return false; }
    if (std::isalpha(static_cast<unsigned char>(name.front())) == 0) { return false; }
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '_') { return false; }
    }
    return true;
}

bool ExpressionEvaluator::isReservedIdentifier(std::string_view name) {
    return name == "if" || name == "true" || name == "false";
}

double ExpressionEvaluator::evaluate(const std::string &expression, const std::vector<Variable> &variables) {
    std::vector<std::string> names;
    std::vector<double> values;
    std::vector<te_variable> teVariables;

    names.reserve(variables.size() + 2);
    values.reserve(variables.size() + 2);
    teVariables.reserve(variables.size() + 3);

    auto appendVariable = [&](const std::string &name, double value) {
        names.push_back(name);
        values.push_back(value);
        teVariables.push_back({
            .name = names.back().c_str(),
            .address = &values.back(),
            .type = TE_VARIABLE,
            .context = nullptr,
        });
    };

    appendVariable("true", trueValue());
    appendVariable("false", falseValue());

    for (const auto &variable : variables) {
        if (!isValidIdentifier(variable.name)) {
            throw std::runtime_error("invalid execution variable name: " + variable.name);
        }
        if (isReservedIdentifier(variable.name)) {
            throw std::runtime_error("reserved execution variable name: " + variable.name);
        }
        appendVariable(variable.name, variable.value);
    }

    teVariables.push_back({
        .name = "if",
        .address = reinterpret_cast<const void *>(&ExpressionEvaluator::ifValue),
        .type = TE_FUNCTION3 | TE_FLAG_PURE,
        .context = nullptr,
    });

    int error = 0;
    te_expr *compiled =
        te_compile(expression.c_str(), teVariables.data(), static_cast<int>(teVariables.size()), &error);
    if (compiled == nullptr) {
        throw std::runtime_error("invalid execution expression: " + expression + " at " + std::to_string(error));
    }

    double value = te_eval(compiled);
    te_free(compiled);
    return value;
}

std::string ShaderPackLoader::toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string ShaderPackLoader::trim(std::string value) {
    auto isWhitespace = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(), [&](unsigned char ch) { return !isWhitespace(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](unsigned char ch) { return !isWhitespace(ch); }).base(),
                value.end());
    return value;
}

uint64_t ShaderPackLoader::calculateFnv1a64Hash(std::string_view text) {
    uint64_t hash = FNV1A64_OFFSET_BASIS;
    updateFnv1a64(hash, text.data(), text.size());
    return hash;
}

std::string ShaderPackLoader::stableFolderName(const fs::path &path) {
    std::error_code ec;
    fs::path normalizedPath = fs::weakly_canonical(path, ec);
    if (ec) { normalizedPath = path.lexically_normal(); }
    std::string key = normalizedPath.string();
    size_t hashValue = static_cast<size_t>(calculateFnv1a64Hash(key));
    std::ostringstream stream;
    stream << normalizedPath.stem().string() << "_" << std::hex << hashValue;
    return stream.str();
}

bool ShaderPackLoader::extractZip(const fs::path &zipPath, const fs::path &destinationPath, std::string &error) {
    std::error_code ec;
    fs::remove_all(destinationPath, ec);
    fs::create_directories(destinationPath, ec);
    if (ec) {
        error = "failed to prepare temp dir: " + destinationPath.string();
        return false;
    }

    std::shared_ptr<void> zipReader(mz_zip_reader_create(), [](void *handle) {
        if (handle == nullptr) { return; }
        void *zipReaderHandle = handle;
        mz_zip_reader_delete(&zipReaderHandle);
    });
    if (zipReader == nullptr) {
        error = "failed to create zip reader";
        return false;
    }

    int32_t openResult = mz_zip_reader_open_file(zipReader.get(), zipPath.string().c_str());
    if (openResult != MZ_OK) {
        error = "failed to open zip file (minizip error " + std::to_string(openResult) + "): " + zipPath.string();
        return false;
    }

    int32_t extractResult = mz_zip_reader_save_all(zipReader.get(), destinationPath.string().c_str());
    int32_t closeResult = mz_zip_reader_close(zipReader.get());

    if (extractResult != MZ_OK) {
        error = "failed to extract zip file (minizip error " + std::to_string(extractResult) + "): " + zipPath.string();
        return false;
    }
    if (closeResult != MZ_OK) {
        error = "failed to close zip file (minizip error " + std::to_string(closeResult) + "): " + zipPath.string();
        return false;
    }

    return true;
}

fs::path ShaderPackLoader::resolveShaderPackRoot(const fs::path &path, std::string &error) {
    if (path.empty()) {
        error = "shader pack path is empty";
        return {};
    }

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        error = "shader pack path does not exist: " + path.string();
        return {};
    }

    if (fs::is_directory(path, ec)) { return fs::weakly_canonical(path, ec); }

    if (toLower(path.extension().string()) != ".zip") {
        error = "shader pack must be a directory or zip: " + path.string();
        return {};
    }

    std::shared_ptr<std::mutex> extractionMutex = shaderPackExtractionMutex(path);
    std::lock_guard<std::mutex> extractionLock(*extractionMutex);

    uint64_t archiveHash = 0;
    if (!hashFileFnv1a64(path, archiveHash, error)) { return {}; }

    std::ostringstream archiveHashStream;
    archiveHashStream << std::hex << archiveHash;
    const fs::path extractRoot = Renderer::folderPath / "temp/shaders/world/ray_tracing";
    const fs::path destination = extractRoot / (stableFolderName(path) + "_" + archiveHashStream.str());
    const fs::path completionMarker = destination / ".mcvr-extract-complete";
    if (fs::is_directory(destination, ec) && fs::is_regular_file(completionMarker, ec)) { return destination; }

    if (!extractZip(path, destination, error)) { return {}; }

    std::ofstream marker(completionMarker, std::ios::binary | std::ios::trunc);
    if (!marker.is_open()) {
        error = "failed to finalize extracted shader pack: " + destination.string();
        return {};
    }
    marker << archiveHashStream.str();
    marker.close();
    if (!marker) {
        error = "failed to finalize extracted shader pack: " + destination.string();
        return {};
    }

    return destination;
}

std::optional<std::reference_wrapper<const json>>
ShaderPackLoader::findKey(const json &value, std::initializer_list<std::string_view> keys) {
    for (std::string_view key : keys) {
        auto iter = value.find(std::string(key));
        if (iter != value.end()) { return std::cref(*iter); }
    }
    return std::nullopt;
}

const json &ShaderPackLoader::requireObject(const json &value, std::string_view key, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end() || !iter->is_object()) {
        throw std::runtime_error(context + "." + keyString + " must be an object");
    }
    return *iter;
}

const json &ShaderPackLoader::requireArray(const json &value, std::string_view key, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end() || !iter->is_array()) {
        throw std::runtime_error(context + "." + keyString + " must be an array");
    }
    return *iter;
}

std::string ShaderPackLoader::requireString(const json &value, std::string_view key, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end() || !iter->is_string()) {
        throw std::runtime_error(context + "." + keyString + " must be a string");
    }
    return iter->get<std::string>();
}

std::optional<std::string>
ShaderPackLoader::optionalString(const json &value, std::string_view key, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end()) { return std::nullopt; }
    if (!iter->is_string()) { throw std::runtime_error(context + "." + keyString + " must be a string"); }
    return iter->get<std::string>();
}

bool ShaderPackLoader::optionalBool(const json &value,
                                    std::string_view key,
                                    bool fallback,
                                    const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end()) { return fallback; }
    if (!iter->is_boolean()) { throw std::runtime_error(context + "." + keyString + " must be a boolean"); }
    return iter->get<bool>();
}

VkFormat ShaderPackLoader::parseFormat(const std::string &value) {
    const std::string lower = toLower(value);
    if (lower == "r8_unorm") return VK_FORMAT_R8_UNORM;
    if (lower == "r8_uint") return VK_FORMAT_R8_UINT;
    if (lower == "r8g8_unorm") return VK_FORMAT_R8G8_UNORM;
    if (lower == "r8g8b8a8_unorm") return VK_FORMAT_R8G8B8A8_UNORM;
    if (lower == "r16_sfloat") return VK_FORMAT_R16_SFLOAT;
    if (lower == "r16g16_sfloat") return VK_FORMAT_R16G16_SFLOAT;
    if (lower == "r16g16b16a16_uint") return VK_FORMAT_R16G16B16A16_UINT;
    if (lower == "r16g16b16a16_sfloat") return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (lower == "r32_uint") return VK_FORMAT_R32_UINT;
    if (lower == "r32_sfloat") return VK_FORMAT_R32_SFLOAT;
    if (lower == "r32g32_sfloat") return VK_FORMAT_R32G32_SFLOAT;
    if (lower == "r32g32_uint") return VK_FORMAT_R32G32_UINT;
    if (lower == "r32g32b32a32_sfloat") return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (lower == "r32g32b32a32_uint") return VK_FORMAT_R32G32B32A32_UINT;
    throw std::runtime_error("unsupported format: " + value);
}

VkFilter ShaderPackLoader::parseFilter(const std::string &value) {
    const std::string lower = toLower(value);
    if (lower == "nearest") return VK_FILTER_NEAREST;
    if (lower == "linear") return VK_FILTER_LINEAR;
    throw std::runtime_error("unsupported filter: " + value);
}

VkSamplerAddressMode ShaderPackLoader::parseAddressMode(const std::string &value) {
    const std::string lower = toLower(value);
    if (lower == "repeat") return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (lower == "clamp_to_edge") return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (lower == "mirrored_repeat") return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    throw std::runtime_error("unsupported address mode: " + value);
}

ShaderPackLoader::TextureDimension ShaderPackLoader::parseTextureDimension(const std::string &value) {
    const std::string lower = toLower(value);
    if (lower == ShaderPackLoader::VALUE_2D) return ShaderPackLoader::TextureDimension::Texture2D;
    if (lower == ShaderPackLoader::VALUE_2D_ARRAY) return ShaderPackLoader::TextureDimension::Texture2DArray;
    if (lower == ShaderPackLoader::VALUE_3D) return ShaderPackLoader::TextureDimension::Texture3D;
    if (lower == ShaderPackLoader::VALUE_CUBE) return ShaderPackLoader::TextureDimension::Cube;
    throw std::runtime_error("unsupported texture dimension: " + value);
}

ShaderPackLoader::AccelerationStructureBuildMode
ShaderPackLoader::parseAccelerationStructureBuildMode(const std::string &value, const std::string &context) {
    const std::string lower = toLower(value);
    if (lower == ShaderPackLoader::VALUE_SYNC) { return ShaderPackLoader::AccelerationStructureBuildMode::Sync; }
    if (lower == ShaderPackLoader::VALUE_DEFERRED) {
        return ShaderPackLoader::AccelerationStructureBuildMode::Deferred;
    }
    throw std::runtime_error(context + " must be \"sync\" or \"deferred\"");
}

ShaderPackLoader::NrdMode ShaderPackLoader::parseNrdMode(const std::string &value, const std::string &context) {
    const std::string lower = toLower(value);
    if (lower == ShaderPackLoader::VALUE_NRD_PROBABILISTIC) { return ShaderPackLoader::NrdMode::Probabilistic; }
    if (lower == ShaderPackLoader::VALUE_NRD_CHECKERBOARD) { return ShaderPackLoader::NrdMode::Checkerboard; }
    if (lower == ShaderPackLoader::VALUE_NRD_HALF_RATE_CHECKERBOARD) {
        return ShaderPackLoader::NrdMode::HalfRateCheckerboard;
    }
    throw std::runtime_error(context + " must be \"probabilistic\", \"checkerboard\", or "
                                       "\"half_rate_checkerboard\"");
}

static bool shaderPackNameLooksRasterPrimary(std::string_view value) {
    std::string lower(value);
    lower = ShaderPackLoader::toLower(std::move(lower));
    return lower.find("raster_primary") != std::string::npos || lower.find("primary_surface") != std::string::npos;
}

static bool shaderPackPathLooksRasterPrimary(const fs::path &path) {
    return shaderPackNameLooksRasterPrimary(path.generic_string());
}

static bool shaderPackResourceListUsesRasterPrimary(const ShaderPackLoader::ResourceList &resources) {
    auto resourceUsesRasterPrimary = [](const std::string &name) { return shaderPackNameLooksRasterPrimary(name); };
    if (std::any_of(resources.images.begin(), resources.images.end(), resourceUsesRasterPrimary)) { return true; }
    if (std::any_of(resources.buffers.begin(), resources.buffers.end(), resourceUsesRasterPrimary)) { return true; }
    for (const auto &access : resources.imageAccesses) {
        if (shaderPackNameLooksRasterPrimary(access.name)) { return true; }
    }
    for (const auto &access : resources.bufferAccesses) {
        if (shaderPackNameLooksRasterPrimary(access.name)) { return true; }
    }
    return false;
}

ShaderPackLoader::AccelerationStructureBuildMode
ShaderPackLoader::inferAccelerationStructureBuildMode(const ShaderPackLoader::ShaderPack &shaderPack) {
    bool hasRasterPrimary = false;
    for (const auto &pass : shaderPack.passes) {
        switch (pass.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen:
                hasRasterPrimary = shaderPackNameLooksRasterPrimary(pass.fullScreen.name) ||
                                   shaderPackNameLooksRasterPrimary(pass.fullScreen.target) ||
                                   shaderPackPathLooksRasterPrimary(pass.fullScreen.fragmentShaderPath) ||
                                   shaderPackResourceListUsesRasterPrimary(pass.fullScreen.inputs) ||
                                   shaderPackResourceListUsesRasterPrimary(pass.fullScreen.outputs);
                break;
            case ShaderPackLoader::PassConfig::Type::Render:
                hasRasterPrimary = shaderPackNameLooksRasterPrimary(pass.render.name) ||
                                   shaderPackNameLooksRasterPrimary(pass.render.content) ||
                                   shaderPackResourceListUsesRasterPrimary(pass.render.inputs) ||
                                   shaderPackResourceListUsesRasterPrimary(pass.render.outputs);
                if (!hasRasterPrimary && pass.render.indirectDraw.has_value()) { hasRasterPrimary = true; }
                if (!hasRasterPrimary) {
                    for (const auto &[name, clearValue] : pass.render.clearOutputs) {
                        (void)clearValue;
                        if (shaderPackNameLooksRasterPrimary(name)) {
                            hasRasterPrimary = true;
                            break;
                        }
                    }
                }
                if (!hasRasterPrimary) {
                    for (const auto &[name, pipeline] : pass.render.pipelineConfigs) {
                        if (shaderPackNameLooksRasterPrimary(name) ||
                            shaderPackPathLooksRasterPrimary(pipeline.shader.vertexShaderPath) ||
                            shaderPackPathLooksRasterPrimary(pipeline.shader.fragmentShaderPath)) {
                            hasRasterPrimary = true;
                            break;
                        }
                    }
                }
                break;
            case ShaderPackLoader::PassConfig::Type::RayTracing:
                hasRasterPrimary = shaderPackNameLooksRasterPrimary(pass.rayTracing.name) ||
                                   shaderPackPathLooksRasterPrimary(pass.rayTracing.rayGenShaderPath) ||
                                   shaderPackResourceListUsesRasterPrimary(pass.rayTracing.inputs) ||
                                   shaderPackResourceListUsesRasterPrimary(pass.rayTracing.outputs);
                break;
            case ShaderPackLoader::PassConfig::Type::Compute:
                hasRasterPrimary = shaderPackNameLooksRasterPrimary(pass.compute.name) ||
                                   shaderPackPathLooksRasterPrimary(pass.compute.computeShaderPath) ||
                                   shaderPackResourceListUsesRasterPrimary(pass.compute.inputs) ||
                                   shaderPackResourceListUsesRasterPrimary(pass.compute.outputs);
                break;
        }
        if (hasRasterPrimary) { break; }
    }

    return hasRasterPrimary ? ShaderPackLoader::AccelerationStructureBuildMode::Deferred :
                              ShaderPackLoader::AccelerationStructureBuildMode::Sync;
}

std::unordered_map<std::string, std::string> ShaderPackLoader::parseTranslations(const fs::path &langPath) {
    std::unordered_map<std::string, std::string> translations;
    std::ifstream stream(langPath);
    if (!stream.is_open()) { return translations; }

    std::string line;
    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') { continue; }
        size_t separator = line.find('=');
        if (separator == std::string::npos) { continue; }
        std::string key = trim(line.substr(0, separator));
        std::string value = trim(line.substr(separator + 1));
        if (!key.empty()) { translations[key] = value; }
    }

    return translations;
}

std::unordered_map<std::string, std::string> ShaderPackLoader::loadLanguage(const fs::path &rootPath,
                                                                            const std::string &language) {
    std::unordered_map<std::string, std::string> translations;
    const fs::path langRoot = rootPath / ShaderPackLoader::KEY_LANG;
    if (!fs::exists(langRoot) || !fs::is_directory(langRoot)) { return translations; }

    auto normalizeLanguageCode = [&](std::string code) {
        code = trim(toLower(std::move(code)));
        std::replace(code.begin(), code.end(), '\\', '/');

        size_t separator = code.find_last_of('/');
        if (separator != std::string::npos) { code = code.substr(separator + 1); }

        auto stripSuffix = [&](std::string_view suffix) {
            if (code.size() < suffix.size()) { return false; }
            if (code.compare(code.size() - suffix.size(), suffix.size(), suffix) != 0) { return false; }
            code.resize(code.size() - suffix.size());
            return true;
        };
        while (stripSuffix(".json") || stripSuffix(".lang")) {}

        std::string normalized;
        normalized.reserve(code.size());
        bool previousUnderscore = false;
        for (char ch : code) {
            unsigned char uch = static_cast<unsigned char>(ch);
            if (std::isalnum(uch) != 0) {
                normalized.push_back(static_cast<char>(std::tolower(uch)));
                previousUnderscore = false;
                continue;
            }
            if (ch == '@' || ch == '#') { break; }
            if ((ch == '-' || ch == '_' || ch == '.' || std::isspace(uch) != 0) && !normalized.empty() &&
                !previousUnderscore) {
                normalized.push_back('_');
                previousUnderscore = true;
            }
        }

        while (!normalized.empty() && normalized.back() == '_') { normalized.pop_back(); }
        return normalized;
    };

    std::unordered_map<std::string, fs::path> availableTranslations;
    for (const auto &entry : fs::directory_iterator(langRoot)) {
        if (!entry.is_regular_file()) { continue; }
        if (toLower(entry.path().extension().string()) != ".lang") { continue; }

        std::string normalizedCode = normalizeLanguageCode(entry.path().stem().string());
        if (!normalizedCode.empty()) { availableTranslations.emplace(std::move(normalizedCode), entry.path()); }
    }

    auto mergeFrom = [&](const std::string &code) {
        std::string normalizedCode = normalizeLanguageCode(code);
        if (normalizedCode.empty()) { return; }

        auto pathIter = availableTranslations.find(normalizedCode);
        if (pathIter == availableTranslations.end()) { return; }

        auto loaded = parseTranslations(pathIter->second);
        for (const auto &[key, value] : loaded) { translations[key] = value; }
    };

    mergeFrom("en_us");
    if (!language.empty()) {
        std::vector<std::string> languageCandidates;
        auto addCandidate = [&](const std::string &candidate) {
            if (candidate.empty()) { return; }
            if (std::find(languageCandidates.begin(), languageCandidates.end(), candidate) !=
                languageCandidates.end()) {
                return;
            }
            languageCandidates.push_back(candidate);
        };

        std::string normalizedLanguage = normalizeLanguageCode(language);
        addCandidate(normalizedLanguage);

        std::string reducedLanguage = normalizedLanguage;
        while (true) {
            size_t underscore = reducedLanguage.rfind('_');
            if (underscore == std::string::npos) { break; }
            reducedLanguage.resize(underscore);
            addCandidate(reducedLanguage);
        }

        for (const std::string &candidate : languageCandidates) {
            if (candidate != "en_us") { mergeFrom(candidate); }
        }
    }

    return translations;
}

std::vector<std::string>
ShaderPackLoader::parseStringArray(const json &value, std::string_view key, const std::string &context) {
    std::vector<std::string> values;
    auto iter = value.find(std::string(key));
    if (iter == value.end()) { return values; }
    if (!iter->is_array()) { throw std::runtime_error(context + "." + std::string(key) + " must be an array"); }
    values.reserve(iter->size());
    for (size_t i = 0; i < iter->size(); i++) {
        if (!(*iter)[i].is_string()) {
            throw std::runtime_error(context + "." + std::string(key) + "[" + std::to_string(i) + "] must be a string");
        }
        values.push_back((*iter)[i].get<std::string>());
    }
    return values;
}

fs::path ShaderPackLoader::resolveRelativeFile(const fs::path &rootPath,
                                               const std::string &reference,
                                               const std::string &context) {
    if (reference.empty()) { throw std::runtime_error(context + " cannot be empty"); }
    const fs::path relative(reference);
    if (relative.is_absolute()) { throw std::runtime_error(context + " must be relative to shader pack root"); }
    const fs::path resolved = rootPath / relative;
    if (!fs::exists(resolved)) { throw std::runtime_error(context + " references missing file: " + reference); }
    return resolved;
}

std::vector<fs::path> ShaderPackLoader::collectShaderFiles(const fs::path &rootPath) {
    std::vector<fs::path> shaderFiles;
    for (const auto &entry : fs::recursive_directory_iterator(rootPath)) {
        if (entry.is_regular_file()) { shaderFiles.push_back(entry.path()); }
    }
    std::sort(shaderFiles.begin(), shaderFiles.end(),
              [](const fs::path &lhs, const fs::path &rhs) { return lhs.generic_string() < rhs.generic_string(); });
    return shaderFiles;
}

std::unordered_map<std::string, ShaderPackLoader::ClassifiedHitShaderPaths>
ShaderPackLoader::classifyHitShaders(const std::vector<fs::path> &shaderFiles) {
    std::unordered_map<std::string, ShaderPackLoader::ClassifiedHitShaderPaths> classified;
    for (const auto &shaderPath : shaderFiles) {
        std::string extension = toLower(shaderPath.extension().string());
        if (extension == ".rahit" || extension == ".rchit" || extension == ".rint") {
            auto &group = classified[shaderPath.stem().string()];
            if (extension == ".rahit" && !group.anyHit.has_value()) {
                group.anyHit = shaderPath;
            } else if (extension == ".rchit" && !group.closestHit.has_value()) {
                group.closestHit = shaderPath;
            } else if (extension == ".rint" && !group.intersection.has_value()) {
                group.intersection = shaderPath;
            }
        }
    }
    return classified;
}

std::unordered_map<std::string, std::string> ShaderPackLoader::parseDefinitions(const json &jsonValue,
                                                                                const std::string &context) {
    std::unordered_map<std::string, std::string> definitions;

    std::optional<std::reference_wrapper<const json>> iter;
    int present = 0;
    for (std::string_view key :
         {std::string_view(ShaderPackLoader::KEY_DEFINITIONS), std::string_view(ShaderPackLoader::KEY_DEFINES),
          std::string_view(ShaderPackLoader::KEY_DEFINE)}) {
        auto candidate = findKey(jsonValue, {key});
        if (candidate.has_value()) {
            iter = candidate;
            present++;
        }
    }
    if (present > 1) { throw std::runtime_error(context + " cannot define multiple define objects"); }
    if (!iter.has_value()) { return definitions; }
    if (!iter->get().is_object()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINITIONS + " must be an object");
    }
    for (auto defIter = iter->get().begin(); defIter != iter->get().end(); ++defIter) {
        if (!defIter.value().is_string()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINITIONS + "." + defIter.key() +
                                     " must be a string");
        }
        definitions[defIter.key()] = defIter.value().get<std::string>();
    }
    return definitions;
}

std::optional<std::string> ShaderPackLoader::inferScalarType(const json &value, bool allowString) {
    if (value.is_boolean()) { return std::string("bool"); }
    if (value.is_number_integer() || value.is_number_unsigned()) { return std::string("int"); }
    if (value.is_number_float()) { return std::string("float"); }
    if (allowString && value.is_string()) { return std::string("string"); }
    return std::nullopt;
}

std::string ShaderPackLoader::parseScalarValue(const json &value, const std::string &context) {
    if (value.is_string()) { return value.get<std::string>(); }
    if (value.is_boolean()) { return value.get<bool>() ? "true" : "false"; }
    if (value.is_number()) { return value.dump(); }
    throw std::runtime_error(context + " must be a string, boolean, or number");
}

std::string ShaderPackLoader::parseNumericExpressionValue(const json &value, const std::string &context) {
    if (value.is_string()) { return value.get<std::string>(); }
    if (value.is_number()) { return value.dump(); }
    throw std::runtime_error(context + " must be a string or number");
}

ShaderPackLoader::DefineConfig ShaderPackLoader::parseAttributeDefinition(const json &attributeJson,
                                                                          const std::string &context) {
    ShaderPackLoader::DefineConfig define;

    auto defineNode = findKey(attributeJson, {ShaderPackLoader::KEY_DEFINE});
    if (!defineNode.has_value()) { return define; }

    if (defineNode->get().is_string()) {
        define.direct = defineNode->get().get<std::string>();
        return define;
    }

    if (!defineNode->get().is_object()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINE + " must be a string or object");
    }

    bool hasStringValues = false;
    bool hasObjectValues = false;
    for (auto iter = defineNode->get().begin(); iter != defineNode->get().end(); ++iter) {
        hasStringValues |= iter.value().is_string();
        hasObjectValues |= iter.value().is_object();
        if (!iter.value().is_string() && !iter.value().is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINE + "." + iter.key() +
                                     " must be a string or object");
        }
    }

    if (hasStringValues && hasObjectValues) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINE +
                                 " cannot mix string and object entries");
    }

    if (hasObjectValues) {
        for (auto valueIter = defineNode->get().begin(); valueIter != defineNode->get().end(); ++valueIter) {
            auto &definitions = define.variants[valueIter.key()];
            for (auto defIter = valueIter.value().begin(); defIter != valueIter.value().end(); ++defIter) {
                if (!defIter.value().is_string()) {
                    throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINE + "." + valueIter.key() +
                                             "." + defIter.key() + " must be a string");
                }
                definitions[defIter.key()] = defIter.value().get<std::string>();
            }
        }
        return define;
    }

    for (auto defIter = defineNode->get().begin(); defIter != defineNode->get().end(); ++defIter) {
        define.expressions[defIter.key()] = defIter.value().get<std::string>();
    }
    return define;
}

ShaderPackLoader::AttributeConfig ShaderPackLoader::parseAttributeConfig(const json &attributeJson,
                                                                         const std::string &context) {
    if (!attributeJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    ShaderPackLoader::AttributeConfig attribute;
    attribute.name = requireString(attributeJson, ShaderPackLoader::KEY_NAME, context);
    attribute.type = requireString(attributeJson, ShaderPackLoader::KEY_TYPE, context);
    attribute.defaultValue = requireString(attributeJson, ShaderPackLoader::KEY_DEFAULT_VALUE, context);
    if (auto description = optionalString(attributeJson, ShaderPackLoader::KEY_DESCRIPTION, context)) {
        attribute.description = *description;
    }
    if (auto group = optionalString(attributeJson, ShaderPackLoader::KEY_GROUP, context)) { attribute.group = *group; }
    if (auto performanceImpact = optionalString(attributeJson, ShaderPackLoader::KEY_PERFORMANCE_IMPACT, context)) {
        attribute.performanceImpact = *performanceImpact;
    }
    attribute.define = parseAttributeDefinition(attributeJson, context);
    auto visibleWhenIter = attributeJson.find(ShaderPackLoader::KEY_VISIBLE_WHEN);
    if (visibleWhenIter != attributeJson.end()) {
        if (!visibleWhenIter->is_array()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_VISIBLE_WHEN + " must be an array");
        }

        for (size_t i = 0; i < visibleWhenIter->size(); i++) {
            const json &conditionJson = (*visibleWhenIter)[i];
            const std::string conditionContext =
                context + "." + ShaderPackLoader::KEY_VISIBLE_WHEN + "[" + std::to_string(i) + "]";
            if (!conditionJson.is_object()) { throw std::runtime_error(conditionContext + " must be an object"); }

            ShaderPackLoader::AttributeConfig::VisibilityCondition condition;
            condition.attribute = requireString(conditionJson, ShaderPackLoader::KEY_ATTRIBUTE, conditionContext);
            if (auto module = optionalString(conditionJson, ShaderPackLoader::KEY_MODULE, conditionContext)) {
                condition.module = *module;
            }
            auto valuesIter = conditionJson.find(ShaderPackLoader::KEY_VALUES);
            if (valuesIter != conditionJson.end()) {
                if (!valuesIter->is_array()) {
                    throw std::runtime_error(conditionContext + "." + ShaderPackLoader::KEY_VALUES +
                                             " must be an array");
                }

                condition.values.reserve(valuesIter->size());
                for (size_t valueIndex = 0; valueIndex < valuesIter->size(); valueIndex++) {
                    condition.values.push_back(parseScalarValue((*valuesIter)[valueIndex],
                                                                conditionContext + "." + ShaderPackLoader::KEY_VALUES +
                                                                    "[" + std::to_string(valueIndex) + "]"));
                }
            }
            attribute.visibleWhen.push_back(std::move(condition));
        }
    }
    auto linkedValuesIter = attributeJson.find(ShaderPackLoader::KEY_LINKED_VALUES);
    if (linkedValuesIter != attributeJson.end()) {
        if (!linkedValuesIter->is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_LINKED_VALUES + " must be an object");
        }
        for (auto valueIter = linkedValuesIter->begin(); valueIter != linkedValuesIter->end(); ++valueIter) {
            if (!valueIter.value().is_object()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_LINKED_VALUES + "." + valueIter.key() +
                                         " must be an object");
            }

            auto &targets = attribute.linkedValues[valueIter.key()];
            for (auto targetIter = valueIter.value().begin(); targetIter != valueIter.value().end(); ++targetIter) {
                targets[targetIter.key()] =
                    parseScalarValue(targetIter.value(), context + "." + ShaderPackLoader::KEY_LINKED_VALUES + "." +
                                                             valueIter.key() + "." + targetIter.key());
            }
        }
    }

    auto presetIter = attributeJson.find(ShaderPackLoader::KEY_PRESET);
    if (presetIter != attributeJson.end()) {
        const std::string presetContext = context + "." + ShaderPackLoader::KEY_PRESET;
        if (!presetIter->is_object()) { throw std::runtime_error(presetContext + " must be an object"); }
        if (!attribute.linkedValues.empty()) {
            throw std::runtime_error(presetContext + " cannot be combined with " + ShaderPackLoader::KEY_LINKED_VALUES);
        }
        if (attribute.define.direct.has_value() || !attribute.define.expressions.empty() ||
            !attribute.define.variants.empty()) {
            throw std::runtime_error(presetContext + " attributes cannot define shader values");
        }
        if (attribute.type.rfind("enum:", 0) != 0) {
            throw std::runtime_error(presetContext + " requires an enum attribute");
        }

        auto isEnumValue = [&](const std::string &candidate) {
            std::stringstream stream(attribute.type.substr(5));
            std::string option;
            while (std::getline(stream, option, '-')) {
                if (option == candidate) { return true; }
            }
            return false;
        };

        ShaderPackLoader::AttributeConfig::PresetConfig preset;
        preset.customValue = requireString(*presetIter, ShaderPackLoader::KEY_CUSTOM_VALUE, presetContext);
        if (preset.customValue.empty()) {
            throw std::runtime_error(presetContext + "." + ShaderPackLoader::KEY_CUSTOM_VALUE + " cannot be empty");
        }
        if (!isEnumValue(preset.customValue)) {
            throw std::runtime_error(presetContext + "." + ShaderPackLoader::KEY_CUSTOM_VALUE +
                                     " must be one of the enum values");
        }

        auto profilesIter = presetIter->find(ShaderPackLoader::KEY_PROFILES);
        if (profilesIter == presetIter->end() || !profilesIter->is_object() || profilesIter->empty()) {
            throw std::runtime_error(presetContext + "." + ShaderPackLoader::KEY_PROFILES +
                                     " must be a non-empty object");
        }
        for (auto profileIter = profilesIter->begin(); profileIter != profilesIter->end(); ++profileIter) {
            const std::string profileContext =
                presetContext + "." + ShaderPackLoader::KEY_PROFILES + "." + profileIter.key();
            if (profileIter.key().empty()) { throw std::runtime_error(profileContext + " has an empty name"); }
            if (profileIter.key() == preset.customValue) {
                throw std::runtime_error(profileContext + " cannot use the custom value");
            }
            if (!isEnumValue(profileIter.key())) {
                throw std::runtime_error(profileContext + " must use an enum value as its name");
            }
            if (!profileIter.value().is_object() || profileIter.value().empty()) {
                throw std::runtime_error(profileContext + " must be a non-empty object");
            }

            auto &targets = preset.profiles[profileIter.key()];
            for (auto targetIter = profileIter.value().begin(); targetIter != profileIter.value().end(); ++targetIter) {
                if (targetIter.key().empty()) {
                    throw std::runtime_error(profileContext + " contains an empty target attribute name");
                }
                if (targetIter.key() == attribute.name) {
                    throw std::runtime_error(profileContext + " cannot target its own preset attribute");
                }
                targets[targetIter.key()] =
                    parseScalarValue(targetIter.value(), profileContext + "." + targetIter.key());
            }
        }
        attribute.preset = std::move(preset);
    }
    return attribute;
}

ShaderPackLoader::VariableConfig ShaderPackLoader::parseVariableConfig(const json &variableJson,
                                                                       const std::string &context) {
    if (!variableJson.is_object() || variableJson.size() != 1) {
        throw std::runtime_error(context + " must be an object with exactly one variable");
    }

    auto variableIter = variableJson.begin();
    ShaderPackLoader::VariableConfig variable;
    variable.name = variableIter.key();
    if (!ExpressionEvaluator::isValidIdentifier(variable.name)) {
        throw std::runtime_error(context + "." + variable.name + " has an invalid variable name");
    }
    if (ExpressionEvaluator::isReservedIdentifier(variable.name)) {
        throw std::runtime_error(context + "." + variable.name + " uses a reserved variable name");
    }
    if (variableIter.value().is_object()) {
        auto defaultValueIter = variableIter.value().find(ShaderPackLoader::KEY_DEFAULT_VALUE);
        if (defaultValueIter == variableIter.value().end()) {
            throw std::runtime_error(context + "." + variable.name + "." + ShaderPackLoader::KEY_DEFAULT_VALUE +
                                     " must be present");
        }
        auto typeIter = variableIter.value().find(ShaderPackLoader::KEY_TYPE);
        if (typeIter != variableIter.value().end()) {
            if (!typeIter->is_string()) {
                throw std::runtime_error(context + "." + variable.name + "." + ShaderPackLoader::KEY_TYPE +
                                         " must be a string");
            }
            variable.type = typeIter->get<std::string>();
        } else {
            auto inferredType = inferScalarType(*defaultValueIter, true);
            if (!inferredType.has_value()) {
                throw std::runtime_error(context + "." + variable.name + "." + ShaderPackLoader::KEY_TYPE +
                                         " could not be inferred");
            }
            variable.type = *inferredType;
        }
        variable.defaultValue = parseScalarValue(*defaultValueIter, context + "." + variable.name + "." +
                                                                        ShaderPackLoader::KEY_DEFAULT_VALUE);
        return variable;
    }

    auto inferredType = inferScalarType(variableIter.value(), true);
    if (!inferredType.has_value()) {
        throw std::runtime_error(context + "." + variable.name + " must be an object or scalar");
    }
    variable.type = *inferredType;
    variable.defaultValue = parseScalarValue(variableIter.value(), context + "." + variable.name);
    return variable;
}

ShaderPackLoader::TextureConfig
ShaderPackLoader::parseTextureConfig(const json &textureJson, const fs::path &rootPath, const std::string &context) {
    if (!textureJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    ShaderPackLoader::TextureConfig texture;
    texture.name = requireString(textureJson, ShaderPackLoader::KEY_NAME, context);
    std::string type = requireString(textureJson, ShaderPackLoader::KEY_TYPE, context);
    std::string loweredType = toLower(type);
    if (loweredType == ShaderPackLoader::VALUE_FILE) {
        texture.imported = true;
    } else if (loweredType == ShaderPackLoader::VALUE_INTERMEDIATE) {
        texture.imported = false;
    } else {
        throw std::runtime_error("unsupported texture type: " + type);
    }
    texture.dimension = parseTextureDimension(
        optionalString(textureJson, ShaderPackLoader::KEY_DIMENSION, context).value_or(ShaderPackLoader::VALUE_2D));
    texture.filter = parseFilter(optionalString(textureJson, ShaderPackLoader::KEY_FILTER, context).value_or("linear"));
    texture.addressMode =
        parseAddressMode(optionalString(textureJson, ShaderPackLoader::KEY_ADDRESS_MODE, context).value_or("repeat"));
    texture.shared = optionalBool(textureJson, ShaderPackLoader::KEY_SHARED, false, context);
    texture.transferSrc = optionalBool(textureJson, ShaderPackLoader::KEY_TRANSFER_SRC, false, context);

    auto sampledBindingIter = textureJson.find(ShaderPackLoader::KEY_SAMPLED_BINDING);
    if (sampledBindingIter != textureJson.end()) {
        if (!sampledBindingIter->is_number_unsigned()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_SAMPLED_BINDING +
                                     " must be an unsigned integer");
        }
        texture.sampledBinding = sampledBindingIter->get<uint32_t>();
    }

    auto storageBindingIter = textureJson.find(ShaderPackLoader::KEY_STORAGE_BINDING);
    if (storageBindingIter != textureJson.end()) {
        if (!storageBindingIter->is_number_unsigned()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_STORAGE_BINDING +
                                     " must be an unsigned integer");
        }
        texture.storageBinding = storageBindingIter->get<uint32_t>();
    }
    if (texture.imported) {
        texture.sourcePath =
            resolveRelativeFile(rootPath, requireString(textureJson, ShaderPackLoader::KEY_PATH, context),
                                context + "." + ShaderPackLoader::KEY_PATH);
        texture.format =
            parseFormat(optionalString(textureJson, ShaderPackLoader::KEY_FORMAT, context).value_or("R8G8B8A8_UNORM"));
        if (!texture.sampledBinding.has_value()) { throw std::runtime_error(context + " requires sampled_binding"); }
        if (texture.dimension == ShaderPackLoader::TextureDimension::Texture3D ||
            texture.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
            auto parseImportedExtent = [&](std::string_view key) {
                const std::string keyString(key);
                auto iter = textureJson.find(keyString);
                if (iter == textureJson.end() || !iter->is_number_unsigned()) {
                    throw std::runtime_error(context + "." + keyString +
                                             " must be an unsigned integer for imported layered textures");
                }
                return iter->get<uint32_t>();
            };
            texture.importedWidth = parseImportedExtent(ShaderPackLoader::KEY_WIDTH);
            texture.importedHeight = parseImportedExtent(ShaderPackLoader::KEY_HEIGHT);
            texture.importedDepth = parseImportedExtent(ShaderPackLoader::KEY_DEPTH);
            if (texture.importedWidth == 0 || texture.importedHeight == 0 || texture.importedDepth <= 1) {
                throw std::runtime_error(context +
                                         " imported layered textures require positive width/height and depth > 1");
            }
        } else if (texture.dimension != ShaderPackLoader::TextureDimension::Texture2D) {
            throw std::runtime_error(context + " currently only supports imported 2d, 2d_array, or 3d textures");
        }
    } else {
        texture.format = parseFormat(requireString(textureJson, ShaderPackLoader::KEY_FORMAT, context));
        auto widthIter = textureJson.find(ShaderPackLoader::KEY_WIDTH);
        auto heightIter = textureJson.find(ShaderPackLoader::KEY_HEIGHT);
        auto depthIter = textureJson.find(ShaderPackLoader::KEY_DEPTH);
        auto scaleIter = textureJson.find(ShaderPackLoader::KEY_SCALE);
        if (widthIter != textureJson.end() && heightIter != textureJson.end()) {
            texture.widthExpression =
                parseNumericExpressionValue(*widthIter, context + "." + ShaderPackLoader::KEY_WIDTH);
            texture.heightExpression =
                parseNumericExpressionValue(*heightIter, context + "." + ShaderPackLoader::KEY_HEIGHT);
            if (texture.dimension == ShaderPackLoader::TextureDimension::Texture3D ||
                texture.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
                if (depthIter == textureJson.end()) {
                    throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEPTH +
                                             " must be present for intermediate layered textures");
                }
                texture.depthExpression =
                    parseNumericExpressionValue(*depthIter, context + "." + ShaderPackLoader::KEY_DEPTH);
            } else if (depthIter != textureJson.end()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEPTH +
                                         " is only supported for intermediate 2d_array or 3d textures");
            }
        } else if (scaleIter != textureJson.end()) {
            if (texture.dimension == ShaderPackLoader::TextureDimension::Texture3D ||
                texture.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
                throw std::runtime_error(context +
                                         " intermediate 2d_array and 3d textures require explicit width/height/depth");
            }
            if (!scaleIter->is_number()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_SCALE + " must be numeric");
            }
            texture.scale = scaleIter->get<float>();
        } else {
            throw std::runtime_error(context + " requires width/height or scale");
        }
    }

    return texture;
}

ShaderPackLoader::BufferConfig ShaderPackLoader::parseBufferConfig(const json &bufferJson, const std::string &context) {
    if (!bufferJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    ShaderPackLoader::BufferConfig buffer;
    buffer.name = requireString(bufferJson, ShaderPackLoader::KEY_NAME, context);
    std::string type = requireString(bufferJson, ShaderPackLoader::KEY_TYPE, context);
    if (toLower(type) != ShaderPackLoader::VALUE_INTERMEDIATE) {
        throw std::runtime_error("unsupported buffer type: " + type);
    }

    buffer.shared = optionalBool(bufferJson, ShaderPackLoader::KEY_SHARED, false, context);
    auto sizeIter = bufferJson.find(ShaderPackLoader::KEY_SIZE);
    if (sizeIter == bufferJson.end()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_SIZE + " must be present");
    }
    buffer.sizeExpression = parseNumericExpressionValue(*sizeIter, context + "." + ShaderPackLoader::KEY_SIZE);

    auto bindingIter = bufferJson.find(ShaderPackLoader::KEY_BINDING);
    if (bindingIter == bufferJson.end() || !bindingIter->is_number_unsigned()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_BINDING + " must be an unsigned integer");
    }
    buffer.binding = bindingIter->get<uint32_t>();
    return buffer;
}

ShaderPackLoader::Stage ShaderPackLoader::parseStage(const json &value, const std::string &context) {
    if (!value.is_string()) { throw std::runtime_error(context + " must be a string"); }

    const std::string stage = toLower(value.get<std::string>());
    if (stage == ShaderPackLoader::KEY_RAY_TRACING) { return ShaderPackLoader::Stage::RayTracing; }
    if (stage == ShaderPackLoader::KEY_POST_RENDER) { return ShaderPackLoader::Stage::PostRender; }
    throw std::runtime_error("unsupported stage: " + value.get<std::string>());
}

ShaderPackLoader::ResourceAccessKind ShaderPackLoader::parseResourceAccessKind(const std::string &value,
                                                                               const std::string &context) {
    const std::string access = toLower(value);
    if (access == "read" || access == "shader_read" || access == "sampled_read" || access == "storage_read") {
        return ShaderPackLoader::ResourceAccessKind::Read;
    }
    if (access == "write" || access == "shader_write") { return ShaderPackLoader::ResourceAccessKind::Write; }
    if (access == "read_write" || access == "readwrite" || access == "storage_read_write" || access == "accumulate") {
        return ShaderPackLoader::ResourceAccessKind::ReadWrite;
    }
    if (access == "overwrite" || access == "discard_write") { return ShaderPackLoader::ResourceAccessKind::Overwrite; }
    if (access == "append") { return ShaderPackLoader::ResourceAccessKind::Append; }
    if (access == "append_counter" || access == "counter") {
        return ShaderPackLoader::ResourceAccessKind::AppendCounter;
    }
    if (access == "indirect_read" || access == "dispatch_indirect_read" || access == "draw_indirect_read") {
        return ShaderPackLoader::ResourceAccessKind::IndirectRead;
    }
    throw std::runtime_error(context + " has unsupported resource access kind: " + value);
}

ShaderPackLoader::ResourceList
ShaderPackLoader::parsePassResourceList(const json &value, std::string_view key, const std::string &context) {
    ShaderPackLoader::ResourceList resources;

    auto iter = value.find(std::string(key));
    if (iter == value.end()) { return resources; }

    const std::string keyContext = context + "." + std::string(key);
    if (!iter->is_object()) { throw std::runtime_error(keyContext + " must be an object"); }

    for (auto fieldIter = iter->begin(); fieldIter != iter->end(); ++fieldIter) {
        if (fieldIter.key() != ShaderPackLoader::KEY_IMAGES && fieldIter.key() != ShaderPackLoader::KEY_BUFFERS) {
            throw std::runtime_error(keyContext + " only supports " + ShaderPackLoader::KEY_IMAGES + " and " +
                                     ShaderPackLoader::KEY_BUFFERS);
        }
    }

    const ShaderPackLoader::ResourceAccessKind defaultAccess = key == ShaderPackLoader::KEY_INPUTS ?
                                                                   ShaderPackLoader::ResourceAccessKind::Read :
                                                                   ShaderPackLoader::ResourceAccessKind::ReadWrite;

    auto appendUniqueName = [](std::vector<std::string> &names, const std::string &name) {
        if (std::find(names.begin(), names.end(), name) == names.end()) { names.push_back(name); }
    };

    auto parseAccessValue = [&](const json &accessJson, const std::string &entryContext) {
        if (accessJson.is_string()) { return parseResourceAccessKind(accessJson.get<std::string>(), entryContext); }
        if (accessJson.is_object()) {
            for (auto fieldIter = accessJson.begin(); fieldIter != accessJson.end(); ++fieldIter) {
                if (fieldIter.key() != ShaderPackLoader::KEY_ACCESS) {
                    throw std::runtime_error(entryContext + " only supports " + ShaderPackLoader::KEY_ACCESS);
                }
            }
            return parseResourceAccessKind(requireString(accessJson, ShaderPackLoader::KEY_ACCESS, entryContext),
                                           entryContext + "." + ShaderPackLoader::KEY_ACCESS);
        }
        throw std::runtime_error(entryContext + " must be a string access kind or an object");
    };

    auto parseResourceEntries = [&](std::string_view resourceKey, std::vector<std::string> &names,
                                    std::vector<ShaderPackLoader::ResourceAccess> &accesses) {
        auto resourceIter = iter->find(std::string(resourceKey));
        if (resourceIter == iter->end()) { return; }

        const std::string resourceContext = keyContext + "." + std::string(resourceKey);
        if (resourceIter->is_array()) {
            for (size_t i = 0; i < resourceIter->size(); ++i) {
                const json &entryJson = (*resourceIter)[i];
                const std::string entryContext = resourceContext + "[" + std::to_string(i) + "]";
                if (entryJson.is_string()) {
                    const std::string name = entryJson.get<std::string>();
                    appendUniqueName(names, name);
                    accesses.push_back({
                        .name = name,
                        .kind = defaultAccess,
                    });
                    continue;
                }
                if (entryJson.is_object()) {
                    for (auto fieldIter = entryJson.begin(); fieldIter != entryJson.end(); ++fieldIter) {
                        if (fieldIter.key() != ShaderPackLoader::KEY_NAME &&
                            fieldIter.key() != ShaderPackLoader::KEY_ACCESS) {
                            throw std::runtime_error(entryContext + " only supports " + ShaderPackLoader::KEY_NAME +
                                                     " and " + ShaderPackLoader::KEY_ACCESS);
                        }
                    }
                    const std::string name = requireString(entryJson, ShaderPackLoader::KEY_NAME, entryContext);
                    const ShaderPackLoader::ResourceAccessKind access =
                        entryJson.contains(ShaderPackLoader::KEY_ACCESS) ?
                            parseAccessValue(entryJson.at(ShaderPackLoader::KEY_ACCESS),
                                             entryContext + "." + ShaderPackLoader::KEY_ACCESS) :
                            defaultAccess;
                    appendUniqueName(names, name);
                    accesses.push_back({
                        .name = name,
                        .kind = access,
                    });
                    continue;
                }
                throw std::runtime_error(entryContext + " must be a string or an object");
            }
            return;
        }

        if (resourceIter->is_object()) {
            for (auto resourceEntry = resourceIter->begin(); resourceEntry != resourceIter->end(); ++resourceEntry) {
                const std::string name = resourceEntry.key();
                appendUniqueName(names, name);
                accesses.push_back({
                    .name = name,
                    .kind = parseAccessValue(resourceEntry.value(), resourceContext + "." + name),
                });
            }
            return;
        }

        throw std::runtime_error(resourceContext + " must be an array or object");
    };

    parseResourceEntries(ShaderPackLoader::KEY_IMAGES, resources.images, resources.imageAccesses);
    parseResourceEntries(ShaderPackLoader::KEY_BUFFERS, resources.buffers, resources.bufferAccesses);
    return resources;
}

ShaderPackLoader::RayTracingPassConfig ShaderPackLoader::parseRayTracingPassConfig(
    const json &passJson,
    const fs::path &rootPath,
    const std::unordered_map<std::string, ShaderPackLoader::ClassifiedHitShaderPaths> &classifiedHitShaders,
    const std::string &context) {
    ShaderPackLoader::RayTracingPassConfig pass;
    pass.name = requireString(passJson, ShaderPackLoader::KEY_NAME, context);
    pass.asyncGroup = optionalString(passJson, ShaderPackLoader::KEY_ASYNC_GROUP, context).value_or("");
    pass.querySharc = optionalBool(passJson, ShaderPackLoader::KEY_QUERY_SHARC, false, context);
    pass.inputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_INPUTS, context);
    pass.outputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_OUTPUTS, context);
    pass.indirectArgsBuffer = optionalString(passJson, ShaderPackLoader::KEY_INDIRECT_ARGS, context);
    if (auto widthIter = passJson.find(ShaderPackLoader::KEY_WIDTH); widthIter != passJson.end()) {
        pass.width = parseNumericExpressionValue(*widthIter, context + "." + ShaderPackLoader::KEY_WIDTH);
    }
    if (auto heightIter = passJson.find(ShaderPackLoader::KEY_HEIGHT); heightIter != passJson.end()) {
        pass.height = parseNumericExpressionValue(*heightIter, context + "." + ShaderPackLoader::KEY_HEIGHT);
    }
    if (auto depthIter = passJson.find(ShaderPackLoader::KEY_DEPTH); depthIter != passJson.end()) {
        pass.depth = parseNumericExpressionValue(*depthIter, context + "." + ShaderPackLoader::KEY_DEPTH);
    }
    const auto recursionIter = passJson.find(ShaderPackLoader::KEY_MAX_RAY_RECURSION_DEPTH);
    if (recursionIter == passJson.end() || !recursionIter->is_number_unsigned() ||
        recursionIter->get<uint32_t>() == 0) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_MAX_RAY_RECURSION_DEPTH +
                                 " must be a positive unsigned integer");
    }
    pass.maxRayRecursionDepth = recursionIter->get<uint32_t>();
    pass.definitions = parseDefinitions(passJson, context);

    pass.rayGenShaderPath = resolveRelativeFile(rootPath, requireString(passJson, ShaderPackLoader::KEY_RGEN, context),
                                                context + "." + ShaderPackLoader::KEY_RGEN);
    pass.defaultHitGroupName = requireString(passJson, ShaderPackLoader::KEY_DEFAULT_HIT_GROUP, context);

    const json &hitGroupsJson = requireObject(passJson, ShaderPackLoader::KEY_HIT_GROUPS, context);
    if (hitGroupsJson.empty()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_HIT_GROUPS + " must not be empty");
    }
    if (!hitGroupsJson.contains(pass.defaultHitGroupName)) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFAULT_HIT_GROUP + " is missing in hit_groups");
    }

    std::vector<std::string> groupNames;
    groupNames.reserve(hitGroupsJson.size());
    groupNames.push_back(pass.defaultHitGroupName);
    for (auto iter = hitGroupsJson.begin(); iter != hitGroupsJson.end(); ++iter) {
        if (iter.key() != pass.defaultHitGroupName) { groupNames.push_back(iter.key()); }
    }
    std::sort(groupNames.begin() + 1, groupNames.end());

    auto parseGroupType = [&](const json &groupJson, const std::string &groupName) {
        const std::string typeValue = requireString(groupJson, ShaderPackLoader::KEY_TYPE,
                                                    context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName);
        if (typeValue == ShaderPackLoader::VALUE_TRIANGLE)
            return VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        if (typeValue == ShaderPackLoader::VALUE_AABB) return VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        throw std::runtime_error("unsupported hit group type: " + typeValue);
    };

    auto parseGroup = [&](const std::string &groupName, const std::optional<fs::path> &fallbackClosestHit) {
        const json &groupJson = hitGroupsJson.at(groupName);
        if (!groupJson.is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName +
                                     " must be an object");
        }

        const json &shadersJson = requireObject(groupJson, ShaderPackLoader::KEY_SHADERS,
                                                context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName);
        auto classifiedIter = classifiedHitShaders.find(groupName);
        std::optional<std::reference_wrapper<const ShaderPackLoader::ClassifiedHitShaderPaths>> classified;
        if (classifiedIter != classifiedHitShaders.end()) { classified = std::cref(classifiedIter->second); }

        ShaderPackLoader::HitGroupConfig group;
        group.name = groupName;
        group.definitions =
            parseDefinitions(groupJson, context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName);
        if (auto closestRef = optionalString(shadersJson, ShaderPackLoader::KEY_RCHIT,
                                             context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName + "." +
                                                 ShaderPackLoader::KEY_SHADERS);
            closestRef.has_value()) {
            group.closestHit =
                resolveRelativeFile(rootPath, *closestRef,
                                    context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName + "." +
                                        ShaderPackLoader::KEY_SHADERS + "." + ShaderPackLoader::KEY_RCHIT);
        } else if (classified.has_value() && classified->get().closestHit.has_value()) {
            group.closestHit = classified->get().closestHit;
        }

        if (auto anyRef = optionalString(shadersJson, ShaderPackLoader::KEY_RAHIT,
                                         context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName + "." +
                                             ShaderPackLoader::KEY_SHADERS);
            anyRef.has_value()) {
            group.anyHit =
                resolveRelativeFile(rootPath, *anyRef,
                                    context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName + "." +
                                        ShaderPackLoader::KEY_SHADERS + "." + ShaderPackLoader::KEY_RAHIT);
        } else if (classified.has_value() && classified->get().anyHit.has_value()) {
            group.anyHit = classified->get().anyHit;
        }

        if (auto intersectionRef = optionalString(shadersJson, ShaderPackLoader::KEY_RINT,
                                                  context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName +
                                                      "." + ShaderPackLoader::KEY_SHADERS);
            intersectionRef.has_value()) {
            group.intersection =
                resolveRelativeFile(rootPath, *intersectionRef,
                                    context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName + "." +
                                        ShaderPackLoader::KEY_SHADERS + "." + ShaderPackLoader::KEY_RINT);
        } else if (classified.has_value() && classified->get().intersection.has_value()) {
            group.intersection = classified->get().intersection;
        }

        if (!group.closestHit.has_value() && !group.intersection.has_value() && fallbackClosestHit.has_value()) {
            group.closestHit = fallbackClosestHit;
        }

        VkRayTracingShaderGroupTypeKHR requestedType = parseGroupType(groupJson, groupName);
        if (group.intersection.has_value()) {
            group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        } else if (requestedType == VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR) {
            throw std::runtime_error("aabb hit group requires rint: " + groupName);
        } else {
            group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        }

        if (!group.closestHit.has_value() && !group.intersection.has_value()) {
            throw std::runtime_error("hit group must provide rchit or rint: " + groupName);
        }

        return group;
    };

    ShaderPackLoader::HitGroupConfig defaultGroup = parseGroup(pass.defaultHitGroupName, std::nullopt);
    pass.hitGroups.push_back(defaultGroup);
    for (size_t i = 1; i < groupNames.size(); i++) {
        pass.hitGroups.push_back(parseGroup(groupNames[i], defaultGroup.closestHit));
    }

    const json &missShadersJson = requireArray(passJson, ShaderPackLoader::KEY_MISS, context);
    if (missShadersJson.empty()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_MISS + " must not be empty");
    }
    for (size_t i = 0; i < missShadersJson.size(); i++) {
        const json &missJson = missShadersJson[i];
        if (!missJson.is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_MISS + "[" + std::to_string(i) +
                                     "] must be an object");
        }
        pass.missShaders.push_back({
            .name = requireString(missJson, ShaderPackLoader::KEY_NAME,
                                  context + "." + ShaderPackLoader::KEY_MISS + "[" + std::to_string(i) + "]"),
            .shaderPath = resolveRelativeFile(
                rootPath,
                requireString(missJson, ShaderPackLoader::KEY_SHADER,
                              context + "." + ShaderPackLoader::KEY_MISS + "[" + std::to_string(i) + "]"),
                context + "." + ShaderPackLoader::KEY_MISS + "[" + std::to_string(i) + "]." +
                    ShaderPackLoader::KEY_SHADER),
        });
    }

    return pass;
}

ShaderPackLoader::FullScreenPassConfig ShaderPackLoader::parseFullScreenPassConfig(const json &passJson,
                                                                                   const fs::path &rootPath,
                                                                                   const std::string &context) {
    ShaderPackLoader::FullScreenPassConfig pass;
    pass.name = requireString(passJson, ShaderPackLoader::KEY_NAME, context);
    pass.fragmentShaderPath =
        resolveRelativeFile(rootPath, requireString(passJson, ShaderPackLoader::KEY_FRAGMENT, context),
                            context + "." + ShaderPackLoader::KEY_FRAGMENT);
    pass.target = requireString(passJson, ShaderPackLoader::KEY_TARGET, context);
    pass.inputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_INPUTS, context);
    pass.outputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_OUTPUTS, context);
    if (std::find(pass.outputs.images.begin(), pass.outputs.images.end(), pass.target) == pass.outputs.images.end()) {
        pass.outputs.images.push_back(pass.target);
        pass.outputs.imageAccesses.push_back({
            .name = pass.target,
            .kind = ShaderPackLoader::ResourceAccessKind::ReadWrite,
        });
    }
    auto faceIter = passJson.find(ShaderPackLoader::KEY_FACE);
    if (faceIter != passJson.end()) {
        if (!faceIter->is_number_integer()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_FACE + " must be an integer");
        }
        int face = faceIter->get<int>();
        if (face < 0 || face > 5) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_FACE + " must be between 0 and 5");
        }
        pass.face = static_cast<uint32_t>(face);
    }
    auto layerIter = passJson.find(ShaderPackLoader::KEY_LAYER);
    if (layerIter != passJson.end()) {
        if (!layerIter->is_number_unsigned()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_LAYER + " must be an unsigned integer");
        }
        pass.layer = layerIter->get<uint32_t>();
    }
    if (pass.face.has_value() && pass.layer.has_value()) {
        throw std::runtime_error(context + " cannot specify both face and layer");
    }
    pass.definitions = parseDefinitions(passJson, context);
    return pass;
}

ShaderPackLoader::RenderShaderConfig ShaderPackLoader::parseRenderShaderConfig(const json &shaderJson,
                                                                               const fs::path &rootPath,
                                                                               const std::string &context) {
    if (!shaderJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    ShaderPackLoader::RenderShaderConfig shader;
    shader.vertexShaderPath =
        resolveRelativeFile(rootPath, requireString(shaderJson, ShaderPackLoader::KEY_VERTEX, context),
                            context + "." + ShaderPackLoader::KEY_VERTEX);
    shader.fragmentShaderPath =
        resolveRelativeFile(rootPath, requireString(shaderJson, ShaderPackLoader::KEY_FRAGMENT, context),
                            context + "." + ShaderPackLoader::KEY_FRAGMENT);
    return shader;
}

ShaderPackLoader::RenderPipelineConfig ShaderPackLoader::parseRenderPipelineConfig(const json &pipelineJson,
                                                                                   const fs::path &rootPath,
                                                                                   const std::string &context) {
    if (!pipelineJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    auto requireBoolField = [&](std::string_view key) -> bool {
        auto iter = pipelineJson.find(std::string(key));
        if (iter == pipelineJson.end()) { throw std::runtime_error(context + "." + std::string(key) + " is required"); }
        if (!iter->is_boolean()) { throw std::runtime_error(context + "." + std::string(key) + " must be a boolean"); }
        return iter->get<bool>();
    };

    ShaderPackLoader::RenderPipelineConfig pipeline;
    pipeline.shader = parseRenderShaderConfig(requireObject(pipelineJson, ShaderPackLoader::KEY_SHADER, context),
                                              rootPath, context + "." + ShaderPackLoader::KEY_SHADER);
    pipeline.depthTest = requireBoolField(ShaderPackLoader::KEY_DEPTH_TEST);
    pipeline.depthWrite = requireBoolField(ShaderPackLoader::KEY_DEPTH_WRITE);
    pipeline.depthCompare = requireString(pipelineJson, ShaderPackLoader::KEY_DEPTH_COMPARE, context);
    pipeline.colorBlend = requireBoolField(ShaderPackLoader::KEY_COLOR_BLEND);
    pipeline.colorWriteMasks = parseStringArray(pipelineJson, ShaderPackLoader::KEY_COLOR_WRITE_MASK, context);
    pipeline.cull = requireString(pipelineJson, ShaderPackLoader::KEY_CULL, context);
    pipeline.frontFace = requireString(pipelineJson, ShaderPackLoader::KEY_FRONT_FACE, context);
    pipeline.definitions = parseDefinitions(pipelineJson, context);
    return pipeline;
}

ShaderPackLoader::RenderPassConfig
ShaderPackLoader::parseRenderPassConfig(const json &passJson, const fs::path &rootPath, const std::string &context) {
    ShaderPackLoader::RenderPassConfig pass;
    pass.name = requireString(passJson, ShaderPackLoader::KEY_NAME, context);
    if (passJson.contains(ShaderPackLoader::KEY_SHADER) || passJson.contains(ShaderPackLoader::KEY_VERTEX) ||
        passJson.contains(ShaderPackLoader::KEY_FRAGMENT) || passJson.contains(ShaderPackLoader::KEY_DEPTH_TEST) ||
        passJson.contains(ShaderPackLoader::KEY_DEPTH_WRITE) ||
        passJson.contains(ShaderPackLoader::KEY_DEPTH_COMPARE) ||
        passJson.contains(ShaderPackLoader::KEY_COLOR_BLEND) ||
        passJson.contains(ShaderPackLoader::KEY_COLOR_WRITE_MASK) || passJson.contains(ShaderPackLoader::KEY_CULL) ||
        passJson.contains(ShaderPackLoader::KEY_FRONT_FACE)) {
        throw std::runtime_error(context +
                                 " render passes must configure per-layer graphics state in pipeline entries");
    }
    const json &pipelineJson = requireObject(passJson, ShaderPackLoader::KEY_PIPELINE, context);
    if (pipelineJson.empty()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_PIPELINE + " must be a non-empty object");
    }
    for (auto pipelineEntry = pipelineJson.begin(); pipelineEntry != pipelineJson.end(); ++pipelineEntry) {
        pass.pipelineConfigs.emplace(
            pipelineEntry.key(),
            parseRenderPipelineConfig(pipelineEntry.value(), rootPath,
                                      context + "." + ShaderPackLoader::KEY_PIPELINE + "." + pipelineEntry.key()));
    }
    if (auto layerOrderIter = passJson.find(ShaderPackLoader::KEY_LAYER_ORDER); layerOrderIter != passJson.end()) {
        if (!layerOrderIter->is_array()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_LAYER_ORDER + " must be an array");
        }
        for (size_t i = 0; i < layerOrderIter->size(); i++) {
            const json &layerJson = (*layerOrderIter)[i];
            if (!layerJson.is_string()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_LAYER_ORDER + "[" + std::to_string(i) +
                                         "] must be a string");
            }
            pass.layerOrder.push_back(layerJson.get<std::string>());
        }
    }
    pass.content = optionalString(passJson, ShaderPackLoader::KEY_CONTENT, context).value_or("");
    pass.inputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_INPUTS, context);
    pass.outputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_OUTPUTS, context);
    if (pass.outputs.images.empty()) {
        pass.outputs.images.push_back("out:ldr");
        pass.outputs.imageAccesses.push_back({
            .name = "out:ldr",
            .kind = ShaderPackLoader::ResourceAccessKind::ReadWrite,
        });
    }
    pass.depthLoad = optionalBool(passJson, ShaderPackLoader::KEY_DEPTH_LOAD, false, context);
    if (auto clearOutputsIter = passJson.find(ShaderPackLoader::KEY_CLEAR_OUTPUTS);
        clearOutputsIter != passJson.end()) {
        if (!clearOutputsIter->is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_CLEAR_OUTPUTS + " must be an object");
        }
        for (auto clearIter = clearOutputsIter->begin(); clearIter != clearOutputsIter->end(); ++clearIter) {
            const std::string &outputName = clearIter.key();
            if (std::find(pass.outputs.images.begin(), pass.outputs.images.end(), outputName) ==
                pass.outputs.images.end()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_CLEAR_OUTPUTS + "." + outputName +
                                         " must reference a render pass output image");
            }
            if (!clearIter.value().is_array() || clearIter.value().size() != 4) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_CLEAR_OUTPUTS + "." + outputName +
                                         " must be an array of four unsigned integers");
            }

            std::array<uint32_t, 4> clearValue{};
            for (size_t componentIndex = 0; componentIndex < clearValue.size(); ++componentIndex) {
                const json &componentJson = clearIter.value()[componentIndex];
                if (!componentJson.is_number_unsigned()) {
                    throw std::runtime_error(context + "." + ShaderPackLoader::KEY_CLEAR_OUTPUTS + "." + outputName +
                                             "[" + std::to_string(componentIndex) + "] must be an unsigned integer");
                }
                clearValue[componentIndex] = componentJson.get<uint32_t>();
            }
            pass.clearOutputs.emplace(outputName, clearValue);
        }
    }
    if (auto indirectIter = passJson.find(ShaderPackLoader::KEY_INDIRECT_DRAW); indirectIter != passJson.end()) {
        if (!indirectIter->is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_INDIRECT_DRAW + " must be an object");
        }
        RenderPassConfig::IndirectDrawConfig indirect;
        const std::string indirectContext = context + "." + ShaderPackLoader::KEY_INDIRECT_DRAW;
        if (indirectIter->contains(ShaderPackLoader::KEY_COMMANDS_BUFFER)) {
            throw std::runtime_error(indirectContext + "." + ShaderPackLoader::KEY_COMMANDS_BUFFER +
                                     " is native-provided and must not be declared");
        }
        if (indirectIter->contains("metadata_buffer")) {
            throw std::runtime_error(indirectContext +
                                     ".metadata_buffer is native-provided and must not be declared in set 5");
        }
        if (indirectIter->contains(ShaderPackLoader::KEY_COMPUTE)) {
            throw std::runtime_error(indirectContext + "." + ShaderPackLoader::KEY_COMPUTE +
                                     " is native-provided and must not be declared");
        }
        if (indirectIter->contains(ShaderPackLoader::KEY_WORKGROUP_SIZE)) {
            throw std::runtime_error(indirectContext + "." + ShaderPackLoader::KEY_WORKGROUP_SIZE +
                                     " is native-provided and must not be declared");
        }
        indirect.batchSortLayers =
            parseStringArray(*indirectIter, ShaderPackLoader::KEY_BATCH_SORT_LAYERS, indirectContext);
        pass.indirectDraw = std::move(indirect);
    }
    pass.definitions = parseDefinitions(passJson, context);
    return pass;
}

ShaderPackLoader::ComputePassConfig
ShaderPackLoader::parseComputePassConfig(const json &passJson, const fs::path &rootPath, const std::string &context) {
    ShaderPackLoader::ComputePassConfig pass;
    pass.name = requireString(passJson, ShaderPackLoader::KEY_NAME, context);
    pass.asyncGroup = optionalString(passJson, ShaderPackLoader::KEY_ASYNC_GROUP, context).value_or("");
    pass.querySharc = optionalBool(passJson, ShaderPackLoader::KEY_QUERY_SHARC, false, context);
    pass.computeShaderPath =
        resolveRelativeFile(rootPath, requireString(passJson, ShaderPackLoader::KEY_COMPUTE, context),
                            context + "." + ShaderPackLoader::KEY_COMPUTE);
    pass.inputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_INPUTS, context);
    pass.outputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_OUTPUTS, context);
    pass.indirectArgsBuffer = optionalString(passJson, ShaderPackLoader::KEY_INDIRECT_ARGS, context);
    if (auto gxIter = passJson.find(ShaderPackLoader::KEY_GX); gxIter != passJson.end()) {
        pass.gx = parseNumericExpressionValue(*gxIter, context + "." + ShaderPackLoader::KEY_GX);
    }
    if (auto gyIter = passJson.find(ShaderPackLoader::KEY_GY); gyIter != passJson.end()) {
        pass.gy = parseNumericExpressionValue(*gyIter, context + "." + ShaderPackLoader::KEY_GY);
    }
    if (auto gzIter = passJson.find(ShaderPackLoader::KEY_GZ); gzIter != passJson.end()) {
        pass.gz = parseNumericExpressionValue(*gzIter, context + "." + ShaderPackLoader::KEY_GZ);
    }
    pass.definitions = parseDefinitions(passJson, context);
    return pass;
}

ShaderPackLoader::PassConfig ShaderPackLoader::parsePassConfig(
    const json &passJson,
    const fs::path &rootPath,
    const std::unordered_map<std::string, ShaderPackLoader::ClassifiedHitShaderPaths> &classifiedHitShaders,
    const std::string &context) {
    if (!passJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    const std::string type = requireString(passJson, ShaderPackLoader::KEY_TYPE, context);
    ShaderPackLoader::PassConfig pass;
    if (type == ShaderPackLoader::VALUE_FULL_SCREEN) {
        pass.type = ShaderPackLoader::PassConfig::Type::FullScreen;
    } else if (type == ShaderPackLoader::VALUE_RENDER) {
        pass.type = ShaderPackLoader::PassConfig::Type::Render;
    } else if (type == ShaderPackLoader::VALUE_RAY_TRACING) {
        pass.type = ShaderPackLoader::PassConfig::Type::RayTracing;
    } else if (type == ShaderPackLoader::VALUE_COMPUTE) {
        pass.type = ShaderPackLoader::PassConfig::Type::Compute;
    } else {
        throw std::runtime_error("unsupported pass type: " + type);
    }

    if (auto stageIter = passJson.find(ShaderPackLoader::KEY_STAGE); stageIter != passJson.end()) {
        pass.stage = parseStage(*stageIter, context + "." + ShaderPackLoader::KEY_STAGE);
    } else {
        pass.stage = ShaderPackLoader::Stage::RayTracing;
    }

    if (pass.type == ShaderPackLoader::PassConfig::Type::RayTracing &&
        pass.stage != ShaderPackLoader::Stage::RayTracing) {
        throw std::runtime_error("ray_tracing pass must use ray_tracing stage: " + context);
    }
    (void)context;

    switch (pass.type) {
        case ShaderPackLoader::PassConfig::Type::FullScreen:
            pass.fullScreen = parseFullScreenPassConfig(passJson, rootPath, context);
            return pass;
        case ShaderPackLoader::PassConfig::Type::Render:
            pass.render = parseRenderPassConfig(passJson, rootPath, context);
            return pass;
        case ShaderPackLoader::PassConfig::Type::RayTracing:
            pass.rayTracing = parseRayTracingPassConfig(passJson, rootPath, classifiedHitShaders, context);
            return pass;
        case ShaderPackLoader::PassConfig::Type::Compute:
            pass.compute = parseComputePassConfig(passJson, rootPath, context);
            return pass;
    }

    throw std::runtime_error("unsupported pass type: " + type);
}

ShaderPackLoader::ExecutionCommand ShaderPackLoader::parseExecutionCommand(const json &commandJson,
                                                                           const std::string &context) {
    if (!commandJson.is_object() || commandJson.size() != 1) {
        throw std::runtime_error(context + " must be an object with exactly one command");
    }

    ShaderPackLoader::ExecutionCommand command;
    if (auto assignIter = commandJson.find(ShaderPackLoader::KEY_ASSIGN); assignIter != commandJson.end()) {
        if (!assignIter->is_object() || assignIter->empty()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_ASSIGN + " must be a non-empty object");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::Assign;
        for (auto valueIter = assignIter->begin(); valueIter != assignIter->end(); ++valueIter) {
            ShaderPackLoader::ExecutionCommand::Assignment assignment;
            if (valueIter.value().is_object()) {
                auto typeIter = valueIter.value().find(ShaderPackLoader::KEY_TYPE);
                if (typeIter != valueIter.value().end()) {
                    if (!typeIter->is_string()) {
                        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_ASSIGN + "." + valueIter.key() +
                                                 "." + ShaderPackLoader::KEY_TYPE + " must be a string");
                    }
                    assignment.type = typeIter->get<std::string>();
                    assignment.explicitType = true;
                }

                auto assignmentValueIter = valueIter.value().find(ShaderPackLoader::KEY_VALUE);
                if (assignmentValueIter == valueIter.value().end()) {
                    throw std::runtime_error(context + "." + ShaderPackLoader::KEY_ASSIGN + "." + valueIter.key() +
                                             "." + ShaderPackLoader::KEY_VALUE + " must be present");
                }

                for (auto fieldIter = valueIter.value().begin(); fieldIter != valueIter.value().end(); ++fieldIter) {
                    if (fieldIter.key() != ShaderPackLoader::KEY_TYPE &&
                        fieldIter.key() != ShaderPackLoader::KEY_VALUE) {
                        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_ASSIGN + "." + valueIter.key() +
                                                 " only supports " + ShaderPackLoader::KEY_TYPE + " and " +
                                                 ShaderPackLoader::KEY_VALUE);
                    }
                }
                assignment.value = *assignmentValueIter;
            } else {
                assignment.type = inferScalarType(valueIter.value(), false);
                assignment.value = valueIter.value();
            }
            command.assignments[valueIter.key()] = std::move(assignment);
        }
        return command;
    }

    if (auto bufferUpdateIter = commandJson.find(ShaderPackLoader::KEY_BUFFER_UPDATE);
        bufferUpdateIter != commandJson.end()) {
        if (!bufferUpdateIter->is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_BUFFER_UPDATE + " must be an object");
        }

        command.type = ShaderPackLoader::ExecutionCommand::Type::BufferUpdate;
        command.bufferUpdate.bufferName = requireString(*bufferUpdateIter, ShaderPackLoader::KEY_BUFFER,
                                                        context + "." + ShaderPackLoader::KEY_BUFFER_UPDATE);
        if (auto offsetIter = bufferUpdateIter->find(ShaderPackLoader::KEY_OFFSET);
            offsetIter != bufferUpdateIter->end()) {
            if (!offsetIter->is_number_unsigned()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_BUFFER_UPDATE + "." +
                                         ShaderPackLoader::KEY_OFFSET + " must be an unsigned integer");
            }
            command.bufferUpdate.offset = offsetIter->get<uint32_t>();
        }

        const json &valuesJson = requireArray(*bufferUpdateIter, ShaderPackLoader::KEY_UINT32_VALUES,
                                              context + "." + ShaderPackLoader::KEY_BUFFER_UPDATE);
        if (valuesJson.empty()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_BUFFER_UPDATE + "." +
                                     ShaderPackLoader::KEY_UINT32_VALUES + " must be non-empty");
        }
        command.bufferUpdate.uint32Values.reserve(valuesJson.size());
        for (size_t i = 0; i < valuesJson.size(); ++i) {
            if (!valuesJson[i].is_number_unsigned()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_BUFFER_UPDATE + "." +
                                         ShaderPackLoader::KEY_UINT32_VALUES + "[" + std::to_string(i) +
                                         "] must be an unsigned integer");
            }
            command.bufferUpdate.uint32Values.push_back(valuesJson[i].get<uint32_t>());
        }

        for (auto fieldIter = bufferUpdateIter->begin(); fieldIter != bufferUpdateIter->end(); ++fieldIter) {
            if (fieldIter.key() != ShaderPackLoader::KEY_BUFFER && fieldIter.key() != ShaderPackLoader::KEY_OFFSET &&
                fieldIter.key() != ShaderPackLoader::KEY_UINT32_VALUES) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_BUFFER_UPDATE + " only supports " +
                                         ShaderPackLoader::KEY_BUFFER + ", " + ShaderPackLoader::KEY_OFFSET + ", and " +
                                         ShaderPackLoader::KEY_UINT32_VALUES);
            }
        }
        return command;
    }

    if (auto passIter = commandJson.find(ShaderPackLoader::KEY_PASS); passIter != commandJson.end()) {
        if (!passIter->is_string()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_PASS + " must be a string");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::Pass;
        command.passName = passIter->get<std::string>();
        return command;
    }

    if (auto sharcUpdateIter = commandJson.find(ShaderPackLoader::KEY_SHARC_UPDATE);
        sharcUpdateIter != commandJson.end()) {
        if (!sharcUpdateIter->is_string()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_SHARC_UPDATE + " must be a string");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::SharcUpdate;
        command.passName = sharcUpdateIter->get<std::string>();
        return command;
    }

    if (auto ifElseIter = commandJson.find(ShaderPackLoader::KEY_IF_ELSE); ifElseIter != commandJson.end()) {
        if (!ifElseIter->is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_IF_ELSE + " must be an object");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::IfElse;
        auto conditionIter = ifElseIter->find(ShaderPackLoader::KEY_CONDITION);
        if (conditionIter == ifElseIter->end()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_IF_ELSE + "." +
                                     ShaderPackLoader::KEY_CONDITION + " must be present");
        }
        command.condition = *conditionIter;
        command.thenCommands = parseExecutionCommands(
            requireArray(*ifElseIter, ShaderPackLoader::KEY_THEN, context + "." + ShaderPackLoader::KEY_IF_ELSE),
            context + "." + ShaderPackLoader::KEY_IF_ELSE + "." + ShaderPackLoader::KEY_THEN);
        auto elseIter = ifElseIter->find(ShaderPackLoader::KEY_ELSE);
        if (elseIter != ifElseIter->end()) {
            command.elseCommands = parseExecutionCommands(
                requireArray(*ifElseIter, ShaderPackLoader::KEY_ELSE, context + "." + ShaderPackLoader::KEY_IF_ELSE),
                context + "." + ShaderPackLoader::KEY_IF_ELSE + "." + ShaderPackLoader::KEY_ELSE);
        }
        return command;
    }

    if (auto whileIter = commandJson.find(ShaderPackLoader::KEY_WHILE); whileIter != commandJson.end()) {
        if (!whileIter->is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_WHILE + " must be an object");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::While;
        auto conditionIter = whileIter->find(ShaderPackLoader::KEY_CONDITION);
        if (conditionIter == whileIter->end()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_WHILE + "." +
                                     ShaderPackLoader::KEY_CONDITION + " must be present");
        }
        command.condition = *conditionIter;
        command.thenCommands = parseExecutionCommands(
            requireArray(*whileIter, ShaderPackLoader::KEY_THEN, context + "." + ShaderPackLoader::KEY_WHILE),
            context + "." + ShaderPackLoader::KEY_WHILE + "." + ShaderPackLoader::KEY_THEN);
        return command;
    }

    throw std::runtime_error(context + " must contain assign, buffer_update, sharc_update, pass, if_else, or while");
}

std::vector<ShaderPackLoader::ExecutionCommand> ShaderPackLoader::parseExecutionCommands(const json &commandsJson,
                                                                                         const std::string &context) {
    if (!commandsJson.is_array()) { throw std::runtime_error(context + " must be an array"); }

    std::vector<ShaderPackLoader::ExecutionCommand> commands;
    commands.reserve(commandsJson.size());
    for (size_t i = 0; i < commandsJson.size(); i++) {
        commands.push_back(parseExecutionCommand(commandsJson[i], context + "[" + std::to_string(i) + "]"));
    }
    return commands;
}

ShaderPackLoader::ExecutionConfig ShaderPackLoader::parseExecutionConfig(const json &configJson,
                                                                         const std::string &context) {
    ShaderPackLoader::ExecutionConfig execution;
    if (!configJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    auto parseVariables = [&](std::vector<ShaderPackLoader::VariableConfig> &variables, std::string_view key,
                              const std::string &variablesContext) {
        auto variablesIter = configJson.find(std::string(key));
        if (variablesIter == configJson.end()) { return; }
        if (!variablesIter->is_array()) { throw std::runtime_error(variablesContext + " must be an array"); }
        variables.reserve(variablesIter->size());
        for (size_t i = 0; i < variablesIter->size(); i++) {
            variables.push_back(
                parseVariableConfig((*variablesIter)[i], variablesContext + "[" + std::to_string(i) + "]"));
        }
    };

    parseVariables(execution.globalVariables, ShaderPackLoader::KEY_GLOBAL_VARIABLES,
                   context + "." + ShaderPackLoader::KEY_GLOBAL_VARIABLES);

    auto commandsIter = configJson.find(ShaderPackLoader::KEY_COMMANDS);
    if (commandsIter != configJson.end()) {
        execution.commands = parseExecutionCommands(*commandsIter, context + "." + ShaderPackLoader::KEY_COMMANDS);
    }

    std::unordered_map<std::string, std::string> names;
    auto checkVariables = [&](const std::vector<ShaderPackLoader::VariableConfig> &variables,
                              const std::string &scope) {
        for (const auto &variable : variables) {
            auto [iter, inserted] = names.emplace(variable.name, scope);
            if (!inserted) {
                throw std::runtime_error("duplicate execution variable " + variable.name + " in " + iter->second +
                                         " and " + scope);
            }
        }
    };
    checkVariables(execution.globalVariables, ShaderPackLoader::KEY_GLOBAL_VARIABLES);

    std::unordered_map<std::string, size_t> variableIndices;
    std::unordered_map<std::string, bool> placeholderTypes;
    execution.variables = execution.globalVariables;
    variableIndices.reserve(execution.variables.size());
    placeholderTypes.reserve(execution.variables.size());
    for (size_t i = 0; i < execution.variables.size(); i++) {
        variableIndices[execution.variables[i].name] = i;
        placeholderTypes[execution.variables[i].name] = false;
    }
    collectExecutionVariables(execution.commands, execution.variables, variableIndices, placeholderTypes);

    return execution;
}

void ShaderPackLoader::validateExecutionCommands(const std::vector<ExecutionCommand> &commands,
                                                 Stage stage,
                                                 const std::optional<SharcConfig> &sharc,
                                                 const std::string &context,
                                                 bool &hasSharcUpdateCommand) {
    for (size_t i = 0; i < commands.size(); ++i) {
        const auto &command = commands[i];
        const std::string commandContext = context + "[" + std::to_string(i) + "]";
        switch (command.type) {
            case ShaderPackLoader::ExecutionCommand::Type::Assign:
            case ShaderPackLoader::ExecutionCommand::Type::BufferUpdate:
            case ShaderPackLoader::ExecutionCommand::Type::Pass: break;
            case ShaderPackLoader::ExecutionCommand::Type::SharcUpdate:
                if (stage != Stage::RayTracing) {
                    throw std::runtime_error(commandContext + "." + ShaderPackLoader::KEY_SHARC_UPDATE +
                                             " is only supported in ray_tracing execution");
                }
                if (!sharc.has_value()) {
                    throw std::runtime_error(commandContext + "." + ShaderPackLoader::KEY_SHARC_UPDATE +
                                             " requires root." + std::string(ShaderPackLoader::KEY_SHARC));
                }
                if (command.passName != sharc->updatePassName) {
                    throw std::runtime_error(commandContext + "." + ShaderPackLoader::KEY_SHARC_UPDATE +
                                             " must reference root." + std::string(ShaderPackLoader::KEY_SHARC) + "." +
                                             std::string(ShaderPackLoader::KEY_UPDATE_PASS));
                }
                hasSharcUpdateCommand = true;
                break;
            case ShaderPackLoader::ExecutionCommand::Type::IfElse:
                validateExecutionCommands(command.thenCommands, stage, sharc,
                                          commandContext + "." + ShaderPackLoader::KEY_IF_ELSE + "." +
                                              ShaderPackLoader::KEY_THEN,
                                          hasSharcUpdateCommand);
                validateExecutionCommands(command.elseCommands, stage, sharc,
                                          commandContext + "." + ShaderPackLoader::KEY_IF_ELSE + "." +
                                              ShaderPackLoader::KEY_ELSE,
                                          hasSharcUpdateCommand);
                break;
            case ShaderPackLoader::ExecutionCommand::Type::While:
                validateExecutionCommands(command.thenCommands, stage, sharc,
                                          commandContext + "." + ShaderPackLoader::KEY_WHILE + "." +
                                              ShaderPackLoader::KEY_THEN,
                                          hasSharcUpdateCommand);
                break;
        }
    }
}

ShaderPackLoader::ShaderPack ShaderPackLoader::parseConfigFile(const fs::path &rootPath, const std::string &language) {
    const fs::path configPath = rootPath / "configs.json";
    if (!fs::exists(configPath)) { throw std::runtime_error("missing shader config: " + configPath.string()); }

    std::ifstream stream(configPath);
    if (!stream.is_open()) { throw std::runtime_error("failed to open shader config: " + configPath.string()); }

    json configJson;
    stream >> configJson;
    if (!configJson.is_object()) { throw std::runtime_error("shader config root must be an object"); }

    ShaderPackLoader::ShaderPack shaderPack;
    shaderPack.rootPath = rootPath;
    shaderPack.requiresEmission = optionalBool(configJson, ShaderPackLoader::KEY_REQUIRES_EMISSION, false, "root");
    if (auto segmentationMaskIter = configJson.find(ShaderPackLoader::KEY_SEGMENTATION_MASK);
        segmentationMaskIter != configJson.end()) {
        if (!segmentationMaskIter->is_object()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_SEGMENTATION_MASK) +
                                     " must be an object");
        }
        shaderPack.segmentationMaskEnabled =
            optionalBool(*segmentationMaskIter, ShaderPackLoader::KEY_ENABLED, false,
                         "root." + std::string(ShaderPackLoader::KEY_SEGMENTATION_MASK));
    }
    shaderPack.nrdMode = ShaderPackLoader::parseNrdMode(
        ShaderPackLoader::requireString(configJson, ShaderPackLoader::KEY_NRD_MODE, "root"),
        "root." + std::string(ShaderPackLoader::KEY_NRD_MODE));
    shaderPack.includeDirectories = {rootPath.string()};
    shaderPack.translations = loadLanguage(rootPath, language);
    auto classifiedHitShaders = classifyHitShaders(collectShaderFiles(rootPath));

    auto attributesIter = configJson.find(ShaderPackLoader::KEY_ATTRIBUTES);
    if (attributesIter != configJson.end()) {
        if (!attributesIter->is_array()) { throw std::runtime_error("root.attributes must be an array"); }
        const json &attributesJson = *attributesIter;
        shaderPack.attributes.resize(attributesJson.size());
        mcvr::parallelFor(attributesJson.size(), [&](size_t i) {
            shaderPack.attributes[i] =
                parseAttributeConfig(attributesJson[i], "root.attributes[" + std::to_string(i) + "]");
        });
    }

    auto texturesIter = configJson.find(ShaderPackLoader::KEY_TEXTURES);
    if (texturesIter != configJson.end()) {
        if (!texturesIter->is_array()) { throw std::runtime_error("root.textures must be an array"); }
        const json &texturesJson = *texturesIter;
        shaderPack.textures.resize(texturesJson.size());
        mcvr::parallelFor(texturesJson.size(), [&](size_t i) {
            shaderPack.textures[i] =
                parseTextureConfig(texturesJson[i], rootPath, "root.textures[" + std::to_string(i) + "]");
        });
    }

    auto buffersIter = configJson.find(ShaderPackLoader::KEY_BUFFERS);
    if (buffersIter != configJson.end()) {
        if (!buffersIter->is_array()) { throw std::runtime_error("root.buffers must be an array"); }
        const json &buffersJson = *buffersIter;
        shaderPack.buffers.resize(buffersJson.size());
        mcvr::parallelFor(buffersJson.size(), [&](size_t i) {
            shaderPack.buffers[i] = parseBufferConfig(buffersJson[i], "root.buffers[" + std::to_string(i) + "]");
        });
    }

    auto passesIter = configJson.find(ShaderPackLoader::KEY_PASSES);
    if (passesIter == configJson.end() || !passesIter->is_array() || passesIter->empty()) {
        throw std::runtime_error("root.passes must be a non-empty array");
    }
    const json &passesJson = *passesIter;
    shaderPack.passes.resize(passesJson.size());
    mcvr::parallelFor(passesJson.size(), [&](size_t i) {
        shaderPack.passes[i] =
            parsePassConfig(passesJson[i], rootPath, classifiedHitShaders, "root.passes[" + std::to_string(i) + "]");
    });

    std::optional<ShaderPackLoader::AccelerationStructureBuildMode> configuredAccelerationStructureBuildMode;
    if (auto modeIter = configJson.find(ShaderPackLoader::KEY_ACCELERATION_STRUCTURE_BUILD_MODE);
        modeIter != configJson.end()) {
        if (!modeIter->is_string()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_ACCELERATION_STRUCTURE_BUILD_MODE) +
                                     " must be a string");
        }
        configuredAccelerationStructureBuildMode = parseAccelerationStructureBuildMode(
            modeIter->get<std::string>(),
            "root." + std::string(ShaderPackLoader::KEY_ACCELERATION_STRUCTURE_BUILD_MODE));
    }
    if (auto syncIter = configJson.find(ShaderPackLoader::KEY_SYNC_ACCELERATION_STRUCTURES);
        syncIter != configJson.end()) {
        if (!syncIter->is_boolean()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_SYNC_ACCELERATION_STRUCTURES) +
                                     " must be a boolean");
        }
        const auto boolMode = syncIter->get<bool>() ? ShaderPackLoader::AccelerationStructureBuildMode::Sync :
                                                      ShaderPackLoader::AccelerationStructureBuildMode::Deferred;
        if (configuredAccelerationStructureBuildMode.has_value() &&
            *configuredAccelerationStructureBuildMode != boolMode) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_SYNC_ACCELERATION_STRUCTURES) +
                                     " conflicts with root." +
                                     std::string(ShaderPackLoader::KEY_ACCELERATION_STRUCTURE_BUILD_MODE));
        }
        configuredAccelerationStructureBuildMode = boolMode;
    }
    shaderPack.accelerationStructureBuildMode =
        configuredAccelerationStructureBuildMode.value_or(inferAccelerationStructureBuildMode(shaderPack));

    auto sharcIter = configJson.find(ShaderPackLoader::KEY_SHARC);
    if (sharcIter != configJson.end()) {
        if (!sharcIter->is_object()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_SHARC) + " must be an object");
        }

        ShaderPackLoader::SharcConfig sharc;
        sharc.updatePassName = requireString(*sharcIter, ShaderPackLoader::KEY_UPDATE_PASS,
                                             "root." + std::string(ShaderPackLoader::KEY_SHARC));
        sharc.resolveCompShaderPath = resolveRelativeFile(
            rootPath,
            requireString(*sharcIter, ShaderPackLoader::KEY_RESOLVE_COMP,
                          "root." + std::string(ShaderPackLoader::KEY_SHARC)),
            "root." + std::string(ShaderPackLoader::KEY_SHARC) + "." + std::string(ShaderPackLoader::KEY_RESOLVE_COMP));

        bool foundUpdatePass = false;
        for (const auto &pass : shaderPack.passes) {
            if (pass.type != ShaderPackLoader::PassConfig::Type::RayTracing) { continue; }
            if (pass.rayTracing.name == sharc.updatePassName) {
                foundUpdatePass = true;
                break;
            }
        }
        if (!foundUpdatePass) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_SHARC) + "." +
                                     std::string(ShaderPackLoader::KEY_UPDATE_PASS) +
                                     " must reference a ray_tracing pass");
        }

        shaderPack.sharc = std::move(sharc);
    }

    std::string rayTracingExecutionCommandsContext = "root.execution.commands";
    auto executionIter = configJson.find(ShaderPackLoader::KEY_EXECUTION);
    if (executionIter != configJson.end()) {
        if (!executionIter->is_object()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_EXECUTION) + " must be an object");
        }

        if (executionIter->contains(ShaderPackLoader::KEY_RAY_TRACING)) {
            if (auto rayTracingIter = executionIter->find(ShaderPackLoader::KEY_RAY_TRACING);
                rayTracingIter != executionIter->end()) {
                shaderPack.rayTracingExecution = parseExecutionConfig(
                    *rayTracingIter, "root.execution." + std::string(ShaderPackLoader::KEY_RAY_TRACING));
                rayTracingExecutionCommandsContext =
                    "root.execution." + std::string(ShaderPackLoader::KEY_RAY_TRACING) + ".commands";
            }
        } else {
            shaderPack.rayTracingExecution = parseExecutionConfig(*executionIter, "root.execution");
        }
    }

    auto executionPostIter = configJson.find(ShaderPackLoader::KEY_EXECUTION_POST);
    if (executionPostIter != configJson.end()) {
        if (!executionPostIter->is_object()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_EXECUTION_POST) +
                                     " must be an object");
        }
        shaderPack.postRenderExecution = parseExecutionConfig(*executionPostIter, "root.execution_post");
    }

    auto appendDefaultExecutionCommands = [&](ShaderPackLoader::ExecutionConfig &execution,
                                              ShaderPackLoader::Stage stage) {
        if (!execution.commands.empty()) { return; }

        for (const auto &pass : shaderPack.passes) {
            if (pass.stage != stage) { continue; }
            execution.commands.push_back({
                .type = ShaderPackLoader::ExecutionCommand::Type::Pass,
                .passName = pass.type == ShaderPackLoader::PassConfig::Type::FullScreen ? pass.fullScreen.name :
                            pass.type == ShaderPackLoader::PassConfig::Type::Render     ? pass.render.name :
                            pass.type == ShaderPackLoader::PassConfig::Type::RayTracing ? pass.rayTracing.name :
                                                                                          pass.compute.name,
            });
        }
    };
    appendDefaultExecutionCommands(shaderPack.rayTracingExecution, ShaderPackLoader::Stage::RayTracing);
    appendDefaultExecutionCommands(shaderPack.postRenderExecution, ShaderPackLoader::Stage::PostRender);
    validateExecutionCommands(shaderPack.rayTracingExecution.commands, ShaderPackLoader::Stage::RayTracing,
                              shaderPack.sharc, rayTracingExecutionCommandsContext,
                              shaderPack.rayTracingExecution.hasSharcUpdateCommand);
    validateExecutionCommands(shaderPack.postRenderExecution.commands, ShaderPackLoader::Stage::PostRender,
                              shaderPack.sharc, "root.execution_post.commands",
                              shaderPack.postRenderExecution.hasSharcUpdateCommand);

    std::unordered_map<uint32_t, std::string> runtimeBindings;
    auto registerRuntimeBinding = [&](uint32_t binding, const std::string &owner) {
        auto [iter, inserted] = runtimeBindings.emplace(binding, owner);
        if (!inserted) {
            throw std::runtime_error("duplicate runtime binding " + std::to_string(binding) + ": " + iter->second +
                                     " and " + owner);
        }
    };
    for (const auto &texture : shaderPack.textures) {
        if (texture.sampledBinding.has_value()) {
            registerRuntimeBinding(*texture.sampledBinding, "texture(sampled):" + texture.name);
        }
        if (texture.storageBinding.has_value()) {
            registerRuntimeBinding(*texture.storageBinding, "texture(storage):" + texture.name);
        }
    }
    for (const auto &buffer : shaderPack.buffers) { registerRuntimeBinding(buffer.binding, "buffer:" + buffer.name); }

    return shaderPack;
}

void ShaderPackLoader::collectExecutionVariables(const std::vector<ExecutionCommand> &commands,
                                                 std::vector<VariableConfig> &variables,
                                                 std::unordered_map<std::string, size_t> &variableIndices,
                                                 std::unordered_map<std::string, bool> &placeholderTypes) {
    for (const auto &command : commands) {
        switch (command.type) {
            case ShaderPackLoader::ExecutionCommand::Type::Assign:
                for (const auto &[name, assignment] : command.assignments) {
                    auto type = inferAssignmentType(assignment);
                    auto variableIter = variableIndices.find(name);
                    if (variableIter == variableIndices.end()) {
                        variables.push_back({
                            .name = name,
                            .type = type.value_or("float"),
                            .defaultValue = defaultValueForType(type.value_or("float")),
                        });
                        variableIndices[name] = variables.size() - 1;
                        placeholderTypes[name] = !type.has_value();
                        continue;
                    }

                    if (!type.has_value()) { continue; }

                    VariableConfig &variable = variables[variableIter->second];
                    bool &placeholderType = placeholderTypes[name];
                    if (placeholderType) {
                        variable.type = *type;
                        variable.defaultValue = defaultValueForType(*type);
                        placeholderType = false;
                        continue;
                    }
                    if (variable.type != *type) {
                        throw std::runtime_error("execution variable type mismatch for " + name + ": " + variable.type +
                                                 " and " + *type);
                    }
                }
                break;
            case ShaderPackLoader::ExecutionCommand::Type::IfElse:
                collectExecutionVariables(command.thenCommands, variables, variableIndices, placeholderTypes);
                collectExecutionVariables(command.elseCommands, variables, variableIndices, placeholderTypes);
                break;
            case ShaderPackLoader::ExecutionCommand::Type::While:
                collectExecutionVariables(command.thenCommands, variables, variableIndices, placeholderTypes);
                break;
            case ShaderPackLoader::ExecutionCommand::Type::SharcUpdate:
            case ShaderPackLoader::ExecutionCommand::Type::BufferUpdate:
            case ShaderPackLoader::ExecutionCommand::Type::Pass: break;
        }
    }
}

std::optional<std::string>
ShaderPackLoader::inferAssignmentType(const ShaderPackLoader::ExecutionCommand::Assignment &assignment) {
    if (assignment.type.has_value()) { return assignment.type; }
    return inferScalarType(assignment.value, false);
}

std::string ShaderPackLoader::defaultValueForType(const std::string &type) {
    if (type == "bool") { return "false"; }
    if (type == "int" || type.rfind("int_range:", 0) == 0 || type.rfind("enum:", 0) == 0) { return "0"; }
    return "0.0";
}

std::string ShaderPackLoader::buildExecutionSource(const std::vector<VariableConfig> &variables,
                                                   uint32_t executionSet) {
    if (variables.empty()) { return ""; }

    auto variableAccessor = [](const VariableConfig &variable) {
        const std::string bufferValue = "rtExecutionVariables." + variable.name + "_";
        if (variable.type == "bool") { return "(" + bufferValue + " != 0.0)"; }
        if (variable.type == "int" || variable.type.rfind("int_range:", 0) == 0 ||
            variable.type.rfind("enum:", 0) == 0) {
            return "int(" + bufferValue + ")";
        }
        return bufferValue;
    };

    std::ostringstream stream;
    (void)executionSet;
    stream << "layout(push_constant, std430) uniform RtExecutionVariables {\n";
    for (const auto &variable : variables) { stream << "    float " << variable.name << "_;\n"; }
    stream << "} rtExecutionVariables;\n";
    for (const auto &variable : variables) {
        stream << "#define " << variable.name << " " << variableAccessor(variable) << "\n";
    }
    return stream.str();
}

std::string ShaderPackLoader::parseValue(const std::string &type, const std::string &currentValue) {
    if (type == "bool") {
        return currentValue == "render_pipeline.true" || currentValue == "true" || currentValue == "1" ? "1" : "0";
    } else if (type.rfind("int_range:", 0) == 0) {
        std::string range = type.substr(10);
        size_t separator = range.find('-', 1);
        if (separator != std::string::npos) {
            try {
                int minValue = std::stoi(range.substr(0, separator));
                int maxValue = std::stoi(range.substr(separator + 1));
                if (minValue > maxValue) { std::swap(minValue, maxValue); }
                int value = std::stoi(currentValue);
                return std::to_string(std::clamp(value, minValue, maxValue));
            } catch (...) {}
        }
    } else if (type.rfind("float_range:", 0) == 0) {
        std::string range = type.substr(12);
        size_t separator = range.find('-', 1);
        if (separator != std::string::npos) {
            try {
                float minValue = std::stof(range.substr(0, separator));
                float maxValue = std::stof(range.substr(separator + 1));
                if (minValue > maxValue) { std::swap(minValue, maxValue); }
                float value = std::stof(currentValue);
                std::ostringstream stream;
                stream << std::clamp(value, minValue, maxValue);
                return stream.str();
            } catch (...) {}
        }
    } else if (type.rfind("enum:", 0) == 0) {
        std::string options = type.substr(5);
        std::stringstream stream(options);
        std::string option;
        int index = 0;
        while (std::getline(stream, option, '-')) {
            if (option == currentValue) { return std::to_string(index); }
            index++;
        }
    }
    return currentValue;
}

std::string ShaderPackLoader::substitutePlaceholder(const std::string &expression, const std::string &value) {
    auto isSymbolChar = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_'; };
    std::string resolved;
    resolved.reserve(expression.size() + value.size());
    for (size_t i = 0; i < expression.size(); i++) {
        if (expression[i] == 'X') {
            bool leftIsSymbol = i > 0 && isSymbolChar(expression[i - 1]);
            bool rightIsSymbol = i + 1 < expression.size() && isSymbolChar(expression[i + 1]);
            if (!leftIsSymbol && !rightIsSymbol) {
                resolved += value;
                continue;
            }
        }
        resolved += expression[i];
    }
    return resolved;
}

void ShaderPackLoader::applyDefinitionExpressions(std::unordered_map<std::string, std::string> &definitions,
                                                  const std::unordered_map<std::string, std::string> &expressions,
                                                  const std::string &value) {
    for (const auto &[name, expression] : expressions) { definitions[name] = substitutePlaceholder(expression, value); }
}

ShaderPackLoader::LoadResult
ShaderPackLoader::load(const fs::path &path, const fs::path &builtInPath, const std::string &language) {
    LoadResult result;

    auto tryLoad = [&](const fs::path &shaderPackPath) -> std::optional<ShaderPackLoader::ShaderPack> {
        std::string error;
        fs::path rootPath = resolveShaderPackRoot(shaderPackPath, error);
        if (rootPath.empty()) {
            result.error = error;
            return std::nullopt;
        }
        try {
            return parseConfigFile(rootPath, language);
        } catch (const std::exception &exception) {
            result.error = exception.what();
            return std::nullopt;
        }
    };

    auto resolvePackPath = [](const fs::path &shaderPackPath) {
        if (shaderPackPath.empty() || shaderPackPath.is_absolute()) { return shaderPackPath; }
        return Renderer::folderPath / shaderPackPath;
    };

    fs::path fallbackPath = resolvePackPath(builtInPath);
    fs::path requestedPath = resolvePackPath(path.empty() ? builtInPath : path);
    if (auto shaderPack = tryLoad(requestedPath); shaderPack.has_value()) {
        result.success = true;
        result.shaderPack = std::move(*shaderPack);
        return result;
    }

    if (!path.empty() && requestedPath != fallbackPath) {
        if (auto fallback = tryLoad(fallbackPath); fallback.has_value()) {
            result.success = true;
            result.shaderPack = std::move(*fallback);
            return result;
        }
    }

    return result;
}

std::unordered_map<std::string, std::string>
ShaderPackLoader::buildAttributeDefinitions(const std::vector<ShaderPackLoader::AttributeConfig> &attributes,
                                            const std::unordered_map<std::string, std::string> &currentValues) {
    std::unordered_map<std::string, std::string> definitions;
    for (const auto &attribute : attributes) {
        auto valueIter = currentValues.find(attribute.name);
        const std::string &currentValue = valueIter == currentValues.end() ? attribute.defaultValue : valueIter->second;
        std::string value = parseValue(attribute.type, currentValue);

        if (attribute.define.direct.has_value()) { definitions[*attribute.define.direct] = value; }

        applyDefinitionExpressions(definitions, attribute.define.expressions, value);

        auto valueDefinesIter = attribute.define.variants.find(currentValue);
        if (valueDefinesIter != attribute.define.variants.end()) {
            applyDefinitionExpressions(definitions, valueDefinesIter->second, value);
        }
    }
    return definitions;
}

#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::shared_ptr<vk::DeviceLocalBuffer> ShaderPack::createPassExecutionBuffer(std::shared_ptr<vk::Device> device,
                                                                             std::shared_ptr<vk::VMA> vma,
                                                                             size_t executionBufferSize) {
    if (executionBufferSize == 0) { return nullptr; }
    return vk::DeviceLocalBuffer::create(vma, device, true, executionBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0,
                                         VMA_MEMORY_USAGE_GPU_ONLY);
}

ShaderPack::ShaderPack(std::shared_ptr<Framework> framework) : framework_(framework) {}

std::filesystem::path ShaderPack::builtInShaderPackPath() {
    return Renderer::folderPath / "shaders/world/ray_tracing/advanced.zip";
}

std::filesystem::path ShaderPack::emissionFallbackShaderPackPath() {
    return Renderer::folderPath / "shaders/world/ray_tracing/vanilla-pt.zip";
}

ShaderPack::BuildConfig ShaderPack::buildConfigFromMainRenderAttributes(const std::vector<std::string> &attributeKVs) {
    BuildConfig config;
    for (size_t i = 0; i + 1 < attributeKVs.size(); i += 2) {
        const std::string &key = attributeKVs[i];
        const std::string &value = attributeKVs[i + 1];
        config.staticAttributes[key] = value;
        if (key == "render_pipeline.module.main_render.attribute.shader_pack_path") {
            config.shaderPackPath = value;
        } else if (key == "render_pipeline.module.main_render.attribute.use_sharc") {
            config.shouldUseSharc = value == "render_pipeline.true";
        }
    }
    return config;
}

bool ShaderPack::initialize(const BuildConfig &config, std::string &error) {
    ShaderPackLoader::LoadResult loadResult = ShaderPackLoader::load(
        config.shaderPackPath.empty() ? std::filesystem::path{} : std::filesystem::path(config.shaderPackPath),
        builtInShaderPackPath(), config.language);
    if (!loadResult.success) {
        error = loadResult.error;
        return false;
    }
    if (loadResult.shaderPack.requiresEmission && !Renderer::options.collectChunkEmission) {
        loadResult =
            ShaderPackLoader::load(emissionFallbackShaderPackPath(), emissionFallbackShaderPackPath(), config.language);
        if (!loadResult.success) {
            error = loadResult.error;
            return false;
        }
    }

    shaderPack_ = std::move(loadResult.shaderPack);
    staticAttributeValues_ = config.staticAttributes;
    shaderAttributes_ = ShaderPackLoader::buildAttributeDefinitions(shaderPack_.attributes, config.staticAttributes);
    const auto nrdEnabledIter = config.injectedDefinitions.find("MCVR_USE_NRD");
    const bool nrdEnabled = nrdEnabledIter != config.injectedDefinitions.end() &&
                            (nrdEnabledIter->second == "1" || nrdEnabledIter->second == "true");
    const bool disableHalfRateWithoutNrd =
        !nrdEnabled && shaderPack_.nrdMode == ShaderPackLoader::NrdMode::HalfRateCheckerboard;
    const uint32_t nrdModeDefinition = disableHalfRateWithoutNrd                                              ? 1u :
                                       shaderPack_.nrdMode == ShaderPackLoader::NrdMode::Checkerboard         ? 1u :
                                       shaderPack_.nrdMode == ShaderPackLoader::NrdMode::HalfRateCheckerboard ? 2u :
                                                                                                                0u;
    injectedExpressionVariables_.clear();
    injectedExpressionVariables_.reserve(config.injectedDefinitions.size() + 1u);
    injectedExpressionVariables_.push_back({
        .name = "ADV_NRD_MODE",
        .value = static_cast<double>(nrdModeDefinition),
    });

    auto hasExecutionVariable = [&](std::string_view name) {
        auto stageHasVariable = [&](const ShaderPackLoader::ExecutionConfig &execution) {
            return std::any_of(execution.variables.begin(), execution.variables.end(),
                               [&](const ShaderPackLoader::VariableConfig &variable) { return variable.name == name; });
        };
        return stageHasVariable(shaderPack_.rayTracingExecution) || stageHasVariable(shaderPack_.postRenderExecution);
    };

    for (const auto &[name, value] : config.injectedDefinitions) {
        if (hasExecutionVariable(name)) {
            error = "injected definition conflicts with execution variable: " + name;
            return false;
        }
        shaderAttributes_[name] = value;

        if (name == "ADV_NRD_MODE") { continue; }
        if (!ExpressionEvaluator::isValidIdentifier(name) || ExpressionEvaluator::isReservedIdentifier(name)) {
            continue;
        }

        const std::string numericValue = trimString(value);
        size_t parsed = 0;
        try {
            const double number = std::stod(numericValue, &parsed);
            if (parsed == numericValue.size() && std::isfinite(number)) {
                injectedExpressionVariables_.push_back({
                    .name = name,
                    .value = number,
                });
            }
        } catch (...) {}
    }

    shaderAttributes_["ADV_NRD_MODE"] = std::to_string(nrdModeDefinition);
    {
        std::lock_guard<std::mutex> lock(numericExpressionCacheMutex_);
        numericExpressionCache_.clear();
    }

    hasSharcRuntime_ = false;
    if (config.shouldUseSharc && shaderPack_.sharc.has_value()) { hasSharcRuntime_ = true; }

    initStageRuntime(rayTracingStageRuntime_, shaderPack_.rayTracingExecution);
    initStageRuntime(postRenderStageRuntime_, shaderPack_.postRenderExecution);
    activeRuntimeResources_ = collectActiveRuntimeResources();

    return true;
}

bool ShaderPack::ActiveRuntimeResources::hasTexture(const std::string &name) const {
    return textures.find(name) != textures.end();
}

bool ShaderPack::ActiveRuntimeResources::hasBuffer(const std::string &name) const {
    return buffers.find(name) != buffers.end();
}

void ShaderPack::ensureRuntimeResources(uint32_t referenceWidth, uint32_t referenceHeight) {
    if (runtimeResourcesReady_ && referenceWidth_ == referenceWidth && referenceHeight_ == referenceHeight) {
        refreshRuntimeBuffers();
        return;
    }

    runtimeTextures_.clear();
    runtimeTextureIndices_.clear();
    runtimeBuffers_.clear();
    runtimeBufferIndices_.clear();

    referenceWidth_ = referenceWidth;
    referenceHeight_ = referenceHeight;

    initRuntimeTextures();
    initRuntimeBuffers();
    loadRuntimeResources();
    runtimeResourcesReady_ = true;
}

bool ShaderPack::refreshRuntimeBuffers() {
    if (!runtimeResourcesReady_) { return false; }
    if (runtimeBuffers_.empty()) {
        runtimeResourceExpressionsDirty_ = false;
        return false;
    }
    if (!runtimeResourceExpressionsDirty_) { return false; }
    runtimeResourceExpressionsDirty_ = false;

    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    auto &frr = framework->frameResourceRetainer();
    uint32_t frameCount = framework->swapchain()->imageCount();
    bool resizedAnyBuffer = false;

    for (auto &runtimeBuffer : runtimeBuffers_) {
        if (runtimeBuffer.frameBuffers.empty()) { continue; }
        const size_t expectedSize =
            std::max(static_cast<size_t>(1),
                     static_cast<size_t>(std::ceil(evaluateNumericExpression(runtimeBuffer.config.sizeExpression))));

        bool needsResize = runtimeBuffer.frameBuffers.empty();
        if (!needsResize) {
            for (const auto &buffer : runtimeBuffer.frameBuffers) {
                if (buffer == nullptr || buffer->size() != expectedSize) {
                    needsResize = true;
                    break;
                }
            }
        }
        if (!needsResize) { continue; }

        const uint32_t bufferFrameCount = runtimeBuffer.config.shared ? 1 : frameCount;
        runtimeBuffer.frameBuffers.resize(bufferFrameCount);
        for (uint32_t frameIndex = 0; frameIndex < bufferFrameCount; ++frameIndex) {
            frr.retain(runtimeBuffer.frameBuffers[frameIndex]);
            runtimeBuffer.frameBuffers[frameIndex] =
                vk::DeviceLocalBuffer::create(vma, device, false, expectedSize,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                              0, VMA_MEMORY_USAGE_GPU_ONLY);
        }
        resizedAnyBuffer = true;
    }
    return resizedAnyBuffer;
}

void ShaderPack::setRuntimeResourceExpressionVariables(std::vector<ExpressionEvaluator::Variable> variables) {
    bool variableNamesChanged = variables.size() != runtimeResourceExpressionVariables_.size();
    bool variableValuesChanged = variableNamesChanged;
    if (!variableNamesChanged) {
        for (size_t i = 0; i < variables.size(); ++i) {
            if (variables[i].name != runtimeResourceExpressionVariables_[i].name) {
                variableNamesChanged = true;
                variableValuesChanged = true;
                break;
            }
            if (variables[i].value != runtimeResourceExpressionVariables_[i].value) { variableValuesChanged = true; }
        }
    }
    runtimeResourceExpressionVariables_ = std::move(variables);
    if (variableValuesChanged) { runtimeResourceExpressionsDirty_ = true; }
    if (variableNamesChanged) {
        std::lock_guard<std::mutex> lock(numericExpressionCacheMutex_);
        numericExpressionCache_.clear();
    }
}

void ShaderPack::bindRuntimeResources(const std::shared_ptr<vk::DescriptorTable> &descriptorTable,
                                      uint32_t setIndex,
                                      uint32_t frameIndex) {
    if (!runtimeResourcesReady_ || descriptorTable == nullptr) { return; }

    for (const auto &texture : runtimeTextures_) {
        if (!texture.config.imported && texture.frameImages.empty()) { continue; }
        if (texture.config.sampledBinding.has_value()) {
            std::shared_ptr<vk::DeviceLocalImage> image =
                texture.config.imported ? texture.importedImage :
                                          texture.frameImages[texture.config.shared ? 0 : frameIndex];
            VkImageLayout layout = texture.config.imported || !texture.config.storageBinding.has_value() ?
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL :
                                       VK_IMAGE_LAYOUT_GENERAL;
            descriptorTable->bindSamplerImage(texture.sampler, image, layout, setIndex, *texture.config.sampledBinding,
                                              0, texture.sampledViewIndex);
        }
        if (texture.config.storageBinding.has_value() && !texture.config.imported) {
            descriptorTable->bindImage(texture.frameImages[texture.config.shared ? 0 : frameIndex],
                                       VK_IMAGE_LAYOUT_GENERAL, setIndex, *texture.config.storageBinding);
        }
    }
    for (const auto &buffer : runtimeBuffers_) {
        if (buffer.frameBuffers.empty()) { continue; }
        descriptorTable->bindBuffer(buffer.frameBuffers[buffer.config.shared ? 0 : frameIndex], setIndex,
                                    buffer.config.binding);
    }
}

void ShaderPack::defineRuntimeResourceDescriptorSet(vk::DescriptorTableBuilder &builder,
                                                    VkShaderStageFlags sampledImageStageFlags,
                                                    VkShaderStageFlags storageImageStageFlags,
                                                    VkShaderStageFlags storageBufferStageFlags) const {
    auto &set = builder.beginDescriptorLayoutSet();
    auto &bindings = set.beginDescriptorLayoutSetBinding();
    if (!hasRuntimeResources()) {
        bindings.endDescriptorLayoutSetBinding();
        set.endDescriptorLayoutSet();
        return;
    }
    for (const auto &texture : shaderPack_.textures) {
        if (texture.sampledBinding.has_value()) {
            bindings.defineDescriptorLayoutSetBinding({
                .binding = *texture.sampledBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = sampledImageStageFlags,
            });
        }
        if (texture.storageBinding.has_value()) {
            bindings.defineDescriptorLayoutSetBinding({
                .binding = *texture.storageBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = storageImageStageFlags,
            });
        }
    }
    for (const auto &buffer : shaderPack_.buffers) {
        bindings.defineDescriptorLayoutSetBinding({
            .binding = buffer.binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = storageBufferStageFlags,
        });
    }
    bindings.endDescriptorLayoutSetBinding();
    set.endDescriptorLayoutSet();
}

void ShaderPack::defineExecutionDescriptorSet(vk::DescriptorTableBuilder &builder,
                                              ShaderPackLoader::Stage stage,
                                              VkShaderStageFlags stageFlags) const {
    if (execution(stage).variables.empty()) { return; }

    const uint32_t size = static_cast<uint32_t>(execution(stage).variables.size() * sizeof(float));
    builder.definePushConstant({
        .stageFlags = stageFlags,
        .offset = 0,
        .size = size,
    });
}

void ShaderPack::copyStageExecutionState(
    ShaderPackLoader::Stage stage,
    std::unordered_map<std::string, ShaderPackLoader::VariableConfig> &variableConfigs,
    std::unordered_map<std::string, std::string> &globalVariables) const {
    const auto &runtime = stageRuntime(stage);
    variableConfigs = runtime.executionVariableConfigs;
    globalVariables = runtime.globalVariables;
}

uint32_t ShaderPack::executionSet(uint32_t runtimeResourceSet) const {
    return runtimeResourceSet + (hasRuntimeResources() ? 1u : 0u);
}

bool ShaderPack::hasRuntimeResources() const {
    return !shaderPack_.textures.empty() || !shaderPack_.buffers.empty();
}

void ShaderPack::preClose() {
    {
        std::lock_guard<std::mutex> lock(numericExpressionCacheMutex_);
        numericExpressionCache_.clear();
    }
    runtimeTextures_.clear();
    runtimeTextureIndices_.clear();
    runtimeBuffers_.clear();
    runtimeBufferIndices_.clear();
    runtimeResourcesReady_ = false;
    referenceWidth_ = 0;
    referenceHeight_ = 0;
}

std::shared_ptr<vk::Shader> ShaderPack::createShader(std::shared_ptr<vk::Device> device,
                                                     const std::filesystem::path &path,
                                                     VkShaderStageFlagBits stage,
                                                     const std::unordered_map<std::string, std::string> &definitions,
                                                     ShaderPackLoader::Stage executionStage,
                                                     uint32_t executionSet) const {
    const std::filesystem::path cacheDir = Renderer::folderPath / "cache/shaders";
    auto compileResult = vk::Shader::compileGlslToSpv(
        path.string(), stage, mergeDefinitions(shaderAttributes_, definitions), shaderPack_.includeDirectories,
        executionSource(executionStage, executionSet), cacheDir);
    return vk::Shader::create(device, std::move(compileResult));
}

bool ShaderPack::isShaderIdentifierStart(char ch) {
    unsigned char uch = static_cast<unsigned char>(ch);
    return std::isalpha(uch) != 0 || ch == '_';
}

bool ShaderPack::isShaderIdentifierContinue(char ch) {
    unsigned char uch = static_cast<unsigned char>(ch);
    return std::isalnum(uch) != 0 || ch == '_';
}

std::string ShaderPack::stripShaderComments(std::string_view line, bool &insideBlockComment) {
    std::string stripped;
    stripped.reserve(line.size());
    for (size_t i = 0; i < line.size();) {
        if (insideBlockComment) {
            if (i + 1 < line.size() && line[i] == '*' && line[i + 1] == '/') {
                insideBlockComment = false;
                i += 2;
            } else {
                i++;
            }
            continue;
        }
        if (i + 1 < line.size() && line[i] == '/' && line[i + 1] == '*') {
            insideBlockComment = true;
            i += 2;
            continue;
        }
        if (i + 1 < line.size() && line[i] == '/' && line[i + 1] == '/') { break; }
        stripped.push_back(line[i]);
        i++;
    }
    return stripped;
}

void ShaderPack::collectShaderAttributeIdentifiers(std::string_view source,
                                                   const std::unordered_set<std::string> &attributeKeys,
                                                   std::set<std::string> &usedAttributes) {
    for (size_t i = 0; i < source.size();) {
        if (!isShaderIdentifierStart(source[i])) {
            i++;
            continue;
        }
        size_t start = i++;
        while (i < source.size() && isShaderIdentifierContinue(source[i])) { i++; }
        std::string identifier(source.substr(start, i - start));
        if (attributeKeys.find(identifier) != attributeKeys.end()) { usedAttributes.insert(std::move(identifier)); }
    }
}

std::optional<std::string> ShaderPack::parseShaderInclude(std::string_view line) {
    size_t i = 0;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
    if (i >= line.size() || line[i] != '#') { return std::nullopt; }
    i++;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
    size_t directiveStart = i;
    while (i < line.size() && isShaderIdentifierContinue(line[i])) { i++; }
    if (line.substr(directiveStart, i - directiveStart) != "include") { return std::nullopt; }
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
    if (i >= line.size() || line[i] != '"') { return std::nullopt; }
    i++;
    size_t pathStart = i;
    while (i < line.size() && line[i] != '"') { i++; }
    if (i >= line.size()) { return std::nullopt; }
    return std::string(line.substr(pathStart, i - pathStart));
}

std::optional<std::filesystem::path>
ShaderPack::resolveShaderInclude(const std::filesystem::path &requestingPath,
                                 std::string_view includePath,
                                 const std::vector<std::filesystem::path> &includeDirectories) {
    std::error_code ec;
    std::filesystem::path relativeCandidate = requestingPath.parent_path() / std::filesystem::path(includePath);
    if (std::filesystem::exists(relativeCandidate, ec) && std::filesystem::is_regular_file(relativeCandidate, ec)) {
        return std::filesystem::weakly_canonical(relativeCandidate, ec);
    }
    for (const auto &includeDirectory : includeDirectories) {
        std::filesystem::path candidate = includeDirectory / std::filesystem::path(includePath);
        if (std::filesystem::exists(candidate, ec) && std::filesystem::is_regular_file(candidate, ec)) {
            return std::filesystem::weakly_canonical(candidate, ec);
        }
    }
    return std::nullopt;
}

std::optional<std::set<std::string>> ShaderPack::collectShaderAttributeDependencies(
    const std::filesystem::path &shaderPath,
    const std::unordered_set<std::string> &attributeKeys,
    const std::vector<std::filesystem::path> &includeDirectories,
    std::unordered_map<std::string, std::optional<std::set<std::string>>> &dependencyCache,
    std::unordered_set<std::string> &activeFiles) {
    std::error_code ec;
    std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(shaderPath, ec);
    if (ec) { canonicalPath = shaderPath.lexically_normal(); }
    const std::string cacheKey = canonicalPath.string();

    if (auto iter = dependencyCache.find(cacheKey); iter != dependencyCache.end()) { return iter->second; }
    if (!activeFiles.insert(cacheKey).second) { return std::set<std::string>{}; }

    std::ifstream file(canonicalPath);
    if (!file.is_open()) {
        activeFiles.erase(cacheKey);
        dependencyCache[cacheKey] = std::nullopt;
        return std::nullopt;
    }

    std::set<std::string> usedAttributes;
    std::string line;
    bool insideBlockComment = false;
    while (std::getline(file, line)) {
        std::string stripped = stripShaderComments(line, insideBlockComment);
        collectShaderAttributeIdentifiers(stripped, attributeKeys, usedAttributes);

        auto includePath = parseShaderInclude(stripped);
        if (!includePath.has_value()) { continue; }
        auto resolvedInclude = resolveShaderInclude(canonicalPath, *includePath, includeDirectories);
        if (!resolvedInclude.has_value()) {
            activeFiles.erase(cacheKey);
            dependencyCache[cacheKey] = std::nullopt;
            return std::nullopt;
        }
        auto includeUsedAttributes = collectShaderAttributeDependencies(
            *resolvedInclude, attributeKeys, includeDirectories, dependencyCache, activeFiles);
        if (!includeUsedAttributes.has_value()) {
            activeFiles.erase(cacheKey);
            dependencyCache[cacheKey] = std::nullopt;
            return std::nullopt;
        }
        usedAttributes.insert(includeUsedAttributes->begin(), includeUsedAttributes->end());
    }

    activeFiles.erase(cacheKey);
    dependencyCache[cacheKey] = usedAttributes;
    return usedAttributes;
}

std::unordered_map<std::string, std::string> ShaderPack::filterShaderAttributes(
    const std::unordered_map<std::string, std::string> &allAttributes,
    const std::filesystem::path &shaderPath,
    const std::unordered_set<std::string> &attributeKeys,
    const std::vector<std::filesystem::path> &includeDirectories,
    std::unordered_map<std::string, std::optional<std::set<std::string>>> &dependencyCache) {
    std::unordered_set<std::string> activeFiles;
    auto usedAttributes =
        collectShaderAttributeDependencies(shaderPath, attributeKeys, includeDirectories, dependencyCache, activeFiles);
    if (!usedAttributes.has_value()) { return allAttributes; }

    std::unordered_map<std::string, std::string> filtered;
    for (const auto &[attributeName, attributeValue] : allAttributes) {
        if (usedAttributes->find(attributeName) != usedAttributes->end()) {
            filtered.emplace(attributeName, attributeValue);
        }
    }
    return filtered;
}

std::unordered_map<std::string, std::string>
ShaderPack::stripAttributeDefinitions(const std::unordered_map<std::string, std::string> &definitions,
                                      const std::unordered_set<std::string> &attributeKeys) {
    std::unordered_map<std::string, std::string> stripped;
    for (const auto &[key, value] : definitions) {
        if (attributeKeys.find(key) == attributeKeys.end()) { stripped.emplace(key, value); }
    }
    return stripped;
}

std::vector<std::shared_ptr<vk::Shader>> ShaderPack::createShaders(std::shared_ptr<vk::Device> device,
                                                                   const std::vector<ShaderCreateInfo> &requests
#ifdef DEBUG
                                                                   ,
                                                                   ShaderBatchStats *stats
#endif
) const {
    if (requests.empty()) { return {}; }

    const std::filesystem::path cacheDir = Renderer::folderPath / "cache/shaders";
    std::vector<std::filesystem::path> includeDirectories;
    includeDirectories.reserve(shaderPack_.includeDirectories.size());
    for (const auto &includeDirectory : shaderPack_.includeDirectories) {
        includeDirectories.emplace_back(includeDirectory);
    }
    std::unordered_set<std::string> attributeKeys;
    attributeKeys.reserve(shaderAttributes_.size());
    for (const auto &[name, value] : shaderAttributes_) {
        (void)value;
        attributeKeys.insert(name);
    }
    std::unordered_map<std::string, std::optional<std::set<std::string>>> dependencyCache;

    const auto buildKey = [&](const ShaderCreateInfo &r, const std::unordered_map<std::string, std::string> &attrDefs,
                              const std::string &execSrc) {
        std::ostringstream key;
        key << r.path.string() << '|' << static_cast<int>(r.stage) << '|';
        auto merged = mergeDefinitions(attrDefs, r.definitions);
        std::vector<std::pair<std::string, std::string>> sortedDefinitions(merged.begin(), merged.end());
        std::sort(sortedDefinitions.begin(), sortedDefinitions.end());
        for (const auto &[k, v] : sortedDefinitions) key << k << '=' << v << ';';
        key << '|' << static_cast<int>(r.executionStage) << '|' << r.executionSet << '|' << execSrc;
        return key.str();
    };

    std::vector<size_t> uniqueIndices;
    std::vector<size_t> requestToUniqueIndex(requests.size());
    std::unordered_map<std::string, size_t> keyToIndex;
    std::vector<std::unordered_map<std::string, std::string>> filteredRequestAttributes(requests.size());
    std::vector<std::unordered_map<std::string, std::string>> requestDefinitions(requests.size());
    std::vector<std::string> executionSources(requests.size());

    for (size_t i = 0; i < requests.size(); i++) {
        filteredRequestAttributes[i] = filterShaderAttributes(shaderAttributes_, requests[i].path, attributeKeys,
                                                              includeDirectories, dependencyCache);
        requestDefinitions[i] = stripAttributeDefinitions(requests[i].definitions, attributeKeys);
        executionSources[i] = executionSource(requests[i].executionStage, requests[i].executionSet);
        ShaderCreateInfo request = requests[i];
        request.definitions = requestDefinitions[i];
        std::string key = buildKey(request, filteredRequestAttributes[i], executionSources[i]);
        auto [iter, inserted] = keyToIndex.emplace(std::move(key), uniqueIndices.size());
        if (inserted) { uniqueIndices.push_back(i); }
        requestToUniqueIndex[i] = iter->second;
    }

    std::vector<vk::Shader::CompileResult> compileResults(uniqueIndices.size());
    mcvr::parallelFor(uniqueIndices.size(), [&](size_t ui) {
        const size_t requestIndex = uniqueIndices[ui];
        const auto &request = requests[requestIndex];
        compileResults[ui] = vk::Shader::compileGlslToSpv(
            request.path.string(), request.stage,
            mergeDefinitions(filteredRequestAttributes[requestIndex], requestDefinitions[requestIndex]),
            shaderPack_.includeDirectories, executionSources[requestIndex], cacheDir);
    });

#ifdef DEBUG
    if (stats != nullptr) {
        stats->requestCount = requests.size();
        stats->uniqueShaderCount = uniqueIndices.size();
        for (const auto &compileResult : compileResults) {
            if (compileResult.cacheHit) {
                stats->cacheHitCount++;
            } else {
                stats->cacheMissCount++;
            }
            if (compileResult.cacheReadFailed) { stats->cacheReadFailureCount++; }
        }
    }
#endif

    std::vector<std::shared_ptr<vk::Shader>> shaders(requests.size());
    mcvr::parallelFor(requests.size(), [&](size_t i) {
        shaders[i] = vk::Shader::create(device, compileResults[requestToUniqueIndex[i]].clone());
    });
    return shaders;
}

const ShaderPackLoader::ShaderPack &ShaderPack::shaderPack() const {
    return shaderPack_;
}

bool ShaderPack::hasSegmentationMask() const {
    return shaderPack_.segmentationMaskEnabled;
}

ShaderPackLoader::AccelerationStructureBuildMode ShaderPack::accelerationStructureBuildMode() const {
    return shaderPack_.accelerationStructureBuildMode;
}

ShaderPackLoader::NrdMode ShaderPack::nrdMode() const {
    return shaderPack_.nrdMode;
}

const std::unordered_map<std::string, std::string> &ShaderPack::shaderAttributes() const {
    return shaderAttributes_;
}

const ShaderPack::StageRuntime &ShaderPack::stageRuntime(ShaderPackLoader::Stage stage) const {
    return stage == ShaderPackLoader::Stage::PostRender ? postRenderStageRuntime_ : rayTracingStageRuntime_;
}

const ShaderPackLoader::ExecutionConfig &ShaderPack::execution(ShaderPackLoader::Stage stage) const {
    return stage == ShaderPackLoader::Stage::PostRender ? shaderPack_.postRenderExecution :
                                                          shaderPack_.rayTracingExecution;
}

std::string ShaderPack::executionSource(ShaderPackLoader::Stage stage, uint32_t executionSet) const {
    return ShaderPackLoader::buildExecutionSource(execution(stage).variables, executionSet);
}

double
ShaderPack::evaluateNumericExpression(ShaderPackLoader::Stage stage,
                                      const std::string &expression,
                                      const ExecutionVariables &variables,
                                      const std::vector<ExpressionEvaluator::Variable> &runtimeExpressionVariables,
                                      bool includeAttributeDefines) const {
    auto parseBoolValue = [](const std::string &value) {
        return value == "true" || value == "1" || value == "render_pipeline.true";
    };
    auto parseNumber = [&](const ShaderPackLoader::VariableConfig &config, const std::string &value) {
        if (config.type == "bool") { return parseBoolValue(value) ? 1.0 : 0.0; }
        if (config.type.rfind("enum:", 0) == 0) {
            std::string options = config.type.substr(5);
            std::stringstream stream(options);
            std::string option;
            int index = 0;
            while (std::getline(stream, option, '-')) {
                if (option == value) { return static_cast<double>(index); }
                index++;
            }
        }
        return std::stod(value);
    };

    std::vector<ExpressionEvaluator::Variable> expressionVariables;
    expressionVariables.reserve(variables.size() + shaderPack_.attributes.size() + runtimeExpressionVariables.size());
    std::unordered_map<std::string, size_t> variableIndices;
    variableIndices.reserve(expressionVariables.capacity());
    auto appendExpressionVariable = [&](const std::string &name, double value) {
        if (variableIndices.find(name) != variableIndices.end()) { return; }
        variableIndices[name] = expressionVariables.size();
        expressionVariables.push_back({
            .name = name,
            .value = value,
        });
    };
    auto appendDynamicVariable = [&](const std::string &name, const std::string &value, const std::string &type) {
        if (type == "bool") {
            appendExpressionVariable(name, parseBoolValue(value) ? 1.0 : 0.0);
            return;
        }
        if (type.rfind("enum:", 0) == 0) {
            appendExpressionVariable(name, parseNumber(
                                               {
                                                   .name = name,
                                                   .type = type,
                                                   .defaultValue = value,
                                               },
                                               value));
            return;
        }

        size_t parsed = 0;
        try {
            double numericValue = std::stod(value, &parsed);
            if (parsed == value.size()) { appendExpressionVariable(name, numericValue); }
        } catch (...) {}
    };

    for (const auto &[name, variable] : variables) {
        appendDynamicVariable(variable.name, variable.value, variable.type);
    }

    if (includeAttributeDefines) {
        for (const auto &attribute : shaderPack_.attributes) {
            const std::string &currentValue = [&]() -> const std::string & {
                auto iter = staticAttributeValues_.find(attribute.name);
                return iter == staticAttributeValues_.end() ? attribute.defaultValue : iter->second;
            }();

            if (attribute.define.direct.has_value()) {
                if (attribute.type == "bool") {
                    appendExpressionVariable(*attribute.define.direct, parseBoolValue(currentValue) ? 1.0 : 0.0);
                } else if (attribute.type == "int" || attribute.type.rfind("int_range:", 0) == 0 ||
                           attribute.type.rfind("float_range:", 0) == 0 || attribute.type.rfind("enum:", 0) == 0) {
                    appendExpressionVariable(*attribute.define.direct, parseNumber(
                                                                           {
                                                                               .name = attribute.name,
                                                                               .type = attribute.type,
                                                                               .defaultValue = currentValue,
                                                                           },
                                                                           currentValue));
                }
            }

            for (const auto &[name, expression] : attribute.define.expressions) {
                auto value = tryEvaluateScalarDefineExpression(attribute, currentValue, expression);
                if (value.has_value()) { appendExpressionVariable(name, *value); }
            }
        }
    }

    for (const auto &variable : runtimeExpressionVariables) { appendExpressionVariable(variable.name, variable.value); }

    return evaluateCachedNumericExpression(stage, expression, expressionVariables, includeAttributeDefines);
}

double ShaderPack::evaluateNumericExpression(ShaderPackLoader::Stage stage,
                                             const std::string &expression,
                                             const std::vector<ExpressionEvaluator::Variable> &expressionVariables,
                                             bool includeAttributeDefines) const {
    return evaluateCachedNumericExpression(stage, expression, expressionVariables, includeAttributeDefines);
}

std::vector<ExpressionEvaluator::Variable> ShaderPack::staticExpressionVariables(bool includeAttributeDefines) const {
    std::vector<ExpressionEvaluator::Variable> expressionVariables;
    expressionVariables.reserve(injectedExpressionVariables_.size() +
                                (includeAttributeDefines ? shaderPack_.attributes.size() : 0));
    std::unordered_set<std::string> variableNames;
    variableNames.reserve(expressionVariables.capacity());
    auto appendExpressionVariable = [&](const std::string &name, double value) {
        if (!variableNames.emplace(name).second) { return; }
        expressionVariables.push_back({
            .name = name,
            .value = value,
        });
    };

    for (const auto &variable : injectedExpressionVariables_) {
        appendExpressionVariable(variable.name, variable.value);
    }
    if (!includeAttributeDefines) { return expressionVariables; }

    for (const auto &attribute : shaderPack_.attributes) {
        const std::string &currentValue = [&]() -> const std::string & {
            auto iter = staticAttributeValues_.find(attribute.name);
            return iter == staticAttributeValues_.end() ? attribute.defaultValue : iter->second;
        }();

        if (attribute.define.direct.has_value()) {
            if (attribute.type == "bool" || attribute.type == "int" || attribute.type.rfind("int_range:", 0) == 0 ||
                attribute.type.rfind("float_range:", 0) == 0 || attribute.type.rfind("enum:", 0) == 0) {
                appendExpressionVariable(*attribute.define.direct,
                                         parseExecutionNumber(attribute.name, attribute.type, currentValue));
            }
        }

        for (const auto &[name, expression] : attribute.define.expressions) {
            auto value = tryEvaluateScalarDefineExpression(attribute, currentValue, expression);
            if (value.has_value()) { appendExpressionVariable(name, *value); }
        }
    }

    return expressionVariables;
}

std::vector<std::string> ShaderPack::expressionVariableNames(ShaderPackLoader::Stage stage,
                                                             bool includeAttributeDefines) const {
    std::vector<std::string> names;
    std::unordered_set<std::string> seen;
    auto appendName = [&](const std::string &name) {
        if (name.empty() || !seen.emplace(name).second) { return; }
        names.push_back(name);
    };

    for (const auto &variable : execution(stage).variables) { appendName(variable.name); }
    for (const auto &variable : runtimeResourceExpressionVariables_) { appendName(variable.name); }
    for (const auto &variable : staticExpressionVariables(includeAttributeDefines)) { appendName(variable.name); }

    appendName("RENDER_WIDTH");
    appendName("RENDER_HEIGHT");
    appendName("CHUNK_SIZE_X");
    appendName("CHUNK_SIZE_Y");
    appendName("CHUNK_SIZE_Z");
    appendName("CHUNK_BOTTOM_SECTION_COORD");
    appendName("CHUNK_NUM");

    return names;
}

double
ShaderPack::evaluateCachedNumericExpression(ShaderPackLoader::Stage stage,
                                            const std::string &expression,
                                            const std::vector<ExpressionEvaluator::Variable> &expressionVariables,
                                            bool includeAttributeDefines) const {
    std::string cacheKey;
    cacheKey.reserve(expression.size() + 8);
    cacheKey += stage == ShaderPackLoader::Stage::PostRender ? "post:" : "rt:";
    cacheKey += includeAttributeDefines ? "attr:" : "noattr:";
    cacheKey += expression;

    std::lock_guard<std::mutex> lock(numericExpressionCacheMutex_);
    auto iter = numericExpressionCache_.find(cacheKey);
    if (iter == numericExpressionCache_.end()) {
        iter = numericExpressionCache_
                   .try_emplace(cacheKey, expression, expressionVariableNames(stage, includeAttributeDefines),
                                staticExpressionVariables(includeAttributeDefines))
                   .first;
    }
    return iter->second.evaluate(expressionVariables);
}

void ShaderPack::uploadExecutionBuffer(ShaderPackLoader::Stage stage,
                                       const std::shared_ptr<vk::DeviceLocalBuffer> &executionBuffer,
                                       const ExecutionVariables &variables,
                                       const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                                       const std::shared_ptr<vk::DescriptorTable> &descriptorTable,
                                       uint32_t executionSet,
                                       uint32_t queueIndex,
                                       VkShaderStageFlags shaderStages) const {
    const auto &execution = this->execution(stage);
    if (executionBuffer == nullptr || execution.variables.empty()) { return; }
    (void)executionBuffer;
    (void)executionSet;
    (void)queueIndex;

    auto parseBoolValue = [](const std::string &value) {
        return value == "true" || value == "1" || value == "render_pipeline.true";
    };
    auto parseFloatValue = [&](const ShaderPackLoader::VariableConfig &config, const std::string &value) {
        if (config.type == "bool") { return parseBoolValue(value) ? 1.0f : 0.0f; }
        if (config.type == "int" || config.type.rfind("int_range:", 0) == 0) {
            try {
                return static_cast<float>(std::stoi(value));
            } catch (...) { return 0.0f; }
        }
        if (config.type.rfind("enum:", 0) == 0) {
            std::string options = config.type.substr(5);
            std::stringstream stream(options);
            std::string option;
            int index = 0;
            while (std::getline(stream, option, '-')) {
                if (option == value) { return static_cast<float>(index); }
                index++;
            }
            try {
                return static_cast<float>(std::stoi(value));
            } catch (...) { return 0.0f; }
        }
        try {
            return std::stof(value);
        } catch (...) { return 0.0f; }
    };

    std::vector<float> values(execution.variables.size(), 0.0f);
    for (size_t i = 0; i < execution.variables.size(); i++) {
        const auto &variable = execution.variables[i];
        auto valueIter = variables.find(variable.name);
        values[i] =
            parseFloatValue(variable, valueIter != variables.end() ? valueIter->second.value : variable.defaultValue);
    }

    if (shaderStages == 0) { return; }
    vkCmdPushConstants(commandBuffer->vkCommandBuffer(), descriptorTable->vkPipelineLayout(), shaderStages, 0,
                       static_cast<uint32_t>(values.size() * sizeof(float)), values.data());
}

const std::vector<ExpressionEvaluator::Variable> &ShaderPack::ExecutionRuntime::expressionVariables() const {
    if (!expressionVariablesDirty) { return cachedExpressionVariables; }

    cachedExpressionVariables.clear();
    cachedExpressionVariables.reserve(executionVariables.size() + runtimeExpressionVariables.size());
    std::unordered_set<std::string> seen;
    seen.reserve(cachedExpressionVariables.capacity());

    auto appendExpressionVariable = [&](const std::string &name, double value) {
        if (!seen.emplace(name).second) { return; }
        cachedExpressionVariables.push_back({
            .name = name,
            .value = value,
        });
    };

    for (const auto &[name, variable] : executionVariables) {
        (void)name;
        try {
            appendExpressionVariable(variable.name,
                                     ShaderPack::parseExecutionNumber(variable.name, variable.type, variable.value));
        } catch (...) {}
    }
    for (const auto &variable : runtimeExpressionVariables) { appendExpressionVariable(variable.name, variable.value); }
    expressionVariablesDirty = false;
    return cachedExpressionVariables;
}

void ShaderPack::ExecutionRuntime::invalidateExpressionVariables() const {
    expressionVariablesDirty = true;
}

ShaderPack::ExecutionVariable &
ShaderPack::ExecutionRuntime::findOrCreateVariable(const std::string &name,
                                                   const ShaderPackLoader::VariableConfig &variableConfig) {
    auto iter = executionVariables.find(name);
    if (iter != executionVariables.end()) { return iter->second; }

    ExecutionVariable variable = {
        .name = name,
        .value = variableConfig.defaultValue,
        .type = variableConfig.type,
    };
    invalidateExpressionVariables();
    return executionVariables.emplace(name, std::move(variable)).first->second;
}

void ShaderPack::SequentialOp::operator()(ExecutionRuntime &runtime) const {
    for (const auto &executionOp : executionOps) { (*executionOp)(runtime); }
}

struct ShaderPack::CompiledExecutionCondition {
    bool hasExpression = false;
    bool constantValue = false;
    ExpressionEvaluator::CompiledExpression compiledExpression;

    bool evaluate(const ShaderPack::ExecutionRuntime &runtime) const {
        if (!hasExpression) { return constantValue; }
        return compiledExpression.evaluate(runtime.expressionVariables()) != 0.0;
    }
};

std::unordered_set<std::string> ShaderPack::activePassNames(ShaderPackLoader::Stage stage,
                                                            bool includeAttributeDefines) const {
    ShaderPack::ExecutionRuntime runtime;
    runtime.loopLimit = 1024;
    const StageRuntime &stageRuntimeConfig = stageRuntime(stage);
    for (const auto &[name, value] : stageRuntimeConfig.globalVariables) {
        auto configIter = stageRuntimeConfig.executionVariableConfigs.find(name);
        runtime.executionVariables.emplace(
            name,
            ExecutionVariable{
                .name = name,
                .value = value,
                .type = configIter == stageRuntimeConfig.executionVariableConfigs.end() ? "" : configIter->second.type,
            });
    }

    const std::vector<std::string> variableNames = expressionVariableNames(stage, includeAttributeDefines);
    const std::vector<ExpressionEvaluator::Variable> staticVariables =
        staticExpressionVariables(includeAttributeDefines);

    std::unordered_set<std::string> passNames;
    std::function<void(const std::vector<ShaderPackLoader::ExecutionCommand> &)> collectCommands;
    collectCommands = [&](const std::vector<ShaderPackLoader::ExecutionCommand> &commands) {
        for (const auto &command : commands) {
            switch (command.type) {
                case ShaderPackLoader::ExecutionCommand::Type::Assign: {
                    for (const auto &[variableName, assignment] : command.assignments) {
                        ShaderPackLoader::VariableConfig variableConfig = buildExecutionVariableConfigForAssignment(
                            variableName, assignment, stageRuntimeConfig.executionVariableConfigs);
                        ExecutionVariable &variable = runtime.findOrCreateVariable(variableName, variableConfig);

                        std::string newValue;
                        if (assignment.value.is_string() && isExecutionNumericExpressionType(variableConfig.type)) {
                            ExpressionEvaluator::CompiledExpression expression(assignment.value.get<std::string>(),
                                                                               variableNames, staticVariables);
                            newValue = formatExecutionValue(variableConfig.type,
                                                            expression.evaluate(runtime.expressionVariables()));
                        } else if (assignment.value.is_boolean()) {
                            newValue = assignment.value.get<bool>() ? "true" : "false";
                        } else if (assignment.value.is_number_integer() || assignment.value.is_number_unsigned()) {
                            newValue = std::to_string(assignment.value.get<int64_t>());
                        } else if (assignment.value.is_number_float()) {
                            std::ostringstream stream;
                            stream << assignment.value.get<double>();
                            newValue = stream.str();
                        } else if (assignment.value.is_string()) {
                            newValue = assignment.value.get<std::string>();
                        } else {
                            newValue = variableConfig.defaultValue;
                        }

                        if (variable.value != newValue || variable.type != variableConfig.type) {
                            variable.value = std::move(newValue);
                            variable.type = variableConfig.type;
                            runtime.invalidateExpressionVariables();
                        }
                    }
                    break;
                }
                case ShaderPackLoader::ExecutionCommand::Type::Pass:
                case ShaderPackLoader::ExecutionCommand::Type::SharcUpdate: passNames.insert(command.passName); break;
                case ShaderPackLoader::ExecutionCommand::Type::BufferUpdate: break;
                case ShaderPackLoader::ExecutionCommand::Type::IfElse: {
                    CompiledExecutionCondition condition =
                        compileExecutionCondition(command.condition, variableNames, staticVariables);
                    collectCommands(condition.evaluate(runtime) ? command.thenCommands : command.elseCommands);
                    break;
                }
                case ShaderPackLoader::ExecutionCommand::Type::While: {
                    CompiledExecutionCondition condition =
                        compileExecutionCondition(command.condition, variableNames, staticVariables);
                    uint32_t iteration = 0;
                    while (condition.evaluate(runtime)) {
                        if (iteration++ >= runtime.loopLimit) {
                            throw std::runtime_error("active pass collection while exceeded iteration limit");
                        }
                        collectCommands(command.thenCommands);
                    }
                    break;
                }
            }
        }
    };
    collectCommands(execution(stage).commands);
    return passNames;
}

ShaderPack::ActiveRuntimeResources ShaderPack::collectActiveRuntimeResources() const {
    ActiveRuntimeResources resources;
    std::unordered_map<std::string, const ShaderPackLoader::PassConfig *> passConfigs;
    passConfigs.reserve(shaderPack_.passes.size());
    for (const auto &passConfig : shaderPack_.passes) {
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen:
                passConfigs[passConfig.fullScreen.name] = &passConfig;
                break;
            case ShaderPackLoader::PassConfig::Type::Render: passConfigs[passConfig.render.name] = &passConfig; break;
            case ShaderPackLoader::PassConfig::Type::RayTracing:
                passConfigs[passConfig.rayTracing.name] = &passConfig;
                break;
            case ShaderPackLoader::PassConfig::Type::Compute: passConfigs[passConfig.compute.name] = &passConfig; break;
        }
    }

    std::unordered_set<std::string> activePasses;
    for (ShaderPackLoader::Stage stage : {ShaderPackLoader::Stage::RayTracing, ShaderPackLoader::Stage::PostRender}) {
        std::unordered_set<std::string> stagePasses = activePassNames(stage, true);
        if (execution(stage).commands.empty()) {
            for (const auto &passConfig : shaderPack_.passes) {
                if (passConfig.stage != stage) { continue; }
                switch (passConfig.type) {
                    case ShaderPackLoader::PassConfig::Type::FullScreen:
                        stagePasses.insert(passConfig.fullScreen.name);
                        break;
                    case ShaderPackLoader::PassConfig::Type::Render: stagePasses.insert(passConfig.render.name); break;
                    case ShaderPackLoader::PassConfig::Type::RayTracing:
                        stagePasses.insert(passConfig.rayTracing.name);
                        break;
                    case ShaderPackLoader::PassConfig::Type::Compute:
                        stagePasses.insert(passConfig.compute.name);
                        break;
                }
            }
        }
        activePasses.insert(stagePasses.begin(), stagePasses.end());
    }

    auto addResourceList = [&](const ShaderPackLoader::ResourceList &resourceList) {
        resources.textures.insert(resourceList.images.begin(), resourceList.images.end());
        resources.buffers.insert(resourceList.buffers.begin(), resourceList.buffers.end());
    };

    std::function<void(const std::vector<ShaderPackLoader::ExecutionCommand> &)> addExecutionCommandResources;
    addExecutionCommandResources = [&](const std::vector<ShaderPackLoader::ExecutionCommand> &commands) {
        for (const auto &command : commands) {
            switch (command.type) {
                case ShaderPackLoader::ExecutionCommand::Type::BufferUpdate:
                    resources.buffers.insert(command.bufferUpdate.bufferName);
                    break;
                case ShaderPackLoader::ExecutionCommand::Type::IfElse:
                    addExecutionCommandResources(command.thenCommands);
                    addExecutionCommandResources(command.elseCommands);
                    break;
                case ShaderPackLoader::ExecutionCommand::Type::While:
                    addExecutionCommandResources(command.thenCommands);
                    break;
                case ShaderPackLoader::ExecutionCommand::Type::Assign:
                case ShaderPackLoader::ExecutionCommand::Type::Pass:
                case ShaderPackLoader::ExecutionCommand::Type::SharcUpdate: break;
            }
        }
    };
    addExecutionCommandResources(shaderPack_.rayTracingExecution.commands);
    addExecutionCommandResources(shaderPack_.postRenderExecution.commands);

    for (const std::string &passName : activePasses) {
        auto passIter = passConfigs.find(passName);
        if (passIter == passConfigs.end()) { continue; }
        const ShaderPackLoader::PassConfig &passConfig = *passIter->second;
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen:
                addResourceList(passConfig.fullScreen.inputs);
                addResourceList(passConfig.fullScreen.outputs);
                resources.textures.insert(passConfig.fullScreen.target);
                break;
            case ShaderPackLoader::PassConfig::Type::Render:
                addResourceList(passConfig.render.inputs);
                addResourceList(passConfig.render.outputs);
                break;
            case ShaderPackLoader::PassConfig::Type::RayTracing:
                addResourceList(passConfig.rayTracing.inputs);
                addResourceList(passConfig.rayTracing.outputs);
                if (passConfig.rayTracing.indirectArgsBuffer.has_value()) {
                    resources.buffers.insert(*passConfig.rayTracing.indirectArgsBuffer);
                }
                break;
            case ShaderPackLoader::PassConfig::Type::Compute:
                addResourceList(passConfig.compute.inputs);
                addResourceList(passConfig.compute.outputs);
                if (passConfig.compute.indirectArgsBuffer.has_value()) {
                    resources.buffers.insert(*passConfig.compute.indirectArgsBuffer);
                }
                break;
        }
    }

    return resources;
}

ShaderPack::CompiledExecutionCondition
ShaderPack::compileExecutionCondition(const nlohmann::json &condition,
                                      const std::vector<std::string> &variableNames,
                                      const std::vector<ExpressionEvaluator::Variable> &staticVariables) {
    ShaderPack::CompiledExecutionCondition compiledCondition;
    if (condition.is_boolean()) {
        compiledCondition.constantValue = condition.get<bool>();
        return compiledCondition;
    }
    if (condition.is_number()) {
        compiledCondition.constantValue = condition.get<double>() != 0.0;
        return compiledCondition;
    }
    if (condition.is_string()) {
        compiledCondition.hasExpression = true;
        compiledCondition.compiledExpression =
            ExpressionEvaluator::CompiledExpression(condition.get<std::string>(), variableNames, staticVariables);
        return compiledCondition;
    }
    throw std::runtime_error("execution condition must be a string, boolean, or number");
}

ShaderPackLoader::VariableConfig ShaderPack::buildExecutionVariableConfigForAssignment(
    const std::string &variableName,
    const ShaderPackLoader::ExecutionCommand::Assignment &assignment,
    const std::unordered_map<std::string, ShaderPackLoader::VariableConfig> &variableConfigs) {
    auto variableConfigIter = variableConfigs.find(variableName);
    if (variableConfigIter != variableConfigs.end()) {
        if (assignment.explicitType && assignment.type.has_value() &&
            *assignment.type != variableConfigIter->second.type) {
            throw std::runtime_error("execution assignment type for " + variableName +
                                     " does not match global variable type");
        }
        return variableConfigIter->second;
    }

    std::string type;
    if (assignment.type.has_value()) {
        type = *assignment.type;
    } else if (assignment.value.is_boolean()) {
        type = "bool";
    } else if (assignment.value.is_number_integer() || assignment.value.is_number_unsigned()) {
        type = "int";
    } else if (assignment.value.is_number_float() || assignment.value.is_string()) {
        type = "float";
    } else {
        type = "string";
    }
    return {
        .name = variableName,
        .type = type,
        .defaultValue = defaultExecutionValueForType(type),
    };
}

struct ShaderPack::AssignmentExecutionOp final : ShaderPack::ExecutionOp {
    enum class ValueKind {
        Literal,
        Expression,
    };

    struct CompiledAssignment {
        std::string name;
        ShaderPackLoader::VariableConfig variableConfig;
        ValueKind valueKind = ValueKind::Literal;
        std::string literalValue;
        ExpressionEvaluator::CompiledExpression compiledExpression;
    };

    std::vector<CompiledAssignment> assignments;

    void operator()(ShaderPack::ExecutionRuntime &runtime) const override {
        for (const auto &assignment : assignments) {
            ShaderPack::ExecutionVariable &variable =
                runtime.findOrCreateVariable(assignment.name, assignment.variableConfig);
            std::string newValue;
            if (assignment.valueKind == ValueKind::Expression) {
                newValue = ShaderPack::formatExecutionValue(
                    assignment.variableConfig.type,
                    assignment.compiledExpression.evaluate(runtime.expressionVariables()));
            } else {
                newValue = assignment.literalValue;
            }
            if (variable.type != assignment.variableConfig.type || variable.value != newValue) {
                variable.type = assignment.variableConfig.type;
                variable.value = std::move(newValue);
                runtime.invalidateExpressionVariables();
            }
        }
    }
};

struct ShaderPack::IfExecutionOp final : ShaderPack::ExecutionOp {
    CompiledExecutionCondition condition;
    std::unique_ptr<ShaderPack::SequentialOp> thenSequence;
    std::unique_ptr<ShaderPack::SequentialOp> elseSequence;

    void operator()(ShaderPack::ExecutionRuntime &runtime) const override {
        if (condition.evaluate(runtime)) {
            (*thenSequence)(runtime);
        } else if (elseSequence != nullptr) {
            (*elseSequence)(runtime);
        }
    }
};

struct ShaderPack::WhileExecutionOp final : ShaderPack::ExecutionOp {
    CompiledExecutionCondition condition;
    std::unique_ptr<ShaderPack::SequentialOp> bodySequence;

    void operator()(ShaderPack::ExecutionRuntime &runtime) const override {
        uint32_t iteration = 0;
        while (condition.evaluate(runtime)) {
            if (iteration++ >= runtime.loopLimit) {
                throw std::runtime_error("execution while exceeded iteration limit");
            }
            (*bodySequence)(runtime);
        }
    }
};

std::unique_ptr<ShaderPack::SequentialOp>
ShaderPack::compileExecutionProgram(ShaderPackLoader::Stage stage,
                                    bool includeAttributeDefines,
                                    const ExecutionOpBuilder &passExecutionOpBuilder,
                                    const ExecutionOpBuilder &sharcUpdateExecutionOpBuilder,
                                    const BufferUpdateExecutionOpBuilder &bufferUpdateExecutionOpBuilder) const {
    const auto &stageRuntimeConfig = stageRuntime(stage);
    const std::vector<std::string> variableNames = expressionVariableNames(stage, includeAttributeDefines);
    const std::vector<ExpressionEvaluator::Variable> staticVariables =
        staticExpressionVariables(includeAttributeDefines);

    std::function<std::unique_ptr<SequentialOp>(const std::vector<ShaderPackLoader::ExecutionCommand> &)>
        compileCommandSequence;
    compileCommandSequence = [&](const std::vector<ShaderPackLoader::ExecutionCommand> &commands) {
        auto commandSequence = std::make_unique<SequentialOp>();
        commandSequence->executionOps.reserve(commands.size());

        for (const auto &command : commands) {
            switch (command.type) {
                case ShaderPackLoader::ExecutionCommand::Type::Assign: {
                    auto assignmentOperation = std::make_unique<AssignmentExecutionOp>();
                    assignmentOperation->assignments.reserve(command.assignments.size());
                    for (const auto &[variableName, assignment] : command.assignments) {
                        auto variableConfig = buildExecutionVariableConfigForAssignment(
                            variableName, assignment, stageRuntimeConfig.executionVariableConfigs);
                        AssignmentExecutionOp::CompiledAssignment compiledAssignment = {
                            .name = variableName,
                            .variableConfig = variableConfig,
                        };

                        const nlohmann::json &assignmentValue = assignment.value;
                        if (assignmentValue.is_string() && isExecutionNumericExpressionType(variableConfig.type)) {
                            compiledAssignment.valueKind = AssignmentExecutionOp::ValueKind::Expression;
                            compiledAssignment.compiledExpression = ExpressionEvaluator::CompiledExpression(
                                assignmentValue.get<std::string>(), variableNames, staticVariables);
                        } else if (assignmentValue.is_string()) {
                            compiledAssignment.literalValue = assignmentValue.get<std::string>();
                        } else if (assignmentValue.is_boolean()) {
                            compiledAssignment.literalValue = assignmentValue.get<bool>() ? "true" : "false";
                        } else if (assignmentValue.is_number()) {
                            compiledAssignment.literalValue =
                                formatExecutionValue(variableConfig.type, assignmentValue.get<double>());
                        } else {
                            throw std::runtime_error("execution assignment for " + variableName +
                                                     " must be a string, boolean, or number");
                        }
                        assignmentOperation->assignments.push_back(std::move(compiledAssignment));
                    }
                    commandSequence->executionOps.push_back(std::move(assignmentOperation));
                    break;
                }
                case ShaderPackLoader::ExecutionCommand::Type::BufferUpdate:
                    if (!bufferUpdateExecutionOpBuilder) {
                        throw std::runtime_error("execution buffer_update is not supported in this stage");
                    }
                    commandSequence->executionOps.push_back(bufferUpdateExecutionOpBuilder(command.bufferUpdate));
                    break;
                case ShaderPackLoader::ExecutionCommand::Type::Pass:
                    if (!passExecutionOpBuilder) {
                        throw std::runtime_error("execution pass is not supported in this stage");
                    }
                    commandSequence->executionOps.push_back(passExecutionOpBuilder(command.passName));
                    break;
                case ShaderPackLoader::ExecutionCommand::Type::SharcUpdate:
                    if (!sharcUpdateExecutionOpBuilder) {
                        throw std::runtime_error("execution sharc_update is not supported in this stage");
                    }
                    commandSequence->executionOps.push_back(sharcUpdateExecutionOpBuilder(command.passName));
                    break;
                case ShaderPackLoader::ExecutionCommand::Type::IfElse: {
                    auto branchOperation = std::make_unique<IfExecutionOp>();
                    branchOperation->condition =
                        compileExecutionCondition(command.condition, variableNames, staticVariables);
                    branchOperation->thenSequence = compileCommandSequence(command.thenCommands);
                    branchOperation->elseSequence = compileCommandSequence(command.elseCommands);
                    commandSequence->executionOps.push_back(std::move(branchOperation));
                    break;
                }
                case ShaderPackLoader::ExecutionCommand::Type::While: {
                    auto loopOperation = std::make_unique<WhileExecutionOp>();
                    loopOperation->condition =
                        compileExecutionCondition(command.condition, variableNames, staticVariables);
                    loopOperation->bodySequence = compileCommandSequence(command.thenCommands);
                    commandSequence->executionOps.push_back(std::move(loopOperation));
                    break;
                }
            }
        }
        return commandSequence;
    };

    return compileCommandSequence(execution(stage).commands);
}

bool ShaderPack::hasSharcRuntime() const {
    return hasSharcRuntime_;
}

bool ShaderPack::runtimeResourcesReady() const {
    return runtimeResourcesReady_;
}

std::optional<std::reference_wrapper<ShaderPack::RuntimeTexture>>
ShaderPack::findRuntimeTexture(std::string_view name) {
    auto iter = runtimeTextureIndices_.find(std::string(name));
    if (iter == runtimeTextureIndices_.end()) { return std::nullopt; }
    return runtimeTextures_[iter->second];
}

std::optional<std::reference_wrapper<const ShaderPack::RuntimeTexture>>
ShaderPack::findRuntimeTexture(std::string_view name) const {
    auto iter = runtimeTextureIndices_.find(std::string(name));
    if (iter == runtimeTextureIndices_.end()) { return std::nullopt; }
    return runtimeTextures_[iter->second];
}

std::optional<std::reference_wrapper<ShaderPack::RuntimeBuffer>> ShaderPack::findRuntimeBuffer(std::string_view name) {
    auto iter = runtimeBufferIndices_.find(std::string(name));
    if (iter == runtimeBufferIndices_.end()) { return std::nullopt; }
    return runtimeBuffers_[iter->second];
}

std::optional<std::reference_wrapper<const ShaderPack::RuntimeBuffer>>
ShaderPack::findRuntimeBuffer(std::string_view name) const {
    auto iter = runtimeBufferIndices_.find(std::string(name));
    if (iter == runtimeBufferIndices_.end()) { return std::nullopt; }
    return runtimeBuffers_[iter->second];
}

std::shared_ptr<vk::DeviceLocalImage> ShaderPack::findRuntimeVKTexture(RuntimeTexture &runtimeTexture,
                                                                       uint32_t frameIndex) {
    return static_cast<const ShaderPack *>(this)->findRuntimeVKTexture(
        const_cast<const RuntimeTexture &>(runtimeTexture), frameIndex);
}

std::shared_ptr<vk::DeviceLocalImage> ShaderPack::findRuntimeVKTexture(const RuntimeTexture &runtimeTexture,
                                                                       uint32_t frameIndex) const {
    if (runtimeTexture.config.imported) { return runtimeTexture.importedImage; }
    return runtimeTexture.frameImages[runtimeTexture.config.shared ? 0 : frameIndex];
}

uint32_t ShaderPack::runtimeTextureSingleLayerViewIndex(const RuntimeTexture &runtimeTexture, uint32_t layer) const {
    if (runtimeTexture.singleLayerViewBaseIndex == 0) {
        throw std::runtime_error("runtime texture does not expose per-layer views: " + runtimeTexture.config.name);
    }
    if (runtimeTexture.config.dimension == ShaderPackLoader::TextureDimension::Cube) {
        if (layer >= 6) { throw std::runtime_error("cube face index out of range: " + runtimeTexture.config.name); }
        return runtimeTexture.singleLayerViewBaseIndex + layer;
    }
    if (runtimeTexture.config.dimension != ShaderPackLoader::TextureDimension::Texture2DArray) {
        throw std::runtime_error("runtime texture is not a 2d_array: " + runtimeTexture.config.name);
    }

    uint32_t layerCount = 0;
    if (runtimeTexture.config.imported) {
        layerCount = runtimeTexture.config.importedDepth;
    } else if (!runtimeTexture.frameImages.empty() && runtimeTexture.frameImages[0] != nullptr) {
        layerCount = runtimeTexture.frameImages[0]->layer();
    }
    if (layer >= layerCount) {
        throw std::runtime_error("2d_array layer index out of range: " + runtimeTexture.config.name);
    }
    return runtimeTexture.singleLayerViewBaseIndex + layer;
}

std::shared_ptr<vk::DeviceLocalBuffer> ShaderPack::findRuntimeVKBuffer(RuntimeBuffer &runtimeBuffer,
                                                                       uint32_t frameIndex) {
    return static_cast<const ShaderPack *>(this)->findRuntimeVKBuffer(const_cast<const RuntimeBuffer &>(runtimeBuffer),
                                                                      frameIndex);
}

std::shared_ptr<vk::DeviceLocalBuffer> ShaderPack::findRuntimeVKBuffer(const RuntimeBuffer &runtimeBuffer,
                                                                       uint32_t frameIndex) const {
    return runtimeBuffer.frameBuffers[runtimeBuffer.config.shared ? 0 : frameIndex];
}

std::unordered_map<std::string, std::string>
ShaderPack::mergeDefinitions(const std::unordered_map<std::string, std::string> &lhs,
                             const std::unordered_map<std::string, std::string> &rhs) {
    std::unordered_map<std::string, std::string> merged = lhs;
    for (const auto &[key, value] : rhs) { merged[key] = value; }
    return merged;
}

double ShaderPack::evaluateNumericExpression(const std::string &expression) const {
    auto parseBoolValue = [](const std::string &value) {
        return value == "true" || value == "1" || value == "render_pipeline.true";
    };
    auto parseNumber = [&](const ShaderPackLoader::AttributeConfig &config, const std::string &value) {
        if (config.type == "bool") { return parseBoolValue(value) ? 1.0 : 0.0; }
        if (config.type.rfind("enum:", 0) == 0) {
            std::string options = config.type.substr(5);
            std::stringstream stream(options);
            std::string option;
            int index = 0;
            while (std::getline(stream, option, '-')) {
                if (option == value) { return static_cast<double>(index); }
                index++;
            }
        }
        return std::stod(value);
    };

    std::vector<ExpressionEvaluator::Variable> expressionVariables = runtimeResourceExpressionVariables_;
    expressionVariables.reserve(expressionVariables.size() + injectedExpressionVariables_.size() +
                                shaderPack_.attributes.size());
    std::unordered_map<std::string, size_t> variableIndices;
    variableIndices.reserve(expressionVariables.size() + shaderPack_.attributes.size());

    for (size_t i = 0; i < expressionVariables.size(); ++i) { variableIndices.emplace(expressionVariables[i].name, i); }

    auto appendExpressionVariable = [&](const std::string &name, double value) {
        if (variableIndices.find(name) != variableIndices.end()) { return; }
        variableIndices[name] = expressionVariables.size();
        expressionVariables.push_back({
            .name = name,
            .value = value,
        });
    };

    for (const auto &variable : injectedExpressionVariables_) {
        appendExpressionVariable(variable.name, variable.value);
    }

    for (const auto &attribute : shaderPack_.attributes) {
        const std::string &currentValue = [&]() -> const std::string & {
            auto iter = staticAttributeValues_.find(attribute.name);
            return iter == staticAttributeValues_.end() ? attribute.defaultValue : iter->second;
        }();

        if (attribute.define.direct.has_value()) {
            if (attribute.type == "bool") {
                appendExpressionVariable(*attribute.define.direct, parseBoolValue(currentValue) ? 1.0 : 0.0);
            } else if (attribute.type == "int" || attribute.type.rfind("int_range:", 0) == 0 ||
                       attribute.type.rfind("float_range:", 0) == 0 || attribute.type.rfind("enum:", 0) == 0) {
                appendExpressionVariable(*attribute.define.direct, parseNumber(attribute, currentValue));
            }
        }

        for (const auto &[name, expression] : attribute.define.expressions) {
            auto value = tryEvaluateScalarDefineExpression(attribute, currentValue, expression);
            if (value.has_value()) { appendExpressionVariable(name, *value); }
        }
    }

    return ExpressionEvaluator::evaluate(expression, expressionVariables);
}

void ShaderPack::initStageRuntime(StageRuntime &stageRuntime, const ShaderPackLoader::ExecutionConfig &execution) {
    stageRuntime.executionVariableConfigs.clear();
    stageRuntime.globalVariables.clear();

    for (const auto &variable : execution.globalVariables) {
        stageRuntime.globalVariables[variable.name] = variable.defaultValue;
    }
    for (const auto &variable : execution.variables) {
        stageRuntime.executionVariableConfigs[variable.name] = variable;
    }
}

void ShaderPack::initRuntimeTextures() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    uint32_t frameCount = framework->swapchain()->imageCount();

    std::vector<RuntimeTexture> runtimeTextures(shaderPack_.textures.size());
    struct TextureAllocationInfo {
        bool shouldAllocate = false;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 1;
        bool isCube = false;
        bool is3D = false;
        bool is2DArray = false;
        VkImageUsageFlags usage = 0;
    };

    constexpr size_t unusedTextureOrder = std::numeric_limits<size_t>::max();

    struct TextureLifetime {
        size_t firstUse = 0;
        size_t lastUse = 0;
        size_t firstRead = 0;
        size_t firstWrite = 0;
        bool hasWrite = false;
        bool hasSelfReadWrite = false;
    };

    struct PassOccurrence {
        std::string passName;
        size_t firstOrder = 0;
        size_t lastOrder = 0;
    };

    struct TextureAliasGroup {
        size_t textureIndex = 0;
        size_t lastUse = 0;
        std::vector<std::shared_ptr<vk::DeviceLocalImage>> frameImages;
    };

    auto textureFrameCount = [&](const ShaderPackLoader::TextureConfig &textureConfig) {
        return textureConfig.shared ? 1u : frameCount;
    };

    std::unordered_map<std::string, size_t> textureIndices;
    textureIndices.reserve(shaderPack_.textures.size());
    for (size_t textureIndex = 0; textureIndex < shaderPack_.textures.size(); textureIndex++) {
        textureIndices[shaderPack_.textures[textureIndex].name] = textureIndex;
    }

    std::vector<TextureAllocationInfo> allocationInfos(shaderPack_.textures.size());
    for (size_t textureIndex = 0; textureIndex < shaderPack_.textures.size(); textureIndex++) {
        const auto &textureConfig = shaderPack_.textures[textureIndex];
        RuntimeTexture runtimeTexture;
        runtimeTexture.config = textureConfig;
        if (!textureConfig.imported && !activeRuntimeResources_.hasTexture(textureConfig.name)) {
            runtimeTextures[textureIndex] = std::move(runtimeTexture);
            continue;
        }
        if (textureConfig.sampledBinding.has_value()) {
            runtimeTexture.sampler = vk::Sampler::create(device, textureConfig.filter, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                                         textureConfig.addressMode);
        }
        if (textureConfig.imported) {
            runtimeTexture.sampledViewIndex = 0;
        } else {
            TextureAllocationInfo allocationInfo;
            allocationInfo.shouldAllocate = true;
            if (textureConfig.scale > 0.0f) {
                allocationInfo.width =
                    std::max<uint32_t>(1, static_cast<uint32_t>(referenceWidth_ * textureConfig.scale));
                allocationInfo.height =
                    std::max<uint32_t>(1, static_cast<uint32_t>(referenceHeight_ * textureConfig.scale));
            } else if (!textureConfig.widthExpression.empty() && !textureConfig.heightExpression.empty()) {
                double widthValue = evaluateNumericExpression(textureConfig.widthExpression);
                double heightValue = evaluateNumericExpression(textureConfig.heightExpression);
                if (!std::isfinite(widthValue) || !std::isfinite(heightValue) || widthValue <= 0.0 ||
                    heightValue <= 0.0) {
                    throw std::runtime_error("invalid runtime texture extent: " + textureConfig.name);
                }
                allocationInfo.width = std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(widthValue)));
                allocationInfo.height = std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(heightValue)));
                if (textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture3D ||
                    textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
                    if (textureConfig.depthExpression.empty()) {
                        throw std::runtime_error("runtime layered texture requires depth: " + textureConfig.name);
                    }
                    double depthValue = evaluateNumericExpression(textureConfig.depthExpression);
                    if (!std::isfinite(depthValue) || depthValue <= 0.0) {
                        throw std::runtime_error("invalid runtime texture depth: " + textureConfig.name);
                    }
                    allocationInfo.depth = std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(depthValue)));
                }
            } else {
                throw std::runtime_error("runtime texture requires width/height or scale: " + textureConfig.name);
            }

            allocationInfo.isCube = textureConfig.dimension == ShaderPackLoader::TextureDimension::Cube;
            allocationInfo.is3D = textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture3D;
            allocationInfo.is2DArray = textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture2DArray;
            allocationInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
            if (!allocationInfo.is3D) { allocationInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }
            if (textureConfig.storageBinding.has_value()) { allocationInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT; }
            if (textureConfig.transferSrc) { allocationInfo.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; }
            allocationInfos[textureIndex] = allocationInfo;
        }

        runtimeTextures[textureIndex] = std::move(runtimeTexture);
    }

    std::unordered_map<std::string, const ShaderPackLoader::PassConfig *> passConfigs;
    passConfigs.reserve(shaderPack_.passes.size());
    for (const auto &passConfig : shaderPack_.passes) {
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen:
                passConfigs[passConfig.fullScreen.name] = &passConfig;
                break;
            case ShaderPackLoader::PassConfig::Type::Render: passConfigs[passConfig.render.name] = &passConfig; break;
            case ShaderPackLoader::PassConfig::Type::RayTracing:
                passConfigs[passConfig.rayTracing.name] = &passConfig;
                break;
            case ShaderPackLoader::PassConfig::Type::Compute: passConfigs[passConfig.compute.name] = &passConfig; break;
        }
    }

    std::vector<PassOccurrence> passOccurrences;
    size_t currentOrder = 0;
    std::function<void(const std::vector<ShaderPackLoader::ExecutionCommand> &)> collectPassOccurrences;
    collectPassOccurrences = [&](const std::vector<ShaderPackLoader::ExecutionCommand> &commands) {
        for (const auto &command : commands) {
            switch (command.type) {
                case ShaderPackLoader::ExecutionCommand::Type::Pass:
                case ShaderPackLoader::ExecutionCommand::Type::SharcUpdate:
                    passOccurrences.push_back({
                        .passName = command.passName,
                        .firstOrder = currentOrder,
                        .lastOrder = currentOrder,
                    });
                    currentOrder++;
                    break;
                case ShaderPackLoader::ExecutionCommand::Type::IfElse:
                    collectPassOccurrences(command.thenCommands);
                    collectPassOccurrences(command.elseCommands);
                    break;
                case ShaderPackLoader::ExecutionCommand::Type::While: {
                    size_t loopFirstOrder = currentOrder;
                    size_t firstLoopOccurrence = passOccurrences.size();
                    collectPassOccurrences(command.thenCommands);
                    size_t loopLastOrder = currentOrder > loopFirstOrder ? currentOrder - 1 : loopFirstOrder;
                    for (size_t occurrenceIndex = firstLoopOccurrence; occurrenceIndex < passOccurrences.size();
                         occurrenceIndex++) {
                        passOccurrences[occurrenceIndex].firstOrder =
                            std::min(passOccurrences[occurrenceIndex].firstOrder, loopFirstOrder);
                        passOccurrences[occurrenceIndex].lastOrder =
                            std::max(passOccurrences[occurrenceIndex].lastOrder, loopLastOrder);
                    }
                    break;
                }
                case ShaderPackLoader::ExecutionCommand::Type::Assign: break;
                case ShaderPackLoader::ExecutionCommand::Type::BufferUpdate: break;
            }
        }
    };
    collectPassOccurrences(shaderPack_.rayTracingExecution.commands);
    collectPassOccurrences(shaderPack_.postRenderExecution.commands);

    auto passImageResources = [](const ShaderPackLoader::PassConfig &passConfig) {
        const ShaderPackLoader::ResourceList *inputs = nullptr;
        const ShaderPackLoader::ResourceList *outputs = nullptr;
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen:
                inputs = &passConfig.fullScreen.inputs;
                outputs = &passConfig.fullScreen.outputs;
                break;
            case ShaderPackLoader::PassConfig::Type::Render:
                inputs = &passConfig.render.inputs;
                outputs = &passConfig.render.outputs;
                break;
            case ShaderPackLoader::PassConfig::Type::RayTracing:
                inputs = &passConfig.rayTracing.inputs;
                outputs = &passConfig.rayTracing.outputs;
                break;
            case ShaderPackLoader::PassConfig::Type::Compute:
                inputs = &passConfig.compute.inputs;
                outputs = &passConfig.compute.outputs;
                break;
        }
        return std::pair<const ShaderPackLoader::ResourceList *, const ShaderPackLoader::ResourceList *>(inputs,
                                                                                                         outputs);
    };

    std::vector<TextureLifetime> lifetimes(shaderPack_.textures.size());
    for (TextureLifetime &lifetime : lifetimes) {
        lifetime.firstUse = unusedTextureOrder;
        lifetime.firstRead = unusedTextureOrder;
        lifetime.firstWrite = unusedTextureOrder;
    }
    auto markTextureRead = [&](const std::string &textureName, size_t firstOrder, size_t lastOrder) {
        auto textureIter = textureIndices.find(textureName);
        if (textureIter == textureIndices.end()) { return; }
        TextureLifetime &lifetime = lifetimes[textureIter->second];
        lifetime.firstUse = std::min(lifetime.firstUse, firstOrder);
        lifetime.lastUse = std::max(lifetime.lastUse, lastOrder);
        lifetime.firstRead = std::min(lifetime.firstRead, firstOrder);
    };
    auto markTextureWrite = [&](const std::string &textureName, size_t firstOrder, size_t lastOrder) {
        auto textureIter = textureIndices.find(textureName);
        if (textureIter == textureIndices.end()) { return; }
        TextureLifetime &lifetime = lifetimes[textureIter->second];
        lifetime.firstUse = std::min(lifetime.firstUse, firstOrder);
        lifetime.lastUse = std::max(lifetime.lastUse, lastOrder);
        lifetime.firstWrite = std::min(lifetime.firstWrite, firstOrder);
        lifetime.hasWrite = true;
    };

    for (const auto &occurrence : passOccurrences) {
        auto passIter = passConfigs.find(occurrence.passName);
        if (passIter == passConfigs.end()) { continue; }

        auto [inputs, outputs] = passImageResources(*passIter->second);
        std::unordered_set<std::string> outputImages(outputs->images.begin(), outputs->images.end());
        for (const std::string &input : inputs->images) {
            markTextureRead(input, occurrence.firstOrder, occurrence.lastOrder);
            if (outputImages.find(input) != outputImages.end()) {
                auto textureIter = textureIndices.find(input);
                if (textureIter != textureIndices.end()) { lifetimes[textureIter->second].hasSelfReadWrite = true; }
            }
        }
        for (const std::string &output : outputs->images) {
            markTextureWrite(output, occurrence.firstOrder, occurrence.lastOrder);
        }
    }

    auto areAliasCompatible = [&](size_t lhsIndex, size_t rhsIndex) {
        const TextureAllocationInfo &lhs = allocationInfos[lhsIndex];
        const TextureAllocationInfo &rhs = allocationInfos[rhsIndex];
        return lhs.shouldAllocate == rhs.shouldAllocate && lhs.width == rhs.width && lhs.height == rhs.height &&
               lhs.depth == rhs.depth && lhs.isCube == rhs.isCube && lhs.is3D == rhs.is3D &&
               lhs.is2DArray == rhs.is2DArray &&
               shaderPack_.textures[lhsIndex].format == shaderPack_.textures[rhsIndex].format &&
               lhs.usage == rhs.usage &&
               textureFrameCount(shaderPack_.textures[lhsIndex]) == textureFrameCount(shaderPack_.textures[rhsIndex]);
    };

    auto canAliasTexture = [&](size_t textureIndex) {
        const auto &textureConfig = shaderPack_.textures[textureIndex];
        const TextureAllocationInfo &allocationInfo = allocationInfos[textureIndex];
        const TextureLifetime &lifetime = lifetimes[textureIndex];

        const bool isPureStorage2D = textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture2D &&
                                     textureConfig.storageBinding.has_value() &&
                                     !textureConfig.sampledBinding.has_value();
        const bool isWrittenBeforeRead =
            lifetime.firstRead == unusedTextureOrder || lifetime.firstWrite <= lifetime.firstRead;
        return allocationInfo.shouldAllocate && !textureConfig.imported && !textureConfig.shared && isPureStorage2D &&
               !textureConfig.transferSrc && lifetime.hasWrite && lifetime.firstUse != unusedTextureOrder &&
               isWrittenBeforeRead && !lifetime.hasSelfReadWrite;
    };

    std::vector<size_t> aliasCandidates;
    aliasCandidates.reserve(shaderPack_.textures.size());
    for (size_t textureIndex = 0; textureIndex < shaderPack_.textures.size(); textureIndex++) {
        if (canAliasTexture(textureIndex)) { aliasCandidates.push_back(textureIndex); }
    }
    std::sort(aliasCandidates.begin(), aliasCandidates.end(), [&](size_t lhs, size_t rhs) {
        if (lifetimes[lhs].firstUse != lifetimes[rhs].firstUse) {
            return lifetimes[lhs].firstUse < lifetimes[rhs].firstUse;
        }
        return shaderPack_.textures[lhs].name < shaderPack_.textures[rhs].name;
    });

    constexpr size_t unusedAliasGroup = std::numeric_limits<size_t>::max();
    std::vector<size_t> textureAliasGroups(shaderPack_.textures.size(), unusedAliasGroup);
    std::vector<TextureAliasGroup> aliasGroups;
    for (size_t textureIndex : aliasCandidates) {
        const TextureLifetime &lifetime = lifetimes[textureIndex];
        size_t selectedGroup = unusedAliasGroup;
        for (size_t groupIndex = 0; groupIndex < aliasGroups.size(); groupIndex++) {
            TextureAliasGroup &group = aliasGroups[groupIndex];
            if (group.lastUse < lifetime.firstUse && areAliasCompatible(group.textureIndex, textureIndex)) {
                selectedGroup = groupIndex;
                break;
            }
        }
        if (selectedGroup == unusedAliasGroup) {
            selectedGroup = aliasGroups.size();
            aliasGroups.push_back({
                .textureIndex = textureIndex,
                .lastUse = lifetime.lastUse,
                .frameImages = {},
            });
        } else {
            aliasGroups[selectedGroup].lastUse = std::max(aliasGroups[selectedGroup].lastUse, lifetime.lastUse);
        }
        textureAliasGroups[textureIndex] = selectedGroup;
    }

    auto createRuntimeFrameImage =
        [&](RuntimeTexture &runtimeTexture,
            const TextureAllocationInfo &allocationInfo) -> std::shared_ptr<vk::DeviceLocalImage> {
        const auto &textureConfig = runtimeTexture.config;
        if (allocationInfo.isCube) {
            auto image = vk::DeviceLocalImage::create(
                device, vma, false, allocationInfo.width, allocationInfo.height, 6, textureConfig.format,
                allocationInfo.usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

            for (int faceIndex = 0; faceIndex < 6; faceIndex++) {
                image->addImageView(VkImageViewCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = image->vkImage(),
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = image->vkFormat(),
                    .components =
                        {
                            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                        },
                    .subresourceRange =
                        {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = static_cast<uint32_t>(faceIndex),
                            .layerCount = 1,
                        },
                });
            }
            image->addImageView(VkImageViewCreateInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = image->vkImage(),
                .viewType = VK_IMAGE_VIEW_TYPE_CUBE,
                .format = image->vkFormat(),
                .components =
                    {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 6,
                    },
            });
            runtimeTexture.singleLayerViewBaseIndex = 1;
            runtimeTexture.sampledViewIndex = 7;
            return image;
        }

        runtimeTexture.sampledViewIndex = 0;
        if (allocationInfo.is3D) {
            return vk::DeviceLocalImage::create(device, vma, false, allocationInfo.width, allocationInfo.height,
                                                allocationInfo.depth, 1, textureConfig.format, allocationInfo.usage);
        }
        if (allocationInfo.is2DArray) {
            auto image = vk::DeviceLocalImage::create(device, vma, false, allocationInfo.width, allocationInfo.height,
                                                      allocationInfo.depth, textureConfig.format, allocationInfo.usage);
            for (uint32_t layerIndex = 0; layerIndex < allocationInfo.depth; layerIndex++) {
                image->addImageView(VkImageViewCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = image->vkImage(),
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = image->vkFormat(),
                    .components =
                        {
                            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                        },
                    .subresourceRange =
                        {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = layerIndex,
                            .layerCount = 1,
                        },
                });
            }
            runtimeTexture.singleLayerViewBaseIndex = 1;
            return image;
        }

        return vk::DeviceLocalImage::create(device, vma, false, allocationInfo.width, allocationInfo.height, 1,
                                            textureConfig.format, allocationInfo.usage);
    };

    for (auto &aliasGroup : aliasGroups) {
        const TextureAllocationInfo &allocationInfo = allocationInfos[aliasGroup.textureIndex];
        uint32_t imageFrameCount = textureFrameCount(shaderPack_.textures[aliasGroup.textureIndex]);
        aliasGroup.frameImages.resize(imageFrameCount);
        for (uint32_t frameIndex = 0; frameIndex < imageFrameCount; frameIndex++) {
            aliasGroup.frameImages[frameIndex] = vk::DeviceLocalImage::create(
                device, vma, false, allocationInfo.width, allocationInfo.height, 1,
                shaderPack_.textures[aliasGroup.textureIndex].format, allocationInfo.usage);
        }
    }

    for (size_t textureIndex = 0; textureIndex < runtimeTextures.size(); textureIndex++) {
        RuntimeTexture &runtimeTexture = runtimeTextures[textureIndex];
        if (runtimeTexture.config.imported) { continue; }
        if (!activeRuntimeResources_.hasTexture(runtimeTexture.config.name)) { continue; }

        size_t aliasGroupIndex = textureAliasGroups[textureIndex];
        if (aliasGroupIndex != unusedAliasGroup) {
            runtimeTexture.sampledViewIndex = 0;
            runtimeTexture.frameImages = aliasGroups[aliasGroupIndex].frameImages;
            continue;
        }

        uint32_t imageFrameCount = textureFrameCount(runtimeTexture.config);
        runtimeTexture.frameImages.resize(imageFrameCount);
        for (uint32_t frameIndex = 0; frameIndex < imageFrameCount; frameIndex++) {
            runtimeTexture.frameImages[frameIndex] =
                createRuntimeFrameImage(runtimeTexture, allocationInfos[textureIndex]);
        }
    }

    runtimeTextures_ = std::move(runtimeTextures);
    runtimeTextureIndices_.clear();
    runtimeTextureIndices_.reserve(runtimeTextures_.size());
    for (size_t i = 0; i < runtimeTextures_.size(); i++) {
        if (!runtimeTextures_[i].config.imported && runtimeTextures_[i].frameImages.empty()) { continue; }
        runtimeTextureIndices_[runtimeTextures_[i].config.name] = i;
    }
}

void ShaderPack::initRuntimeBuffers() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    uint32_t frameCount = framework->swapchain()->imageCount();

    std::vector<RuntimeBuffer> runtimeBuffers(shaderPack_.buffers.size());
    mcvr::parallelFor(shaderPack_.buffers.size(), [&](size_t bufferIndex) {
        const auto &bufferConfig = shaderPack_.buffers[bufferIndex];
        RuntimeBuffer runtimeBuffer;
        runtimeBuffer.config = bufferConfig;
        if (!activeRuntimeResources_.hasBuffer(bufferConfig.name)) {
            runtimeBuffers[bufferIndex] = std::move(runtimeBuffer);
            return;
        }

        uint32_t bufferFrameCount = bufferConfig.shared ? 1 : frameCount;
        runtimeBuffer.frameBuffers.resize(bufferFrameCount);
        const size_t bufferSize =
            std::max(static_cast<size_t>(1),
                     static_cast<size_t>(std::ceil(evaluateNumericExpression(bufferConfig.sizeExpression))));
        for (uint32_t frameIndex = 0; frameIndex < bufferFrameCount; frameIndex++) {
            runtimeBuffer.frameBuffers[frameIndex] =
                vk::DeviceLocalBuffer::create(vma, device, false, bufferSize,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                              0, VMA_MEMORY_USAGE_GPU_ONLY);
        }

        runtimeBuffers[bufferIndex] = std::move(runtimeBuffer);
    });

    runtimeBuffers_ = std::move(runtimeBuffers);
    runtimeBufferIndices_.clear();
    runtimeBufferIndices_.reserve(runtimeBuffers_.size());
    for (size_t i = 0; i < runtimeBuffers_.size(); i++) {
        if (runtimeBuffers_[i].frameBuffers.empty()) { continue; }
        runtimeBufferIndices_[runtimeBuffers_[i].config.name] = i;
    }
}

void ShaderPack::loadRuntimeResources() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    auto physicalDevice = framework->physicalDevice();

    bool hasImportedTextures = false;
    for (const auto &texture : runtimeTextures_) {
        if (texture.config.imported) {
            hasImportedTextures = true;
            break;
        }
    }
    if (!hasImportedTextures) { return; }

    auto readBinaryTexture = [](const fs::path &path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream.is_open()) { throw std::runtime_error("failed to open imported texture: " + path.string()); }

        std::streamsize size = stream.tellg();
        if (size < 0) { throw std::runtime_error("failed to read imported texture size: " + path.string()); }
        stream.seekg(0, std::ios::beg);

        std::vector<uint8_t> data(static_cast<size_t>(size));
        if (size > 0 && !stream.read(reinterpret_cast<char *>(data.data()), size)) {
            throw std::runtime_error("failed to read imported texture data: " + path.string());
        }
        return data;
    };

    mcvr::parallelFor(runtimeTextures_.size(), [&](size_t textureIndex) {
        auto &texture = runtimeTextures_[textureIndex];
        if (!texture.config.imported) { return; }
        if (texture.config.dimension == ShaderPackLoader::TextureDimension::Texture2D) {
            if (texture.config.format != VK_FORMAT_R8G8B8A8_UNORM) {
                throw std::runtime_error("Imported 2d textures currently require R8G8B8A8_UNORM: " +
                                         texture.config.name);
            }

            auto loader =
                vk::ImageLoader::create(std::vector<std::string>{texture.config.sourcePath.string()}, 4, false);
            texture.importedImage = vk::DeviceLocalImage::create(device, vma, true, loader->width(), loader->height(),
                                                                 1, texture.config.format, VK_IMAGE_USAGE_SAMPLED_BIT);
            texture.importedImage->setMemoryCategory(vk::ResourceMemoryCategory::Texture);
            texture.importedImage->uploadToStagingBuffer(loader->data());
            return;
        }

        if (texture.config.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
            const size_t expectedSize =
                static_cast<size_t>(texture.config.importedWidth) * static_cast<size_t>(texture.config.importedHeight) *
                static_cast<size_t>(texture.config.importedDepth) * vk::formatToByte(texture.config.format);
            std::vector<uint8_t> raw = readBinaryTexture(texture.config.sourcePath);
            if (raw.size() != expectedSize) {
                throw std::runtime_error("imported 2d_array texture size mismatch for " + texture.config.name +
                                         ": expected " + std::to_string(expectedSize) + " bytes but got " +
                                         std::to_string(raw.size()));
            }

            texture.importedImage = vk::DeviceLocalImage::create(
                device, vma, true, texture.config.importedWidth, texture.config.importedHeight,
                texture.config.importedDepth, texture.config.format, VK_IMAGE_USAGE_SAMPLED_BIT);
            texture.importedImage->setMemoryCategory(vk::ResourceMemoryCategory::Texture);
            texture.importedImage->uploadToStagingBuffer(raw.data());
            return;
        }

        if (texture.config.dimension != ShaderPackLoader::TextureDimension::Texture3D) {
            throw std::runtime_error("unsupported imported texture dimension: " + texture.config.name);
        }

        const size_t expectedSize =
            static_cast<size_t>(texture.config.importedWidth) * static_cast<size_t>(texture.config.importedHeight) *
            static_cast<size_t>(texture.config.importedDepth) * vk::formatToByte(texture.config.format);
        std::vector<uint8_t> raw = readBinaryTexture(texture.config.sourcePath);
        if (raw.size() != expectedSize) {
            throw std::runtime_error("imported 3d texture size mismatch for " + texture.config.name + ": expected " +
                                     std::to_string(expectedSize) + " bytes but got " + std::to_string(raw.size()));
        }

        texture.importedImage = vk::DeviceLocalImage::create(
            device, vma, true, texture.config.importedWidth, texture.config.importedHeight,
            texture.config.importedDepth, 1, texture.config.format, VK_IMAGE_USAGE_SAMPLED_BIT);
        texture.importedImage->setMemoryCategory(vk::ResourceMemoryCategory::Texture);
        texture.importedImage->uploadToStagingBuffer(raw.data());
    });

    auto commandPool = vk::CommandPool::create(physicalDevice, device);
    auto commandBuffer = vk::CommandBuffer::create(device, commandPool);
    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    bool hasUploads = false;
    for (auto &texture : runtimeTextures_) {
        if (!texture.config.imported) { continue; }

        commandBuffer->barriersBufferImage({}, {{
                                                   .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                                   .srcAccessMask = 0,
                                                   .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                   .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                   .oldLayout = texture.importedImage->imageLayout(),
                                                   .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                   .srcQueueFamilyIndex = physicalDevice->mainQueueIndex(),
                                                   .dstQueueFamilyIndex = physicalDevice->mainQueueIndex(),
                                                   .image = texture.importedImage,
                                                   .subresourceRange = texture.importedImage->fullSubresourceRange(),
                                               }});
        texture.importedImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        texture.importedImage->uploadToImage(commandBuffer);
        commandBuffer->barriersBufferImage({}, {{
                                                   .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                   .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                   .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                   .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                   .oldLayout = texture.importedImage->imageLayout(),
                                                   .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                   .srcQueueFamilyIndex = physicalDevice->mainQueueIndex(),
                                                   .dstQueueFamilyIndex = physicalDevice->mainQueueIndex(),
                                                   .image = texture.importedImage,
                                                   .subresourceRange = texture.importedImage->fullSubresourceRange(),
                                               }});
        texture.importedImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        hasUploads = true;
    }

    commandBuffer->end();
    if (hasUploads) {
        commandBuffer->submitMainQueueIndividual(device);
        std::unique_lock<std::recursive_mutex> queueLock(device->mainQueueMutex());
        vk::requireSuccess(vkQueueWaitIdle(device->mainVkQueue()), "vkQueueWaitIdle(shader-pack texture upload)");
    }
}
