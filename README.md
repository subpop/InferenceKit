# InferenceKit

Multi-provider LLM inference for Swift apps: one stateless, canonical API over
remote APIs (Anthropic, OpenAI, Google Gemini, Ollama) and local inference
(MLX, Apple Foundation Models).

- **Five wire formats.** Anthropic Messages, OpenAI Responses, OpenAI Chat
  Completions, Gemini `generateContent`, and Ollama's first-party `/api/chat`.
  Any provider compatible with one of those formats works through the same
  client — including custom endpoints and gateways.
- **Canonical content model.** Text, images, files, thinking blocks with
  provider round-trip signatures, tool calls, and tool results all share one
  `Message` shape. History is caller-owned; engines are stateless.
- **Model awareness.** A bundled catalog of known models (context windows,
  thinking budgets and efforts, input modalities) merges with live model
  discovery, and user-defined models slot into the same type.
- **Local inference.** MLX on Apple Silicon and Apple's on-device Foundation
  Models run behind the same `InferenceEngine` protocol as remote providers.

## Modules

| Product | Purpose | Platforms |
| --- | --- | --- |
| `InferenceKit` | Core: canonical model, provider/wire-format codecs, remote engine, catalog. Zero dependencies. | macOS 14+, iOS 17+ |
| `InferenceKitMLX` | Local inference with [MLX](https://github.com/ml-explore/mlx-swift-lm) (Apple Silicon only). Adds `mlx-swift-lm` and friends. | macOS 14+, iOS 17+ |
| `InferenceKitFoundationModels` | Local inference with Apple's on-device Foundation Models (Apple Intelligence). | macOS 26+, iOS 26+ |

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/subpop/InferenceKit", from: "1.0.0"),
]
// and in your target:
.product(name: "InferenceKit", package: "InferenceKit")
```

## Quick start

```swift
import InferenceKit

let client = InferenceClient(
    provider: .anthropic(apiKey: apiKey),
    model: .claudeSonnet5_5
)

let request = InferenceRequest(
    system: ["You are a concise assistant."],
    messages: [.user("Explain MLX in one sentence.")]
)

for try await event in client.stream(request) {
    if case .textDelta(let text) = event {
        print(text, terminator: "")
    }
}
```

For one-shot calls:

```swift
let completion = try await client.complete(request)
print(completion.assistantMessage?.content.compactMap(\.text) ?? "")
print(completion.usage ?? "")
```

## Providers

Presets cover the first-class providers. Each picks the endpoint, wire format,
and authentication scheme:

| Preset | Wire format | Auth |
| --- | --- | --- |
| `.anthropic(apiKey:)` | Anthropic Messages | `x-api-key` |
| `.openAI(apiKey:)` | OpenAI Responses | `Authorization: Bearer` |
| `.openAIChatCompletions(apiKey:)` | OpenAI Chat Completions | `Authorization: Bearer` |
| `.gemini(apiKey:)` | Gemini `generateContent` | `x-goog-api-key` |
| `.ollama()` | Ollama `/api/chat` | none |

Any endpoint compatible with a wire format is a custom provider:

```swift
let gateway = ProviderConfiguration(
    id: "my-gateway",
    displayName: "Internal Gateway",
    baseURL: URL(string: "https://llm.internal/v1")!,
    wireFormatID: .openAIChatCompletions,
    authentication: .bearer(token),
    defaultHeaders: ["X-Team": "platform"]
)
let client = InferenceClient(provider: gateway, model: .custom("my-model"))
```

## Models

Three sources of `ModelConfiguration`, all interchangeable:

```swift
// 1. Constants for flagship models (backed by the bundled catalog).
let sonnet = ModelConfiguration.claudeSonnet4_5

// 2. Live discovery, merged with bundled catalog capabilities
//    (context window, thinking support, modalities).
let models = try await InferenceClient.discoverModels(provider: .openAI(apiKey: apiKey))
let model = models.first { $0.id == "gpt-5.1" } ?? .custom("gpt-5.1")

// 3. Your own definitions for custom or fine-tuned models.
let finetune = ModelConfiguration.custom(
    "my-finetune",
    providerModelID: "ft:abc123",
    contextWindowTokens: 16_384,
    thinking: ThinkingCapability(supportedEfforts: [.low, .high])
)
```

The bundled catalog resolves dated wire ids (`claude-sonnet-4-5-20250929`) to
their canonical entries while preserving the discovered id for the wire.
Unknown ids resolve to permissive configurations with unknown capabilities.

## Streaming and events

`client.stream(_:)` yields `StreamEvent`s; every stream ends with
`.completed` or an error:

```swift
for try await event in client.stream(request) {
    switch event {
    case .thinkingDelta(let text): break   // incremental reasoning text
    case .textDelta(let text):    break    // incremental response text
    case .toolCall(let call):     break    // a completed tool call
    case .usage(let usage):       break    // token counts
    case .completed(let completion): break // finish reason, usage, and the
                                           // reconstructed assistant turn
    }
}
```

`Completion.assistantMessage` is the full assistant turn — including thinking
signatures and tool-call ids — ready to append to history.

## Tool calling

```swift
let weatherTool = try ToolDefinition(
    name: "get_weather",
    description: "Get the current weather for a city.",
    parametersJSON: """
    {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name"}
      },
      "required": ["city"]
    }
    """
)

var history: [Message] = [.user("What's the weather in Cupertino?")]
let request = InferenceRequest(
    messages: history,
    tools: [weatherTool],
    options: GenerationOptions(maxOutputTokens: 1024)
)

let completion = try await client.complete(request)

// The reconstructed assistant turn — thinking signatures and tool-call ids
// included — becomes part of the history.
guard let assistantTurn = completion.assistantMessage else { return }
history.append(assistantTurn)

// Run the calls the model asked for and feed results back. The full history
// travels with every request; engines are stateless.
for block in assistantTurn.content {
    if case .toolCall(let call) = block {
        let args = try JSONDecoder().decode([String: String].self, from: Data(call.arguments.utf8))
        let result = getWeather(city: args["city"] ?? "")
        history.append(.toolResult(callID: call.id, name: call.name, output: result))
    }
}
let final = try await client.complete(InferenceRequest(messages: history, tools: [weatherTool]))
```

## Thinking and reasoning

`GenerationOptions.thinking` configures reasoning uniformly; each wire format
maps what it can and throws `InferenceError.unsupportedFeature` for what it
cannot:

```swift
GenerationOptions(thinking: .disabled)               // off
GenerationOptions(thinking: .automatic)              // provider default
GenerationOptions(thinking: .budgetedTokens(8_192))  // Anthropic (4.5 era), Gemini, Ollama
GenerationOptions(thinking: .effort(.high))          // OpenAI, Anthropic (5.5 era)
GenerationOptions(thinking: .effort(.none))          // OpenAI reasoning off (Sol, Luna)
```

Budgets and efforts validate against the model's known capabilities before
anything hits the network: budget-era models (Claude 4.5, Gemini) reject
efforts, effort-era models (Claude 5.5, GPT-6) reject budgets, and
always-thinking models (Claude Opus 5.5, Sonnet 5.5) reject `.disabled`.
Effort `none` is model-specific — GPT-6 Sol and Luna accept it; Astra and
6.1 Sol reject it. Thinking blocks returned in history carry an opaque
`signature` that providers require for multi-turn tool use — persist it.

## Local inference

### MLX (Apple Silicon)

```swift
import InferenceKitMLX

// Downloads from Hugging Face on first use (cached afterwards):
let engine = try await MLXEngine.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
let client = InferenceClient(engine: engine)

for try await event in client.stream(InferenceRequest(messages: [.user("Hello!")])) {
    if case .textDelta(let text) = event { print(text, terminator: "") }
}
```

Text and tool calling are supported. Vision models, stop sequences, and
reasoning budgets/efforts are rejected with `InferenceError.unsupportedFeature`
in v1. Already-loaded containers (e.g. from `MLXVLM` or a custom factory) can
be wrapped with `MLXEngine(container:model:)`.

### Foundation Models (Apple Intelligence)

```swift
import InferenceKitFoundationModels

let engine = FoundationModelsEngine()
let client = InferenceClient(engine: engine)
```

Requires macOS 26 / iOS 26 and an available Apple Intelligence model. Sessions
are created per request; text content only (tools and attachments are not
bridged to Foundation Models' Swift-typed tool system in v1).

## Design notes

- **Stateless core.** Every call carries the full conversation; no hidden
  session state to keep in sync across providers.
- **One remote engine, many codecs.** `RemoteEngine` is generic over a
  `WireFormat` (SSE and newline-delimited JSON transports), with one codec per
  wire format. New formats plug in without touching the engine.
- **Errors, not option soup.** A field a wire format cannot express either
  throws (`InferenceError.unsupportedFeature`) or is silently advisory
  (`contextWindowTokens` for hosted APIs).

## Testing

```sh
swift test
```

Codec golden tests, engine end-to-end tests (stubbed HTTP transport), catalog
tests, and live Foundation Models tests run by default. The MLX smoke test is
opt-in because it downloads model weights (hundreds of MB on first run):

```sh
INFERENCEKIT_MLX_SMOKE=1 swift test --filter MLXEngineTests
# optionally: INFERENCEKIT_MLX_MODEL=mlx-community/Qwen3-4B-4bit
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
