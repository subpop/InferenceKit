// Copyright 2026 Link Dupont
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

/// The Anthropic Messages API: `POST /v1/messages`, streamed over SSE.
public struct AnthropicMessagesFormat: WireFormat {
    public let id = WireFormatID.anthropicMessages
    public let transport = WireTransport.sse

    public init() {}

    public func makeChatRequest(
        _ request: InferenceRequest,
        provider: ProviderConfiguration,
        model: ModelConfiguration
    ) throws -> URLRequest {
        // Anthropic requires max_tokens; fall back to the model configuration.
        guard let maxTokens = request.options.maxOutputTokens ?? model.maxOutputTokens else {
            throw InferenceError.invalidConfiguration(
                "Anthropic requires maxOutputTokens. Set it in GenerationOptions or the model configuration."
            )
        }

        var thinkingField: AnthropicThinking?
        var outputConfig: AnthropicOutputConfig?
        if let thinking = request.options.thinking {
            switch thinking {
            case .disabled:
                // Thinking is off by default; omitting the parameter is the
                // only universally accepted way to leave it off.
                thinkingField = nil
            case .automatic:
                thinkingField = AnthropicThinking(type: "adaptive")
            case .budgetedTokens(let budget):
                thinkingField = AnthropicThinking(type: "enabled", budget_tokens: budget)
            case .effort(let effort):
                // Adaptive-thinking models scale their depth with an effort;
                // Anthropic's scale starts at low and has no "none".
                guard effort != .none else {
                    throw InferenceError.unsupportedFeature(
                        "Reasoning effort \"none\"",
                        context: id.rawValue
                    )
                }
                thinkingField = AnthropicThinking(type: "adaptive")
                outputConfig = AnthropicOutputConfig(effort: effort.rawValue)
            }
        }

        let wireMessages = try encodeMessages(request.messages)

        var body = AnthropicRequestBody(
            model: model.wireID,
            max_tokens: maxTokens,
            system: request.system.isEmpty
                ? nil
                : request.system.map { AnthropicTextBlock(text: $0) },
            messages: wireMessages,
            stream: true,
            temperature: request.options.temperature,
            top_p: request.options.topP,
            stop_sequences: request.options.stopSequences,
            thinking: thinkingField,
            output_config: outputConfig,
            tools: try request.tools.map(encodeTool),
            context_management: request.options.compaction.map {
                AnthropicContextManagement(edits: [
                    .init(trigger: .init(value: $0.triggerThresholdTokens))
                ])
            }
        )
        // Extended thinking requires temperature and top_p to be unset.
        if thinkingField != nil {
            body.temperature = nil
            body.top_p = nil
        }

        var urlRequest = URLRequest(url: provider.baseURL.appending(path: "messages"))
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = try JSONEncoder().encode(body)
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        return urlRequest
    }

    public func makeStreamDecoder() -> any WireStreamDecoder {
        AnthropicStreamDecoder()
    }

    public func listModels(
        provider: ProviderConfiguration,
        perform: RequestPerformer
    ) async throws -> [DiscoveredModel] {
        var models: [DiscoveredModel] = []
        var afterID: String?
        while true {
            let url = provider.baseURL.appending(path: "models")
            guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
                throw InferenceError.invalidConfiguration("Invalid models URL for provider \"\(provider.id)\".")
            }
            var queryItems = [URLQueryItem(name: "limit", value: "100")]
            if let afterID {
                queryItems.append(URLQueryItem(name: "after_id", value: afterID))
            }
            components.queryItems = queryItems
            guard let pageURL = components.url else {
                throw InferenceError.invalidConfiguration("Invalid models URL for provider \"\(provider.id)\".")
            }

            var urlRequest = URLRequest(url: pageURL)
            urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
            let (data, http) = try await perform(Transport.finalized(urlRequest, provider: provider))
            try Transport.check(http, data: data, provider: provider, format: self)

            let page = try decodeDTO(AnthropicModelsPage.self, from: data)
            models += page.data.map { DiscoveredModel(id: $0.id, displayName: $0.display_name) }

            if page.has_more == true, let lastID = page.last_id {
                afterID = lastID
            } else {
                return models
            }
        }
    }

    // MARK: - Encoding

    private func encodeTool(_ tool: ToolDefinition) throws -> AnthropicTool {
        guard tool.parametersSchema.objectValue != nil else {
            throw InferenceError.invalidConfiguration(
                "Tool \"\(tool.name)\" schema must be a JSON object."
            )
        }
        return AnthropicTool(
            name: tool.name,
            description: tool.description,
            input_schema: tool.parametersSchema
        )
    }

    /// Folds canonical messages into wire messages: tool-result messages
    /// merge into a single user message with `tool_result` blocks, as the
    /// Messages API requires.
    private func encodeMessages(_ messages: [Message]) throws -> [AnthropicWireMessage] {
        var wireMessages: [AnthropicWireMessage] = []
        var pendingToolResults: [JSONValue] = []

        func flushToolResults() {
            guard !pendingToolResults.isEmpty else { return }
            wireMessages.append(AnthropicWireMessage(role: "user", content: pendingToolResults))
            pendingToolResults = []
        }

        for message in messages {
            switch message.role {
            case .user:
                flushToolResults()
                var blocks: [JSONValue] = []
                for block in message.content {
                    blocks.append(try encodeUserBlock(block))
                }
                wireMessages.append(AnthropicWireMessage(role: "user", content: blocks))

            case .assistant:
                flushToolResults()
                var blocks: [JSONValue] = []
                for block in message.content {
                    if let encoded = try encodeAssistantBlock(block) {
                        blocks.append(encoded)
                    }
                }
                if !blocks.isEmpty {
                    wireMessages.append(AnthropicWireMessage(role: "assistant", content: blocks))
                }

            case .tool:
                for block in message.content {
                    guard case .toolResult(let result) = block else {
                        throw InferenceError.invalidConfiguration(
                            "Tool messages may only contain tool results."
                        )
                    }
                    pendingToolResults.append(encodeToolResult(result))
                }
            }
        }
        flushToolResults()
        return wireMessages
    }

    private func encodeUserBlock(_ block: ContentBlock) throws -> JSONValue {
        switch block {
        case .text(let text):
            return .object(["type": .string("text"), "text": .string(text)])

        case .image(let image):
            let source: JSONValue
            if let data = image.data {
                source = .object([
                    "type": .string("base64"),
                    "media_type": .string(image.mediaType),
                    "data": .string(data.base64EncodedString()),
                ])
            } else if let url = image.url {
                source = .object(["type": .string("url"), "url": .string(url.absoluteString)])
            } else {
                throw InferenceError.invalidConfiguration("Image block has neither data nor url.")
            }
            return .object(["type": .string("image"), "source": source])

        case .file(let file):
            if file.isPDF {
                return .object([
                    "type": .string("document"),
                    "source": .object([
                        "type": .string("base64"),
                        "media_type": .string(file.mediaType),
                        "data": .string(file.data.base64EncodedString()),
                    ]),
                ])
            }
            if file.isText, let text = file.textContent {
                return .object([
                    "type": .string("text"),
                    "text": .string("[\(file.filename)]:\n\(text)"),
                ])
            }
            throw InferenceError.unsupportedFeature(
                "Non-PDF binary file attachments",
                context: id.rawValue
            )

        case .toolResult(let result):
            return encodeToolResult(result)

        case .thinking:
            throw InferenceError.invalidConfiguration(
                "Thinking blocks are only valid in assistant messages."
            )

        case .toolCall:
            throw InferenceError.invalidConfiguration(
                "Tool calls are only valid in assistant messages."
            )
        }
    }

    private func encodeAssistantBlock(_ block: ContentBlock) throws -> JSONValue? {
        switch block {
        case .text(let text):
            return .object(["type": .string("text"), "text": .string(text)])

        case .thinking(let thinking):
            var block: [String: JSONValue] = [
                "type": .string("thinking"),
                "thinking": .string(thinking.text),
            ]
            if let signature = thinking.signature {
                block["signature"] = .string(signature)
            }
            return .object(block)

        case .toolCall(let call):
            let input = parseToolArguments(call.arguments)
            return .object([
                "type": .string("tool_use"),
                "id": .string(call.id),
                "name": .string(call.name),
                "input": input,
            ])

        case .toolResult:
            throw InferenceError.invalidConfiguration(
                "Tool results belong in tool or user messages."
            )

        case .image, .file:
            throw InferenceError.unsupportedFeature("Attachments in assistant messages", context: id.rawValue)
        }
    }

    private func encodeToolResult(_ result: ToolResultBlock) -> JSONValue {
        var block: [String: JSONValue] = [
            "type": .string("tool_result"),
            "tool_use_id": .string(result.callID),
            "content": .string(result.output),
        ]
        if result.isError {
            block["is_error"] = .bool(true)
        }
        return .object(block)
    }

    private func parseToolArguments(_ arguments: String) -> JSONValue {
        (try? JSONValue(parsing: arguments)) ?? .object([:])
    }
}
