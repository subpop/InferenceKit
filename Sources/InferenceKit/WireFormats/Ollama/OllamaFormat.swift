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

/// Ollama's first-party chat API: `POST /api/chat`, streamed as
/// newline-delimited JSON. Model listing uses `GET /api/tags`.
public struct OllamaFormat: WireFormat {
    public let id = WireFormatID.ollama
    public let transport = WireTransport.ndjson

    public init() {}

    public func makeChatRequest(
        _ request: InferenceRequest,
        provider: ProviderConfiguration,
        model: ModelConfiguration
    ) throws -> URLRequest {
        if case .effort = request.options.thinking {
            throw InferenceError.unsupportedFeature("Reasoning effort", context: id.rawValue)
        }

        var think: OllamaThink?
        if let thinking = request.options.thinking {
            switch thinking {
            case .disabled:
                think = .enabled(false)
            case .automatic:
                think = .enabled(true)
            case .budgetedTokens(let budget):
                think = .budget(budget)
            case .effort:
                throw InferenceError.unsupportedFeature("Reasoning effort", context: id.rawValue)
            }
        }

        var messages: [OllamaWireMessage] = []
        if !request.system.isEmpty {
            messages.append(OllamaWireMessage(
                role: "system",
                content: request.system.joinedPrompt
            ))
        }
        for message in request.messages {
            switch message.role {
            case .user:
                var images: [String] = []
                var text = ""
                for block in message.content {
                    switch block {
                    case .text(let value):
                        text += value
                    case .image(let image):
                        if let data = image.data {
                            images.append(data.base64EncodedString())
                        } else if let url = image.url, let data = try? Data(contentsOf: url) {
                            images.append(data.base64EncodedString())
                        }
                    case .file(let file):
                        if file.isText, let content = file.textContent {
                            text += "\n\n[\(file.filename)]:\n\(content)"
                        } else {
                            text += "\n\n[Attached file: \(file.filename) — not supported by this model]"
                        }
                    case .thinking, .toolCall, .toolResult:
                        throw InferenceError.invalidConfiguration(
                            "Block is not valid in a user message."
                        )
                    }
                }
                messages.append(OllamaWireMessage(
                    role: "user",
                    content: text,
                    images: images.isEmpty ? nil : images
                ))

            case .assistant:
                var text = ""
                var thinkingText: String?
                var toolCalls: [JSONValue] = []
                for block in message.content {
                    switch block {
                    case .text(let value):
                        text += value
                    case .thinking(let thinking):
                        thinkingText = thinking.text
                    case .toolCall(let call):
                        toolCalls.append(.object([
                            "function": .object([
                                "name": .string(call.name),
                                "arguments": (try? JSONValue(parsing: call.arguments))
                                    ?? .object([:]),
                            ]),
                        ]))
                    case .image, .file:
                        throw InferenceError.unsupportedFeature(
                            "Attachments in assistant messages",
                            context: id.rawValue
                        )
                    case .toolResult:
                        throw InferenceError.invalidConfiguration(
                            "Tool results belong in tool messages."
                        )
                    }
                }
                messages.append(OllamaWireMessage(
                    role: "assistant",
                    content: text.isEmpty ? nil : text,
                    thinking: thinkingText,
                    tool_calls: toolCalls.isEmpty ? nil : toolCalls
                ))

            case .tool:
                for block in message.content {
                    guard case .toolResult(let result) = block else {
                        throw InferenceError.invalidConfiguration(
                            "Tool messages may only contain tool results."
                        )
                    }
                    messages.append(OllamaWireMessage(
                        role: "tool",
                        content: result.output,
                        tool_name: result.name
                    ))
                }
            }
        }

        let body = OllamaChatBody(
            model: model.wireID,
            messages: messages,
            stream: true,
            think: think,
            tools: request.tools.isEmpty ? nil : try request.tools.map(encodeTool),
            options: .init(
                num_ctx: request.options.contextWindowTokens,
                temperature: request.options.temperature,
                top_p: request.options.topP,
                stop: request.options.stopSequences
            )
        )

        var urlRequest = URLRequest(url: provider.baseURL.appending(path: "api").appending(path: "chat"))
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = try JSONEncoder().encode(body)
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return urlRequest
    }

    public func makeStreamDecoder() -> any WireStreamDecoder {
        OllamaStreamDecoder()
    }

    public func listModels(
        provider: ProviderConfiguration,
        perform: RequestPerformer
    ) async throws -> [DiscoveredModel] {
        let url = provider.baseURL.appending(path: "api").appending(path: "tags")
        let urlRequest = URLRequest(url: url)
        let (data, http) = try await perform(Transport.finalized(urlRequest, provider: provider))
        try Transport.check(http, data: data, provider: provider, format: self)
        let tags = try decodeDTO(OllamaTagsResponse.self, from: data)
        return (tags.models ?? []).compactMap { item in
            guard let name = item.name ?? item.model else { return nil }
            return DiscoveredModel(id: name)
        }
    }

    private func encodeTool(_ tool: ToolDefinition) throws -> JSONValue {
        guard tool.parametersSchema.objectValue != nil else {
            throw InferenceError.invalidConfiguration(
                "Tool \"\(tool.name)\" schema must be a JSON object."
            )
        }
        return .object([
            "type": .string("function"),
            "function": .object([
                "name": .string(tool.name),
                "description": .string(tool.description),
                "parameters": tool.parametersSchema,
            ]),
        ])
    }
}
