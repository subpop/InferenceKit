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

/// The OpenAI Chat Completions API: `POST /v1/chat/completions`, streamed
/// over SSE. Also the wire format spoken by most OpenAI-compatible services.
///
/// Reasoning content emitted by compatible providers (`reasoning_content` or
/// `reasoning` deltas) surfaces as thinking deltas, but cannot be resent in
/// history; thinking blocks are dropped when encoding assistant messages.
public struct OpenAIChatCompletionsFormat: WireFormat {
    public let id = WireFormatID.openAIChatCompletions
    public let transport = WireTransport.sse

    public init() {}

    public func makeChatRequest(
        _ request: InferenceRequest,
        provider: ProviderConfiguration,
        model: ModelConfiguration
    ) throws -> URLRequest {
        if case .budgetedTokens = request.options.thinking {
            throw InferenceError.unsupportedFeature("Reasoning token budgets", context: id.rawValue)
        }

        var reasoningEffort: String?
        if let thinking = request.options.thinking {
            switch thinking {
            case .effort(let effort):
                reasoningEffort = effort.rawValue
            case .automatic, .disabled:
                reasoningEffort = nil
            case .budgetedTokens:
                throw InferenceError.unsupportedFeature("Reasoning token budgets", context: id.rawValue)
            }
        }

        var messages: [JSONValue] = []
        if !request.system.isEmpty {
            messages.append(.object([
                "role": .string("system"),
                "content": .string(request.system.joinedPrompt),
            ]))
        }
        messages += try encodeMessages(request.messages)

        let body = OpenAIChatCompletionsBody(
            model: model.wireID,
            messages: messages,
            stream: true,
            stream_options: .init(),
            temperature: request.options.temperature,
            top_p: request.options.topP,
            max_completion_tokens: request.options.maxOutputTokens ?? model.maxOutputTokens,
            stop: request.options.stopSequences,
            tools: request.tools.isEmpty ? nil : try request.tools.map(encodeTool),
            reasoning_effort: reasoningEffort
        )

        var urlRequest = URLRequest(url: provider.baseURL.appending(path: "chat/completions"))
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = try JSONEncoder().encode(body)
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return urlRequest
    }

    public func makeStreamDecoder() -> any WireStreamDecoder {
        OpenAIChatCompletionsStreamDecoder()
    }

    public func listModels(
        provider: ProviderConfiguration,
        perform: RequestPerformer
    ) async throws -> [DiscoveredModel] {
        let urlRequest = URLRequest(url: provider.baseURL.appending(path: "models"))
        let (data, http) = try await perform(Transport.finalized(urlRequest, provider: provider))
        try Transport.check(http, data: data, provider: provider, format: self)
        let page = try decodeDTO(OpenAIModelsPage.self, from: data)
        return page.data.map { DiscoveredModel(id: $0.id) }
    }

    // MARK: - Encoding

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

    private func encodeMessage(_ message: Message) throws -> JSONValue {
        switch message.role {
        case .user:
            var parts: [JSONValue] = []
            var plainText = ""
            var hasAttachments = false
            for block in message.content {
                switch block {
                case .text(let text):
                    if hasAttachments {
                        parts.append(.object(["type": .string("text"), "text": .string(text)]))
                    } else {
                        plainText += text
                    }
                case .image(let image):
                    hasAttachments = true
                    let url: String
                    if let data = image.data {
                        url = "data:\(image.mediaType);base64,\(data.base64EncodedString())"
                    } else if let remote = image.url {
                        url = remote.absoluteString
                    } else {
                        throw InferenceError.invalidConfiguration("Image block has neither data nor url.")
                    }
                    parts.append(.object([
                        "type": .string("image_url"),
                        "image_url": .object(["url": .string(url)]),
                    ]))
                case .file(let file):
                    if file.isPDF {
                        hasAttachments = true
                        let dataURL = "data:\(file.mediaType);base64,\(file.data.base64EncodedString())"
                        var fileObject: [String: JSONValue] = ["file_data": .string(dataURL)]
                        if !file.filename.isEmpty {
                            fileObject["filename"] = .string(file.filename)
                        }
                        parts.append(.object(["type": .string("file"), "file": .object(fileObject)]))
                    } else if file.isText, let text = file.textContent {
                        if hasAttachments {
                            parts.append(.object(["type": .string("text"), "text": .string("[\(file.filename)]:\n\(text)")]))
                        } else {
                            plainText += "[\(file.filename)]:\n\(text)"
                        }
                    } else {
                        throw InferenceError.unsupportedFeature(
                            "Non-PDF binary file attachments",
                            context: id.rawValue
                        )
                    }
                case .thinking, .toolCall, .toolResult:
                    throw InferenceError.invalidConfiguration(
                        "Block is not valid in a user message."
                    )
                }
            }

            if hasAttachments {
                if !plainText.isEmpty {
                    parts.append(.object(["type": .string("text"), "text": .string(plainText)]))
                }
                return .object([
                    "role": .string("user"),
                    "content": .array(parts),
                ])
            }
            return .object([
                "role": .string("user"),
                "content": .string(plainText),
            ])

        case .assistant:
            var text = ""
            var toolCalls: [JSONValue] = []
            for block in message.content {
                switch block {
                case .text(let value):
                    text += value
                case .thinking:
                    // Reasoning content cannot be resent to Chat Completions.
                    continue
                case .toolCall(let call):
                    toolCalls.append(.object([
                        "id": .string(call.id),
                        "type": .string("function"),
                        "function": .object([
                            "name": .string(call.name),
                            "arguments": .string(call.arguments),
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
            var message: [String: JSONValue] = ["role": .string("assistant")]
            message["content"] = text.isEmpty ? .null : .string(text)
            if !toolCalls.isEmpty {
                message["tool_calls"] = .array(toolCalls)
            }
            return .object(message)

        case .tool:
            // Handled by `encodeMessages`, which splits tool messages.
            throw InferenceError.invalidConfiguration(
                "Tool messages must be encoded via encodeMessages."
            )
        }
    }

    private func encodeToolResult(_ result: ToolResultBlock) -> JSONValue {
        .object([
            "role": .string("tool"),
            "tool_call_id": .string(result.callID),
            "content": .string(result.output),
        ])
    }
}

extension OpenAIChatCompletionsFormat {
    /// Splits canonical tool messages into one wire message per result.
    private func encodeMessages(_ messages: [Message]) throws -> [JSONValue] {
        var encoded: [JSONValue] = []
        for message in messages {
            if message.role == .tool {
                for block in message.content {
                    guard case .toolResult(let result) = block else {
                        throw InferenceError.invalidConfiguration(
                            "Tool messages may only contain tool results."
                        )
                    }
                    encoded.append(encodeToolResult(result))
                }
            } else {
                encoded.append(try encodeMessage(message))
            }
        }
        return encoded
    }
}
