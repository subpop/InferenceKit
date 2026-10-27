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

/// The OpenAI Responses API: `POST /v1/responses`, streamed over SSE.
public struct OpenAIResponsesFormat: WireFormat {
    public let id = WireFormatID.openAIResponses
    public let transport = WireTransport.sse

    public init() {}

    public func makeChatRequest(
        _ request: InferenceRequest,
        provider: ProviderConfiguration,
        model: ModelConfiguration
    ) throws -> URLRequest {
        if request.options.stopSequences != nil {
            throw InferenceError.unsupportedFeature("Stop sequences", context: id.rawValue)
        }
        if case .budgetedTokens = request.options.thinking {
            throw InferenceError.unsupportedFeature("Reasoning token budgets", context: id.rawValue)
        }

        var reasoning: OpenAIResponsesBody.Reasoning?
        if let thinking = request.options.thinking {
            switch thinking {
            case .disabled:
                reasoning = nil
            case .automatic:
                reasoning = .init(effort: nil, summary: "auto")
            case .effort(let effort):
                reasoning = .init(effort: effort.rawValue, summary: "auto")
            case .budgetedTokens:
                throw InferenceError.unsupportedFeature("Reasoning token budgets", context: id.rawValue)
            }
        }

        let body = OpenAIResponsesBody(
            model: model.wireID,
            instructions: request.system.isEmpty ? nil : request.system.joinedPrompt,
            input: try encodeInput(request.messages),
            stream: true,
            temperature: request.options.temperature,
            top_p: request.options.topP,
            max_output_tokens: request.options.maxOutputTokens ?? model.maxOutputTokens,
            tools: request.tools.isEmpty ? nil : try request.tools.map(encodeTool),
            reasoning: reasoning,
            context_management: request.options.compaction.map {
                [.init(compact_threshold: $0.triggerThresholdTokens)]
            }
        )

        var urlRequest = URLRequest(url: provider.baseURL.appending(path: "responses"))
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = try JSONEncoder().encode(body)
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return urlRequest
    }

    public func makeStreamDecoder() -> any WireStreamDecoder {
        OpenAIResponsesStreamDecoder()
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
            "name": .string(tool.name),
            "description": .string(tool.description),
            "parameters": tool.parametersSchema,
        ])
    }

    private func encodeInput(_ messages: [Message]) throws -> [JSONValue] {
        var items: [JSONValue] = []
        for message in messages {
            switch message.role {
            case .user:
                items.append(try encodeUserMessage(message))

            case .assistant:
                for block in message.content {
                    if let item = try encodeAssistantBlock(block) {
                        items.append(item)
                    }
                }

            case .tool:
                for block in message.content {
                    guard case .toolResult(let result) = block else {
                        throw InferenceError.invalidConfiguration(
                            "Tool messages may only contain tool results."
                        )
                    }
                    items.append(.object([
                        "type": .string("function_call_output"),
                        "call_id": .string(result.callID),
                        "output": .string(result.output),
                    ]))
                }
            }
        }
        return items
    }

    private func encodeUserMessage(_ message: Message) throws -> JSONValue {
        var parts: [JSONValue] = []
        for block in message.content {
            switch block {
            case .text(let text):
                parts.append(.object(["type": .string("input_text"), "text": .string(text)]))

            case .image(let image):
                if let data = image.data {
                    let dataURL = "data:\(image.mediaType);base64,\(data.base64EncodedString())"
                    parts.append(.object([
                        "type": .string("input_image"),
                        "image_url": .string(dataURL),
                    ]))
                } else if let url = image.url {
                    parts.append(.object([
                        "type": .string("input_image"),
                        "image_url": .string(url.absoluteString),
                    ]))
                }

            case .file(let file):
                if file.isPDF {
                    let dataURL = "data:\(file.mediaType);base64,\(file.data.base64EncodedString())"
                    var part: [String: JSONValue] = [
                        "type": .string("input_file"),
                        "file_data": .string(dataURL),
                    ]
                    if !file.filename.isEmpty {
                        part["filename"] = .string(file.filename)
                    }
                    parts.append(.object(part))
                } else if file.isText, let text = file.textContent {
                    parts.append(.object([
                        "type": .string("input_text"),
                        "text": .string("[\(file.filename)]:\n\(text)"),
                    ]))
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
        return .object([
            "type": .string("message"),
            "role": .string("user"),
            "content": .array(parts),
        ])
    }

    private func encodeAssistantBlock(_ block: ContentBlock) throws -> JSONValue? {
        switch block {
        case .thinking(let thinking):
            // The signature stores the provider reasoning item id, which must
            // be resent for tool-use loops on reasoning models.
            var item: [String: JSONValue] = [
                "type": .string("reasoning"),
                "summary": .array([
                    .object(["type": .string("summary_text"), "text": .string(thinking.text)])
                ]),
            ]
            if let signature = thinking.signature {
                item["id"] = .string(signature)
            }
            return .object(item)

        case .text(let text):
            return .object([
                "type": .string("message"),
                "role": .string("assistant"),
                "content": .array([
                    .object(["type": .string("output_text"), "text": .string(text)])
                ]),
            ])

        case .toolCall(let call):
            return .object([
                "type": .string("function_call"),
                "call_id": .string(call.id),
                "name": .string(call.name),
                "arguments": .string(call.arguments),
            ])

        case .toolResult:
            throw InferenceError.invalidConfiguration(
                "Tool results belong in tool messages."
            )

        case .image, .file:
            throw InferenceError.unsupportedFeature("Attachments in assistant messages", context: id.rawValue)
        }
    }
}
