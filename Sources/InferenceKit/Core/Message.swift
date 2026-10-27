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

/// The role of a message in a conversation.
public enum Role: String, Sendable, Hashable, Codable {
    case user
    case assistant
    /// A message carrying tool results. Encoders fold these into whatever
    /// shape the target wire format expects (tool-result user turns,
    /// `role: "tool"` messages, or function-call output items).
    case tool
}

/// One turn in a conversation. Callers own the history: pass the full message
/// list with every request.
public struct Message: Sendable, Hashable, Codable, Identifiable {
    public var id: UUID
    public var role: Role
    public var content: [ContentBlock]

    public init(id: UUID = UUID(), role: Role, content: [ContentBlock]) {
        self.id = id
        self.role = role
        self.content = content
    }
}

/// A piece of message content. The union covers every content shape the
/// supported wire formats can send or receive.
public enum ContentBlock: Sendable, Hashable, Codable {
    case text(String)
    case image(ImageBlock)
    case file(FileBlock)
    case thinking(ThinkingBlock)
    case toolCall(ToolCallBlock)
    case toolResult(ToolResultBlock)

    /// The text content if this block is text.
    public var text: String? {
        if case .text(let value) = self { return value }
        return nil
    }
}

/// An image attached to a message, as inline data or a remote URL.
public struct ImageBlock: Sendable, Hashable, Codable {
    /// Base64-encoded image bytes. Exactly one of `data` and `url` is set.
    public var data: Data?
    /// A remote image URL.
    public var url: URL?
    /// The image MIME type (e.g. `image/png`).
    public var mediaType: String

    public init(data: Data, mediaType: String) {
        self.data = data
        self.url = nil
        self.mediaType = mediaType
    }

    public init(url: URL, mediaType: String) {
        self.data = nil
        self.url = url
        self.mediaType = mediaType
    }
}

/// A file attached to a message (e.g. a PDF).
public struct FileBlock: Sendable, Hashable, Codable {
    public var data: Data
    public var filename: String
    public var mediaType: String

    public init(data: Data, filename: String, mediaType: String) {
        self.data = data
        self.filename = filename
        self.mediaType = mediaType
    }

    /// Whether the file is a PDF.
    public var isPDF: Bool { mediaType == "application/pdf" }

    /// Whether the file is textual and can be inlined as text.
    public var isText: Bool { mediaType.hasPrefix("text/") }

    /// Decodes the file data as UTF-8 text, if possible.
    public var textContent: String? { String(data: data, encoding: .utf8) }
}

/// Model reasoning ("thinking") content. The signature is an opaque
/// provider datum (Anthropic's `signature`, Gemini's `thoughtSignature`,
/// OpenAI's reasoning item id) that must be echoed back for multi-turn
/// conversations with tool use; persist it alongside the text.
public struct ThinkingBlock: Sendable, Hashable, Codable {
    public var text: String
    public var signature: String?

    public init(text: String, signature: String? = nil) {
        self.text = text
        self.signature = signature
    }
}

/// A tool call the model wants performed, as sent back in conversation history.
public struct ToolCallBlock: Sendable, Hashable, Codable {
    public var id: String
    public var name: String
    /// The call arguments as raw JSON text.
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

/// The result of a tool call, to be fed back to the model.
public struct ToolResultBlock: Sendable, Hashable, Codable {
    public var callID: String
    /// The tool name. Required by some wire formats (Gemini's
    /// `functionResponse`), ignored by others.
    public var name: String?
    public var output: String
    public var isError: Bool

    public init(callID: String, name: String? = nil, output: String, isError: Bool = false) {
        self.callID = callID
        self.name = name
        self.output = output
        self.isError = isError
    }
}

// MARK: - Construction conveniences

extension Message {
    /// A user message containing a single text block.
    public static func user(_ text: String) -> Message {
        Message(role: .user, content: [.text(text)])
    }

    /// A user message with attachments followed by text.
    public static func user(_ text: String, attachments: [ContentBlock]) -> Message {
        Message(role: .user, content: attachments + [.text(text)])
    }

    /// An assistant message with optional thinking, text, and tool calls, in that order.
    public static func assistant(
        thinking: String? = nil,
        text: String? = nil,
        toolCalls: [ToolCall] = []
    ) -> Message {
        var content: [ContentBlock] = []
        if let thinking {
            content.append(.thinking(ThinkingBlock(text: thinking)))
        }
        if let text, !text.isEmpty {
            content.append(.text(text))
        }
        for call in toolCalls {
            content.append(.toolCall(ToolCallBlock(id: call.id, name: call.name, arguments: call.arguments)))
        }
        return Message(role: .assistant, content: content)
    }

    /// A tool result message.
    public static func toolResult(
        callID: String,
        name: String? = nil,
        output: String,
        isError: Bool = false
    ) -> Message {
        Message(
            role: .tool,
            content: [.toolResult(ToolResultBlock(callID: callID, name: name, output: output, isError: isError))]
        )
    }
}
