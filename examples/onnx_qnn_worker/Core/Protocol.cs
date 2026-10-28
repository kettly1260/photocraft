using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PhotoCraft.OnnxQnn;

internal sealed record ImageMetadata(int x, int y, int width, int height, int channels,
    string color_mode, int source_depth, bool alpha, int[] icc_profile);
internal sealed record RunHeader(int protocol, string operation, string? resource, JsonElement @params,
    string? device, bool allow_fallback, ImageMetadata image, ImageMetadata? selection);
internal sealed record InputImage(RunHeader Header, float[] Samples);
internal sealed record MaskOutput(string kind, ImageMetadata image);
internal sealed record Execution(string[] devices, string backend, string? fallback_reason);
// Exactly the Rust v1 response keys, including nulls. Do not add descriptor/readiness extensions.
internal sealed record ResponseHeader(int protocol, string status, string message,
    MaskOutput? output = null, Execution? execution = null);
internal sealed record Reply(ResponseHeader Header, float[]? Mask = null)
{
    internal static Reply Status(string status, string message) => new(new ResponseHeader(1, status, message));
}

// ---------------------------------------------------------------------------
// Protocol v2 Records
// ---------------------------------------------------------------------------

internal sealed record SectionEntry(uint Kind, uint Reserved, uint Offset, uint Length);
internal sealed record V2SectionMeta(string kind, uint offset, uint length);
internal sealed record V2ClientNegotiation(int[]? wire_versions, string[]? supported_features, string[]? required_features);
internal sealed record V2ServerNegotiation(int selected_wire_version, string[] enabled_features);
internal sealed record V2ResourceIdentity(string id, string? version, string? manifest_digest);
internal sealed record V2DevicePolicy(string mode, string[] allowed_backends, bool allow_graph_fallback);
internal sealed record V2Port(string kind, int? x, int? y, int width, int height, int channels,
    string? color_mode, int? source_depth, bool? alpha,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? alpha_type,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? color_encoding,
    int data_section_index,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? icc_section_index);
internal sealed record V2Device(string id, string kind);
internal sealed record V2Timing(double cpu_preprocess_ms, double graph_execution_ms, double cpu_postprocess_ms, double total_ms);
internal sealed record V2Execution(V2DevicePolicy? requested_policy, V2Device[] devices, string backend,
    string runtime_id, string runtime_version, V2ResourceIdentity? resource, string? configuration_digest,
    bool graph_partitioned, bool graph_fallback, [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] string? fallback_reason, string? evidence, V2Timing timing);
internal sealed record V2ErrorDetail(string code, bool retryable);

internal sealed record V2RequestHeader(
    int protocol,
    string request_id,
    string type,
    string? action,
    string? operation,
    string? operation_version,
    V2ResourceIdentity? resource,
    V2DevicePolicy? device_policy,
    V2ClientNegotiation? negotiation,
    JsonElement? @params,
    Dictionary<string, V2Port>? inputs,
    JsonElement? sections
);

internal sealed record V2ResponseHeader(
    int protocol,
    string request_id,
    string type,
    string status,
    string message,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? action = null,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? readiness = null,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] V2ServerNegotiation? negotiation = null,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] Dictionary<string, V2Port>? outputs = null,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] V2Execution? execution = null,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] V2ErrorDetail? error = null,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] V2SectionMeta[]? sections = null
);

internal sealed record V2ParsedRequest(
    string Action,
    string RequestId,
    V2ResourceIdentity? Resource,
    V2DevicePolicy? DevicePolicy,
    Dictionary<string, V2Port>? Inputs,
    List<SectionEntry> Sections,
    byte[] Payload
);

internal sealed record ReplyV2(V2ResponseHeader Header, List<SectionEntry> SectionTable, byte[] Payload)
{
    internal static ReplyV2 Hello(string requestId) => new(
        new V2ResponseHeader(
            2,
            requestId,
            "hello",
            "negotiated",
            "PhotoCraft ONNX QNN Worker v2 ready",
            negotiation: new V2ServerNegotiation(2, ["image-mask-v2", "same-geometry", "resource-control"]),
            sections: []
        ),
        [],
        []
    );

    internal static ReplyV2 Status(string requestId, string action, string status, string message, string? readiness = null, string? errorCode = null) => new(
        new V2ResponseHeader(
            2,
            requestId,
            "control",
            status,
            message,
            action: action,
            readiness: status == "ready" ? (readiness ?? "files_verified") : null,
            error: status != "ready" ? new V2ErrorDetail(errorCode ?? "resource_missing", false) : null,
            sections: []
        ),
        [],
        []
    );

    internal static ReplyV2 RunOk(
        string requestId,
        float[] mask,
        int width,
        int height,
        int x,
        int y,
        V2DevicePolicy? requestedPolicy,
        V2ResourceIdentity? resource,
        string? configDigest,
        double preMs,
        double execMs,
        double postMs)
    {
        byte[] payload = new byte[mask.Length * 4];
        Buffer.BlockCopy(mask, 0, payload, 0, payload.Length);
        var secEntry = new SectionEntry(2, 0, 0, (uint)payload.Length);
        var port = new V2Port("mask", x, y, width, height, 1, "gray", 32, false, null, null, 0, null);
        var outputs = new Dictionary<string, V2Port> { ["selection"] = port };
        var exec = new V2Execution(
            requestedPolicy,
            [new V2Device("qnn.htp", "npu")],
            "qnn",
            "qualcomm-qnn",
            "1.24.4",
            resource,
            configDigest,
            false,
            false,
            null,
            "worker_report",
            new V2Timing(preMs, execMs, postMs, preMs + execMs + postMs)
        );
        return new ReplyV2(
            new V2ResponseHeader(
                2,
                requestId,
                "run",
                "ok",
                "Inference complete",
                outputs: outputs,
                execution: exec,
                sections: [new V2SectionMeta("f32", 0, (uint)payload.Length)]
            ),
            [secEntry],
            payload
        );
    }

    internal static ReplyV2 RunFailed(string requestId, string status, string message, string errorCode) => new(
        new V2ResponseHeader(
            2,
            requestId,
            "run",
            status,
            message,
            error: new V2ErrorDetail(errorCode, false),
            sections: []
        ),
        [],
        []
    );
}

internal static class Protocol
{
    internal const string Operation = "select-subject";
    internal const string Device = "qnn.htp";
    internal const string Backend = "qnn";
    internal const int MaxHeaderBytes = 1024 * 1024;
    internal const int MaxPixels = 16 * 1024 * 1024;
    internal const int MaxSide = 16384;
    internal const int MaxSections = 64;
    internal const int MaxPayloadBytes = 256 * 1024 * 1024;

    private static readonly byte[] RequestMagicV1 = "PCADREQ1"u8.ToArray();
    private static readonly byte[] RequestMagicV2 = "PCADREQ2"u8.ToArray();
    private static readonly byte[] ResponseMagicV1 = "PCADRES1"u8.ToArray();
    private static readonly byte[] ResponseMagicV2 = "PCADRES2"u8.ToArray();

    internal static JsonDocument ReadHeader(Stream input)
    {
        if (!input.CanSeek) throw new WorkerException("A seekable bounded request file is required.");
        if (input.Length > 12L + MaxHeaderBytes + (long)MaxPixels * 5 * 4)
            throw new WorkerException("Request exceeds worker byte limit.");
        Span<byte> prefix = stackalloc byte[12];
        input.ReadExactly(prefix);
        if (!prefix[..8].SequenceEqual(RequestMagicV1)) throw new WorkerException("Expected PCADREQ1 request magic.");
        uint length = BinaryPrimitives.ReadUInt32LittleEndian(prefix[8..]);
        if (length is 0 or > MaxHeaderBytes) throw new WorkerException("JSON header must be 1..1048576 bytes.");
        if (input.Length - input.Position < length) throw new WorkerException("Truncated JSON header.");
        var bytes = new byte[(int)length];
        input.ReadExactly(bytes);
        var document = Json.Parse(bytes);
        try
        {
            if (!document.RootElement.TryGetProperty("protocol", out var protocol) || protocol.GetInt32() != 1)
                throw new WorkerException("Only adapter protocol 1 is supported.");
            return document;
        }
        catch { document.Dispose(); throw; }
    }

    internal static V2ParsedRequest ReadV2(Stream input)
    {
        if (!input.CanSeek) throw new WorkerException("A seekable bounded request file is required.");
        if (input.Length > 16L + MaxHeaderBytes + (long)MaxSections * 16 + MaxPayloadBytes)
            throw new WorkerException("Request exceeds worker v2 byte limit.");

        Span<byte> prefix = stackalloc byte[12];
        input.ReadExactly(prefix);
        if (!prefix[..8].SequenceEqual(RequestMagicV2)) throw new WorkerException("Expected PCADREQ2 request magic.");

        uint headerLength = BinaryPrimitives.ReadUInt32LittleEndian(prefix[8..]);
        if (headerLength is 0 or > MaxHeaderBytes) throw new WorkerException("JSON header must be 1..1048576 bytes.");
        if (input.Length - input.Position < headerLength + 4) throw new WorkerException("Truncated JSON header.");

        var headerBytes = new byte[(int)headerLength];
        input.ReadExactly(headerBytes);
        if (headerBytes.Length >= 3 && headerBytes[0] == 0xef && headerBytes[1] == 0xbb && headerBytes[2] == 0xbf)
            throw new WorkerException("UTF-8 byte order mark (BOM) is forbidden in protocol v2.");

        using var document = Json.Parse(headerBytes);
        var root = document.RootElement;
        if (!root.TryGetProperty("protocol", out var protoElem) || protoElem.GetInt32() != 2)
            throw new WorkerException("Only adapter protocol 2 is supported in v2 frames.");

        Span<byte> secCountBuf = stackalloc byte[4];
        input.ReadExactly(secCountBuf);
        uint sectionCount = BinaryPrimitives.ReadUInt32LittleEndian(secCountBuf);
        if (sectionCount > MaxSections)
            throw new WorkerException($"Section count {sectionCount} exceeds limit of {MaxSections}.");

        long secTableBytes = (long)sectionCount * 16;
        if (input.Length - input.Position < secTableBytes)
            throw new WorkerException("Truncated section table.");

        var sections = new List<SectionEntry>((int)sectionCount);
        Span<byte> entryBuf = stackalloc byte[16];
        for (int i = 0; i < sectionCount; i++)
        {
            input.ReadExactly(entryBuf);
            uint kind = BinaryPrimitives.ReadUInt32LittleEndian(entryBuf[0..4]);
            uint reserved = BinaryPrimitives.ReadUInt32LittleEndian(entryBuf[4..8]);
            uint offset = BinaryPrimitives.ReadUInt32LittleEndian(entryBuf[8..12]);
            uint length = BinaryPrimitives.ReadUInt32LittleEndian(entryBuf[12..16]);

            if (reserved != 0) throw new WorkerException($"Section {i} reserved field is non-zero.");
            if (offset % 4 != 0) throw new WorkerException($"Section {i} offset is not 4-byte aligned.");
            if (length % 4 != 0) throw new WorkerException($"Section {i} length is not 4-byte aligned.");
            sections.Add(new SectionEntry(kind, reserved, offset, length));
        }

        long payloadLength = input.Length - input.Position;
        if (payloadLength > MaxPayloadBytes)
            throw new WorkerException($"Payload length {payloadLength} exceeds limit of {MaxPayloadBytes}.");

        var payload = new byte[(int)payloadLength];
        input.ReadExactly(payload);

        foreach (var sec in sections)
        {
            if ((long)sec.Offset + sec.Length > payload.Length)
                throw new WorkerException("Section bounds exceed payload buffer.");
        }

        var header = root.Deserialize<V2RequestHeader>(Json.Options) ?? throw new WorkerException("Missing v2 request header.");
        return new V2ParsedRequest(
            header.type == "run" ? "run" : (header.action ?? header.type),
            header.request_id,
            header.resource,
            header.device_policy,
            header.inputs,
            sections,
            payload
        );
    }

    internal static void Control(JsonElement root, Stream input)
    {
        Json.Fields(root, "protocol");
        End(input);
    }

    internal static InputImage ReadImage(JsonElement root, Stream input)
    {
        Json.Fields(root, "protocol", "operation", "resource", "params", "device", "allow_fallback", "image", "selection");
        ImageFields(root.GetProperty("image"));
        if (root.GetProperty("selection").ValueKind != JsonValueKind.Null) ImageFields(root.GetProperty("selection"));
        var header = root.Deserialize<RunHeader>(Json.Options) ?? throw new WorkerException("Missing run header.");
        if (header.operation != Operation) throw new WorkerException("Only select-subject/mask is supported.");
        if (header.device is not null && header.device != Device) throw new WorkerException("Only qnn.htp is supported; no fallback exists.");
        if (header.resource is not null && !Json.Id(header.resource))
            throw new WorkerException("resource must be an opaque registered model ID, never a path or URL.");
        if (header.@params.ValueKind != JsonValueKind.Object) throw new WorkerException("params must be an object.");
        Json.Fields(header.@params, "colorPolicy");
        if (header.@params.GetProperty("colorPolicy").GetString() != "assume-srgb")
            throw new WorkerException("Explicit params.colorPolicy='assume-srgb' is required for untagged RGB.");
        var image = header.image ?? throw new WorkerException("Run requires image metadata.");
        int pixels = Geometry(image);
        if (image.color_mode != "rgb" || image.channels != (image.alpha ? 4 : 3))
            throw new WorkerException("Only RGB with 3 channels or 4 straight-alpha-last channels is supported.");
        if (image.source_depth is not (8 or 16 or 32)) throw new WorkerException("source_depth must be 8, 16 or 32; samples are f32le.");
        if (image.icc_profile is null || image.icc_profile.Length != 0)
            throw new WorkerException("ICC-tagged input is unsupported: convert explicitly in the color-managed host; no ICC is ignored or substituted.");
        if (header.selection is { } selection)
        {
            Geometry(selection);
            if (selection.x != image.x || selection.y != image.y || selection.width != image.width ||
                selection.height != image.height || selection.channels != 1 || selection.alpha ||
                selection.color_mode is not ("gray" or "grayscale") || selection.source_depth != 32 ||
                selection.icc_profile is null || selection.icc_profile.Length != 0)
                throw new WorkerException("Selection must be same-geometry, one-channel profile-free gray F32.");
        }
        int count = checked(pixels * image.channels);
        long bytes = checked(((long)count + (header.selection is null ? 0 : pixels)) * 4);
        if (input.Length - input.Position != bytes) throw new WorkerException("Sample payload length does not match metadata.");
        var samples = new float[count];
        using var reader = new BinaryReader(input, Encoding.UTF8, leaveOpen: true);
        for (int i = 0; i < count; i++) samples[i] = Sdr(reader.ReadSingle());
        if (header.selection is not null)
            for (int i = 0; i < pixels; i++) Sdr(reader.ReadSingle());
        End(input);
        return new(header, samples);
    }

    private static void ImageFields(JsonElement image) => Json.Fields(image, "x", "y", "width", "height", "channels", "color_mode", "source_depth", "alpha", "icc_profile");

    internal static int Geometry(ImageMetadata image)
    {
        if (image.width is <= 0 or > MaxSide || image.height is <= 0 or > MaxSide ||
            (long)image.width * image.height > MaxPixels ||
            (long)image.x + image.width > int.MaxValue || (long)image.y + image.height > int.MaxValue)
            throw new WorkerException("Invalid/oversized image geometry (maximum 16 Mi pixels, side 16384, i32 endpoints).");
        return checked(image.width * image.height);
    }

    internal static float Sdr(float sample)
    {
        if (!float.IsFinite(sample) || sample < 0 || sample > 1)
            throw new WorkerException("Only finite SDR [0,1] RGB/alpha is supported; HDR is rejected, not clamped.");
        return sample;
    }

    private static void End(Stream input)
    {
        if (input.ReadByte() != -1) throw new WorkerException("Unexpected trailing request bytes.");
    }

    internal static void Write(Stream stream, Reply reply)
    {
        var h = reply.Header;
        if (h.protocol != 1 || h.status is not ("ok" or "ready" or "needs_model" or "unavailable" or "error") ||
            h.message is null || Encoding.UTF8.GetByteCount(h.message) > 1024 || h.message.Any(char.IsControl))
            throw new WorkerException("Invalid response status/message.");
        if (h.status == "ok")
        {
            if (h.output is not { kind: "mask" } output || h.execution is not { } execution || reply.Mask is not { } mask ||
                execution.backend != Backend || execution.devices.Length != 1 || execution.devices[0] != Device || execution.fallback_reason is not null)
                throw new WorkerException("Invalid successful mask response.");
            var image = output.image;
            if (Geometry(image) != mask.Length || image.channels != 1 || image.alpha || image.color_mode != "gray" ||
                image.source_depth != 32 || image.icc_profile.Length != 0)
                throw new WorkerException("Invalid mask metadata/size.");
            foreach (float value in mask) Sdr(value);
        }
        else if (h.output is not null || h.execution is not null || reply.Mask is not null)
            throw new WorkerException("Status replies must have null output/execution and no payload.");
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(h, Json.Options);
        if (json.Length > MaxHeaderBytes) throw new WorkerException("Oversized response header.");
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(ResponseMagicV1);
        writer.Write((uint)json.Length);
        writer.Write(json);
        if (reply.Mask is { } data) foreach (float value in data) writer.Write(value);
        writer.Flush();
    }

    internal static void WriteV2(Stream stream, ReplyV2 reply)
    {
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(reply.Header, Json.Options);
        if (json.Length > MaxHeaderBytes) throw new WorkerException("Oversized v2 response header.");
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(ResponseMagicV2);
        writer.Write((uint)json.Length);
        writer.Write(json);
        writer.Write((uint)reply.SectionTable.Count);
        foreach (var sec in reply.SectionTable)
        {
            writer.Write(sec.Kind);
            writer.Write(sec.Reserved);
            writer.Write(sec.Offset);
            writer.Write(sec.Length);
        }
        writer.Write(reply.Payload);
        writer.Flush();
    }

    internal static ImageMetadata MaskMetadata(ImageMetadata source) =>
        new(source.x, source.y, source.width, source.height, 1, "gray", 32, false, []);

    internal static void Publish(string path, Reply reply)
    {
        string directory = Path.GetDirectoryName(path) ?? throw new WorkerException("Response must have a parent directory.");
        string temporary = Path.Combine(directory, ".photocraft-qnn-" + Guid.NewGuid().ToString("N"));
        try
        {
            using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            { Write(file, reply); file.Flush(flushToDisk: true); }
            File.Move(temporary, path, overwrite: false);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    internal static void PublishV2(string path, ReplyV2 reply)
    {
        string directory = Path.GetDirectoryName(path) ?? throw new WorkerException("Response must have a parent directory.");
        string temporary = Path.Combine(directory, ".photocraft-qnn-v2-" + Guid.NewGuid().ToString("N"));
        try
        {
            using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            { WriteV2(file, reply); file.Flush(flushToDisk: true); }
            File.Move(temporary, path, overwrite: false);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    internal static void PublishV2Multi(string path, IEnumerable<ReplyV2> replies)
    {
        string directory = Path.GetDirectoryName(path) ?? throw new WorkerException("Response must have a parent directory.");
        string temporary = Path.Combine(directory, ".photocraft-qnn-v2-m-" + Guid.NewGuid().ToString("N"));
        try
        {
            using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                foreach (var reply in replies)
                {
                    WriteV2(file, reply);
                }
                file.Flush(flushToDisk: true);
            }
            File.Move(temporary, path, overwrite: false);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }
}

internal sealed record Arguments(string Action, string Request, string Response)
{
    internal static Arguments Parse(string[] args)
    {
        if (args.Length != 6 || args[0] != "--action" || args[2] != "--request" || args[4] != "--response" ||
            args[1] is not ("hello" or "probe" or "prepare" or "run"))
            throw new WorkerException("Expected exactly --action hello|probe|prepare|run --request absolute --response absolute.");
        string request = LocalPath(args[3]);
        string response = LocalPath(args[5]);
        if (request.Equals(response, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(Path.GetDirectoryName(request), Path.GetDirectoryName(response), StringComparison.OrdinalIgnoreCase) || File.Exists(response))
            throw new WorkerException("Request/response must be distinct files in one private job directory; response must not exist.");
        return new(args[1], request, response);
    }
    private static string LocalPath(string path)
    {
        if (OperatingSystem.IsWindows())
        {
            if (!Path.IsPathFullyQualified(path) || path.Length < 4 || !char.IsAsciiLetter(path[0]) || path[1] != ':' ||
                (path[2] != '\\' && path[2] != '/') || path.IndexOf(':', 2) >= 0)
                throw new WorkerException("Only absolute local drive paths are accepted; no UNC/device paths or alternate data streams.");
        }
        else
        {
            if (!Path.IsPathFullyQualified(path))
                throw new WorkerException("Only absolute local paths are accepted.");
        }
        return Path.GetFullPath(path);
    }
}
