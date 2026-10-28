namespace PhotoCraft.OnnxQnn;

internal interface IInferenceBackend
{
    void CheckPayload(string root);
    IInferenceSession Create(string root, VerifiedModel model);
}
internal interface IInferenceSession : IDisposable
{
    TensorBinding[] Inputs { get; }
    TensorBinding[] Outputs { get; }
    TensorData Run(string inputName, string outputName, TensorData input);
}

internal static class Worker
{
    internal static Reply Execute(string action, InputImage? image, string root, bool nativeWindowsArm64, IInferenceBackend backend)
    {
        if (action is not ("probe" or "prepare" or "run")) throw new WorkerException("Unsupported action.");
        if ((action == "run") != (image is not null)) throw new WorkerException("Action/request mismatch.");
        try
        {
            var profiles = Profiles.Load(root);
            var profile = profiles.Select(image?.Header.resource);
            using var model = profiles.Verify(profile);
            if (!nativeWindowsArm64)
                return Reply.Status("unavailable", "Configured model found, but QNN HTP requires native Windows ARM64 and a compatible Qualcomm NPU/driver. No inference fallback exists.");
            backend.CheckPayload(root);
            if (action == "probe")
                return Reply.Status("ready", "Package/default profile and model digest verified; probe creates no QNN session, performs no inference/download and does not attest NPU compatibility. Use prepare to validate the default session.");
            using var session = backend.Create(root, model);
            Profiles.Bindings(profile, session.Inputs, session.Outputs);
            if (action == "prepare")
                return Reply.Status("ready", "Default model QNN HTP session created and tensor bindings checked with CPU graph fallback disabled. No inference/download. v1 prepare has no resource; run re-creates its selected session.");
            if (image is null) throw new WorkerException("Missing run image.");
            var tensor = Imaging.Preprocess(image, profile);
            var output = session.Run(profile.input.name, profile.output.name, tensor);
            var mask = Imaging.Postprocess(output, profile, image.Header.image);
            return new Reply(new ResponseHeader(1, "ok", "", new MaskOutput("mask", Protocol.MaskMetadata(image.Header.image)),
                new Execution([Protocol.Device], Protocol.Backend, null)), mask);
        }
        catch (NeedsModelException error) { return Reply.Status("needs_model", error.Message); }
    }

    internal static ReplyV2 ExecuteV2(string action, V2ParsedRequest req, string root, bool nativeWindowsArm64, IInferenceBackend backend)
    {
        if (action == "hello")
        {
            return ReplyV2.Hello(req.RequestId);
        }

        try
        {
            var profiles = Profiles.Load(root);
            var profile = profiles.Select(req.Resource?.id);
            using var model = profiles.Verify(profile);

            if (!nativeWindowsArm64)
            {
                return action == "run"
                    ? ReplyV2.RunFailed(req.RequestId, "unavailable", "Configured model found, but QNN HTP requires native Windows ARM64 and a compatible Qualcomm NPU/driver. No inference fallback exists.", "device_unavailable")
                    : ReplyV2.Status(req.RequestId, action, "unavailable", "Configured model found, but QNN HTP requires native Windows ARM64 and a compatible Qualcomm NPU/driver. No inference fallback exists.", null, "device_unavailable");
            }

            backend.CheckPayload(root);

            if (action == "probe")
            {
                return ReplyV2.Status(req.RequestId, "probe", "ready", "Package/default profile and model digest verified; probe creates no QNN session, performs no inference/download and does not attest NPU compatibility. Use prepare to validate the session.", "files_verified");
            }

            using var session = backend.Create(root, model);
            Profiles.Bindings(profile, session.Inputs, session.Outputs);

            if (action == "prepare")
            {
                return ReplyV2.Status(req.RequestId, "prepare", "ready", "Default model QNN HTP session created and tensor bindings checked with CPU graph fallback disabled.", "session_created");
            }

            if (action != "run") throw new WorkerException($"Unsupported v2 action: {action}");

            if (req.Inputs is null || (!req.Inputs.TryGetValue("source", out var srcPort) && !req.Inputs.TryGetValue("image", out srcPort)))
                return ReplyV2.RunFailed(req.RequestId, "error", "Run requires 'source' or 'image' port in inputs.", "invalid_request");
            if (srcPort.data_section_index >= req.Sections.Count)
                return ReplyV2.RunFailed(req.RequestId, "error", "Source port data_section_index out of bounds.", "invalid_request");

            var sec = req.Sections[srcPort.data_section_index];
            int expectedFloats = srcPort.width * srcPort.height * srcPort.channels;
            if (sec.Length != expectedFloats * 4)
                return ReplyV2.RunFailed(req.RequestId, "error", "Section byte length mismatch with geometry.", "invalid_request");

            float[] samples = new float[expectedFloats];
            Buffer.BlockCopy(req.Payload, (int)sec.Offset, samples, 0, (int)sec.Length);
            for (int i = 0; i < samples.Length; i++) Protocol.Sdr(samples[i]);

            var fakeMeta = new ImageMetadata(srcPort.x ?? 0, srcPort.y ?? 0, srcPort.width, srcPort.height,
                srcPort.channels, srcPort.color_mode ?? "rgb", srcPort.source_depth ?? 32, srcPort.alpha ?? false, []);
            var inputImage = new InputImage(new RunHeader(2, "select-subject", req.Resource?.id, default, "qnn.htp", false, fakeMeta, null), samples);

            var swPre = System.Diagnostics.Stopwatch.StartNew();
            var tensor = Imaging.Preprocess(inputImage, profile);
            swPre.Stop();
            var swExec = System.Diagnostics.Stopwatch.StartNew();
            var output = session.Run(profile.input.name, profile.output.name, tensor);
            swExec.Stop();
            var swPost = System.Diagnostics.Stopwatch.StartNew();
            var mask = Imaging.Postprocess(output, profile, fakeMeta);
            swPost.Stop();

            var policy = req.DevicePolicy ?? new V2DevicePolicy("exact", ["qnn"], false);
            var res = req.Resource ?? new V2ResourceIdentity(profile.id, "1.0.0", profile.sha256);
            return ReplyV2.RunOk(
                req.RequestId,
                mask,
                srcPort.width,
                srcPort.height,
                srcPort.x ?? 0,
                srcPort.y ?? 0,
                policy,
                res,
                profile.sha256,
                swPre.Elapsed.TotalMilliseconds,
                swExec.Elapsed.TotalMilliseconds,
                swPost.Elapsed.TotalMilliseconds
            );
        }
        catch (NeedsModelException error)
        {
            return action == "run"
                ? ReplyV2.RunFailed(req.RequestId, "needs_model", error.Message, "resource_missing")
                : ReplyV2.Status(req.RequestId, action, "needs_model", error.Message, null, "resource_missing");
        }
    }
}

internal sealed class WorkerSessionV2
{
    private enum SessionState { Uninitialized, Negotiated, Closed }
    private SessionState _state = SessionState.Uninitialized;
    private readonly HashSet<string> _seenRequests = new(StringComparer.Ordinal);
    private readonly string _root;
    private readonly bool _nativeWindowsArm64;
    private readonly IInferenceBackend _backend;

    internal bool IsClosed => _state == SessionState.Closed;

    internal WorkerSessionV2(string root, bool nativeWindowsArm64, IInferenceBackend backend)
    {
        _root = root;
        _nativeWindowsArm64 = nativeWindowsArm64;
        _backend = backend;
    }

    internal ReplyV2 Handle(V2ParsedRequest req)
    {
        if (_state == SessionState.Closed)
        {
            return req.Action == "run"
                ? ReplyV2.RunFailed(req.RequestId, "error", "Session endpoint is closed.", "invalid_request")
                : ReplyV2.Status(req.RequestId, req.Action, "error", "Session endpoint is closed.", null, "invalid_request");
        }

        if (req.Action == "hello")
        {
            if (_state != SessionState.Uninitialized)
            {
                _state = SessionState.Closed;
                return ReplyV2.Status(req.RequestId, req.Action, "error", "Repeated hello handshake check: session endpoint closed.", null, "invalid_request");
            }
            if (!_seenRequests.Add(req.RequestId))
            {
                _state = SessionState.Closed;
                return ReplyV2.Status(req.RequestId, req.Action, "error", $"Duplicate request_id: {req.RequestId}", null, "invalid_request");
            }
            _state = SessionState.Negotiated;
            return ReplyV2.Hello(req.RequestId);
        }

        if (_state != SessionState.Negotiated)
        {
            _state = SessionState.Closed;
            return req.Action == "run"
                ? ReplyV2.RunFailed(req.RequestId, "error", "Command sent before hello negotiation.", "invalid_request")
                : ReplyV2.Status(req.RequestId, req.Action, "error", "Command sent before hello negotiation.", null, "invalid_request");
        }

        if (!_seenRequests.Add(req.RequestId))
        {
            return req.Action == "run"
                ? ReplyV2.RunFailed(req.RequestId, "error", $"Duplicate request_id: {req.RequestId}", "invalid_request")
                : ReplyV2.Status(req.RequestId, req.Action, "error", $"Duplicate request_id: {req.RequestId}", null, "invalid_request");
        }

        return Worker.ExecuteV2(req.Action, req, _root, _nativeWindowsArm64, _backend);
    }
}
