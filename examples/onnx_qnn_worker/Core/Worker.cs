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
                return ReplyV2.Status(req.RequestId, action, "unavailable", "Configured model found, but QNN HTP requires native Windows ARM64 and a compatible Qualcomm NPU/driver. No inference fallback exists.");

            backend.CheckPayload(root);

            if (action == "probe")
            {
                return ReplyV2.Status(req.RequestId, "probe", "ready", "Package/default profile and model digest verified; probe creates no QNN session, performs no inference/download and does not attest NPU compatibility. Use prepare to validate the session.");
            }

            using var session = backend.Create(root, model);
            Profiles.Bindings(profile, session.Inputs, session.Outputs);

            if (action == "prepare")
            {
                return ReplyV2.Status(req.RequestId, "prepare", "ready", "Default model QNN HTP session created and tensor bindings checked with CPU graph fallback disabled.");
            }

            if (action != "run") throw new WorkerException($"Unsupported v2 action: {action}");

            if (req.Inputs is null || !req.Inputs.TryGetValue("source", out var srcPort))
                throw new WorkerException("Run requires 'source' image port in inputs.");
            if (srcPort.data_section_index >= req.Sections.Count)
                throw new WorkerException("Source port data_section_index out of bounds.");

            var sec = req.Sections[srcPort.data_section_index];
            int expectedFloats = srcPort.width * srcPort.height * srcPort.channels;
            if (sec.Length != expectedFloats * 4)
                throw new WorkerException("Section byte length mismatch with geometry.");

            float[] samples = new float[expectedFloats];
            Buffer.BlockCopy(req.Payload, (int)sec.Offset, samples, 0, (int)sec.Length);
            for (int i = 0; i < samples.Length; i++) Protocol.Sdr(samples[i]);

            var fakeMeta = new ImageMetadata(srcPort.x ?? 0, srcPort.y ?? 0, srcPort.width, srcPort.height,
                srcPort.channels, srcPort.color_mode ?? "rgb", srcPort.source_depth ?? 32, srcPort.alpha ?? false, []);
            var inputImage = new InputImage(new RunHeader(2, "select-subject", req.Resource?.id, default, "qnn.htp", false, fakeMeta, null), samples);

            var tensor = Imaging.Preprocess(inputImage, profile);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var output = session.Run(profile.input.name, profile.output.name, tensor);
            sw.Stop();
            var mask = Imaging.Postprocess(output, profile, fakeMeta);
            return ReplyV2.RunOk(req.RequestId, mask, srcPort.width, srcPort.height, srcPort.x ?? 0, srcPort.y ?? 0, sw.Elapsed.TotalMilliseconds);
        }
        catch (NeedsModelException error)
        {
            return ReplyV2.Status(req.RequestId, action, "needs_model", error.Message, "missing_model", "ERR_MODEL_NOT_FOUND");
        }
    }
}
