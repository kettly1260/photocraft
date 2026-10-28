using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using PhotoCraft.OnnxQnn;

namespace PhotoCraft.OnnxQnn.Tests;

internal static partial class Program
{
    private sealed record TestResult(string name, bool passed, string? error);
    private static readonly List<TestResult> Results = [];
    private static int Main(string[] args)
    {
        if (args.Length == 2 && args[0] == "--setup-synthetic")
        {
            string targetDir = Path.GetFullPath(args[1]);
            Directory.CreateDirectory(targetDir);
            InstallSynthetic(targetDir);
            Console.WriteLine($"Synthetic models installed to {targetDir}");
            return 0;
        }
        if (args.Length >= 6 && args.Contains("--action") && args.Contains("--request") && args.Contains("--response"))
        {
            return RunWorker(args);
        }
        if (args.Length != 2 || args[0] != "--results" || string.IsNullOrWhiteSpace(args[1]))
        { Console.Error.WriteLine("Required: --results <evidence.json>; no missing-evidence skip."); return 2; }
        Console.WriteLine("QNN pure tests: synthetic protocol/config/envelope/imaging and FAKE scheduling only; NO ORT/native/model/hardware inference.");
        TestProtocol(); TestEnvelope(); TestProfilesAndWorker(); TestImaging();
        int failed = Results.Count(r => !r.passed);
        string path = Path.GetFullPath(args[1]);
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        File.WriteAllText(path, JsonSerializer.Serialize(new
        {
            scope = "T1 pure C# Core tests, no ORT package/native library/model/hardware; generated protobuf is envelope-only, NOT an inference fixture",
            excludes = new[] { "Rust codec conformance (T2)", "native DLL loading", "NPU graph placement", "model accuracy/performance", "redistribution approval" },
            passed = Results.Count - failed, failed, tests = Results
        }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"{Results.Count - failed}/{Results.Count} passed; {failed} failed.");
        return failed == 0 ? 0 : 1;
    }
    private static void Run(string name, Action action)
    {
        try { action(); Results.Add(new(name, true, null)); Console.WriteLine("PASS " + name); }
        catch (Exception error) { Results.Add(new(name, false, error.ToString())); Console.Error.WriteLine($"FAIL {name}: {error.Message}"); }
    }
    private static void Check(bool condition, string message = "Assertion failed")
    { if (!condition) throw new InvalidOperationException(message); }
    private static void Near(float expected, float actual) => Check(MathF.Abs(expected - actual) < 0.00001f, $"Expected {expected}, actual {actual}");
    private static void Reject<T>(Action action) where T : Exception
    {
        try { action(); } catch (T) { return; }
        throw new InvalidOperationException($"Expected {typeof(T).Name}");
    }
    private static void Temp(Action<string> action)
    {
        string root = Path.Combine(Path.GetTempPath(), "photocraft-qnn-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try { action(root); }
        finally { Directory.Delete(root, recursive: true); }
    }
    private static ModelProfile Profile() => new("test.scalar", Protocol.Operation, "model.onnx", new string('a', 64),
        new("input", "nchw", [1, 3, 2, 2], [0f, 0f, 0f], [1f, 1f, 1f], "nearest", "assume-srgb", "ignore-straight"),
        new("output", "nchw", [1, 1, 2, 2], "probability", "nearest"));
    private static ImageMetadata Metadata(int depth = 16, bool alpha = false) => new(-7, 11, 2, 2, alpha ? 4 : 3, "rgb", depth, alpha, []);
    private static InputImage Image(int depth = 16, bool alpha = false)
    {
        var meta = Metadata(depth, alpha);
        var samples = Enumerable.Range(0, 4).SelectMany(_ => alpha ? new[] { 0f, 0.5f, 1f, 0.25f } : [0f, 0.5f, 1f]).ToArray();
        return new InputImage(new RunHeader(1, Protocol.Operation, null, JsonSerializer.SerializeToElement(new { colorPolicy = "assume-srgb" }), Protocol.Device, false, meta, null), samples);
    }
    private static ModelProfile InstallSynthetic(string root)
    {
        byte[] bytes = Model(); // Envelope-only, invalid for inference; used solely with the explicitly fake backend.
        File.WriteAllBytes(Path.Combine(root, "model.onnx"), bytes);
        var profile = Profile() with { sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant() };
        Registry(root, new(1, profile.id, [profile]));
        return profile;
    }
    private static void Registry(string root, WorkerConfig config) =>
        File.WriteAllBytes(Path.Combine(root, Profiles.ConfigName), JsonSerializer.SerializeToUtf8Bytes(config, Json.Options));

    private static int RunWorker(string[] args)
    {
        string action = "";
        string requestPath = "";
        string responsePath = "";
        string? root = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--action" && i + 1 < args.Length) action = args[i + 1];
            else if (args[i] == "--request" && i + 1 < args.Length) requestPath = args[i + 1];
            else if (args[i] == "--response" && i + 1 < args.Length) responsePath = args[i + 1];
            else if (args[i] == "--root" && i + 1 < args.Length) root = args[i + 1];
        }

        if (string.IsNullOrEmpty(requestPath) || string.IsNullOrEmpty(responsePath))
        {
            Console.Error.WriteLine("Missing --request or --response path.");
            return 2;
        }

        root ??= Path.GetDirectoryName(Path.GetFullPath(requestPath)) ?? AppContext.BaseDirectory;

        var backend = new FakeBackend();
        using var input = new FileStream(requestPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        Span<byte> magic = stackalloc byte[8];
        int bytesRead = input.Read(magic);
        input.Position = 0;

        if (bytesRead == 8 && magic.SequenceEqual("PCADREQ2"u8))
        {
            var session = new WorkerSessionV2(root, nativeWindowsArm64: true, backend);
            var replies = new List<ReplyV2>();
            while (input.Position < input.Length)
            {
                var v2Req = Protocol.ReadV2(input);
                var reply = session.Handle(v2Req);
                replies.Add(reply);
                if (session.IsClosed) break;
            }
            Protocol.PublishV2Multi(responsePath, replies);
            return 0;
        }
        else if (bytesRead == 8 && magic.SequenceEqual("PCADREQ1"u8))
        {
            using var document = Protocol.ReadHeader(input);
            InputImage? image = null;
            if (action == "run") image = Protocol.ReadImage(document.RootElement, input);
            else Protocol.Control(document.RootElement, input);
            var reply = Worker.Execute(action, image, root, nativeWindowsArm64: true, backend);
            Protocol.Publish(responsePath, reply);
            return 0;
        }
        else
        {
            Console.Error.WriteLine("Unrecognized request wire magic.");
            return 3;
        }
    }
}
