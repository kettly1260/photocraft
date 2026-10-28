using System.Runtime.InteropServices;
using System.Text.Json;

namespace PhotoCraft.OnnxQnn;

internal static class Program
{
    private static int Main(string[] args)
    {
        Arguments arguments;
        try { arguments = Arguments.Parse(args); }
        catch { Console.Error.WriteLine("Invalid QNN worker arguments; expected the six arguments with absolute local job paths."); return 2; }

        bool isV2 = false;
        string v2RequestId = "unknown";

        try
        {
            using var input = new FileStream(arguments.Request, FileMode.Open, FileAccess.Read, FileShare.Read);
            Span<byte> magic = stackalloc byte[8];
            if (input.Length >= 8)
            {
                input.ReadExactly(magic);
                input.Position = 0;
            }

            bool supported = OperatingSystem.IsWindows() && RuntimeInformation.ProcessArchitecture == Architecture.Arm64 &&
                RuntimeInformation.OSArchitecture == Architecture.Arm64;

            if (magic.SequenceEqual("PCADREQ2"u8))
            {
                isV2 = true;
                var v2Req = Protocol.ReadV2(input);
                v2RequestId = v2Req.RequestId;
                var replyV2 = Worker.ExecuteV2(arguments.Action, v2Req, AppContext.BaseDirectory, supported, new QnnBackend());
                Protocol.PublishV2(arguments.Response, replyV2);
                return 0;
            }
            else
            {
                using var document = Protocol.ReadHeader(input);
                InputImage? image = null;
                if (arguments.Action == "run") image = Protocol.ReadImage(document.RootElement, input);
                else Protocol.Control(document.RootElement, input);
                var reply = Worker.Execute(arguments.Action, image, AppContext.BaseDirectory, supported, new QnnBackend());
                Protocol.Publish(arguments.Response, reply);
                return 0;
            }
        }
        catch (Exception error)
        {
            // Do not echo model paths, image data, user JSON, or opaque native error text to the host.
            string message = error switch
            {
                WorkerException => error.Message,
                NeedsModelException => error.Message,
                EndOfStreamException => isV2 ? "Truncated v2 request file." : "Truncated v1 request file.",
                JsonException => "Invalid request/registry JSON, fields or value types.",
                _ => $"QNN worker failed ({error.GetType().Name}, HRESULT 0x{error.HResult:X8}); no CPU/GPU graph fallback was attempted.",
            };

            if (isV2)
            {
                string status = error is NeedsModelException ? "needs_model" : "error";
                string? errorCode = error is NeedsModelException ? "ERR_MODEL_NOT_FOUND" : "ERR_INTERNAL";
                var replyV2 = ReplyV2.Status(v2RequestId, arguments.Action, status, message, errorCode: errorCode);
                try { Protocol.PublishV2(arguments.Response, replyV2); return 0; }
                catch { Console.Error.WriteLine("Cannot publish bounded QNN worker v2 response."); return 3; }
            }
            else
            {
                var reply = Reply.Status(error is NeedsModelException ? "needs_model" : "error", message);
                try { Protocol.Publish(arguments.Response, reply); return 0; }
                catch { Console.Error.WriteLine("Cannot publish bounded QNN worker response."); return 3; }
            }
        }
    }
}
