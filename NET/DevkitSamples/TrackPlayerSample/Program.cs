using Datafeel;
using Datafeel.NET.Serial;
using System.Reflection;

var manager = new DotManagerConfiguration()
    .AddDot<Dot_63x_xxx>(1)
    .AddDot<Dot_63x_xxx>(2)
    .AddDot<Dot_63x_xxx>(3)
    .AddDot<Dot_63x_xxx>(4)
    .CreateDotManager();

var client = new DatafeelModbusClientConfiguration()
    .UseWindowsSerialPortTransceiver()
    //.UseSerialPort("COM3") // Uncomment this line to specify the serial port by name
    .CreateClient();

var trackPlayer = new TrackPlayer(manager);

await manager.Start(client);

string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), @"Tracks\my-track.json");

try
{
    await trackPlayer.PlayTrack(path);
}
catch (Exception e)
{
    Console.WriteLine(e.Message);
}

await Task.Delay(1000);
await manager.Stop();


