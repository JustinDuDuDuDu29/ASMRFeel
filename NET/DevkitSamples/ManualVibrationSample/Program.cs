using Datafeel;
using Datafeel.NET.Serial;

var manager = new DotManagerConfiguration()
    .AddDot<Dot_63x_xxx>(1)
    .AddDot<Dot_63x_xxx>(2)
    .AddDot<Dot_63x_xxx>(3)
    .AddDot<Dot_63x_xxx>(4)
    .CreateDotManager();

// Modify these and flush them out using manager.Write()
var dots = new List<DotPropsWritable>()
{
    new DotPropsWritable() { Address = 1, LedMode = LedModes.Breathe, GlobalLed = new(), VibrationMode = VibrationModes.Manual, VibrationIntensity = 1.0f, VibrationFrequency = 170},
    new DotPropsWritable() { Address = 2, LedMode = LedModes.Breathe, GlobalLed = new(), VibrationMode = VibrationModes.Manual, VibrationIntensity = 1.0f, VibrationFrequency = 170},
    new DotPropsWritable() { Address = 3, LedMode = LedModes.Breathe, GlobalLed = new(), VibrationMode = VibrationModes.Manual, VibrationIntensity = 1.0f, VibrationFrequency = 170},
    new DotPropsWritable() { Address = 4, LedMode = LedModes.Breathe, GlobalLed = new(), VibrationMode = VibrationModes.Manual, VibrationIntensity = 1.0f, VibrationFrequency = 170},
};

try
{
    using var cts = new CancellationTokenSource(1000);
    var serialClient = new DatafeelModbusClientConfiguration()
        .UseWindowsSerialPortTransceiver()
        //.UseSerialPort("COM3") // Uncomment this line to specify the serial port by name
        .CreateClient();
    var clients = new List<DatafeelModbusClient> { serialClient };
    var result = await manager.Start(clients, cts.Token);
    if (result)
    {
        Console.WriteLine("Started");
    }
    else
    {
        Console.WriteLine("Failed to start");
    }
}
catch (Exception e)
{
    Console.WriteLine(e.Message);
}
var random = new Random();

while (true)
{
    var delay = Task.Delay(100);
    foreach (var d in dots)
    {
        d.VibrationIntensity = 1.0f;
        d.VibrationFrequency += 10;
        if(d.VibrationFrequency > 250)
        {
            d.VibrationFrequency = 100;
        }

        try
        {
            await manager.Write(d);
            var result = await manager.Read(d);
        }
        catch (Exception e)
        {
            Console.WriteLine(e.Message);
        }
    }
    await delay;
}
