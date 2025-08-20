using Datafeel;
using Datafeel.NET.Serial;

/*
 * The low level API requires a DatafeelModbusClient to be provided to a HardwareDot object. A specific HardwareDot subclass corresponds to a specific device model.
 */

var commClient = new DatafeelModbusClientConfiguration()
    .UseWindowsSerialPortTransceiver()
    //.UseSerialPort("COM3") // Uncomment this line to specify the serial port by name
    .CreateClient();

var myDot = new Dot_63x_xxx(1);
await commClient.Open();

var random = new Random();

while (true)
{
    var delay = Task.Delay(250);

    myDot.LedMode = LedModes.GlobalManual;
    myDot.GlobalLed.Red = (byte)random.Next(0, 255);
    myDot.GlobalLed.Green = (byte)random.Next(0, 255);
    myDot.GlobalLed.Blue = (byte)random.Next(0, 255);
    try
    {
        using var writeCancelSource = new CancellationTokenSource(100);
        await myDot.WriteAllSettings(commClient, writeCancelSource.Token);
    }
    catch (Exception e)
    {
        Console.WriteLine(e.Message);
    }

    await delay;
}